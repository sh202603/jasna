# torchcodec バックエンド（実験的、vali / PyNvVideoCodec フォールバック付き）

`--video-backend` で選べる、torchcodec ベースの decode/encode 経路の設計をまとめる。

既存のネイティブ経路（`python_vali` デコーダと `PyNvVideoCodec` エンコーダ）はそのまま残す。
torchcodec が扱えるケースだけ torchcodec を使い、扱えないケースは自動的にネイティブへフォールバックする。
既定は `native` で、これは従来と完全に同一の挙動になる。
torchcodec はオプトインの実験的機能である。

**実行環境**：jasna 本体と同じ GPU 専用スタック（NVIDIA GPU、CUDA 13.x、torch 2.12.0+cu130、ffmpeg 8 の共有ビルドを PATH）に加え、optional 依存 `torchcodec>=0.14.0`（cu130 ビルド）を要する。
本書の数値は Windows 11 / Python 3.13.9 / torch 2.12.0+cu130 / CUDA 13.0 / RTX 5080 / ffmpeg 8.1 full-shared で計測した（付録 A は同一 RTX 5080 の Linux 側。デュアルブートで GPU は共通）。

## 背景と目的

jasna は decode を `python_vali`、encode を `PyNvVideoCodec` という二つの vendored ネイティブ依存に依存している。
どちらもビルドが重い。
これを PyTorch 公式の単一 wheel で完結する `torchcodec>=0.14.0` に寄せたい。

ネイティブのみが担う必要があるのは、実質 **10bit 出力**（torchcodec の GPU エンコードは 8bit nv12 固定）と、一部のマッピング不可な NVENC 設定、streaming、frame-gen に絞られる。
AV1 は `av1_nvenc` で出せ、色は既存の remux が付与し、NVENC 設定の多くは `extra_options` でマッピングできる。
10bit などでネイティブが必要なため、全面置換ではなく torchcodec 経路を**追加**し、非対応はネイティブへ**フォールバック**する方式を採る。

## torchcodec 0.14.0 の能力

| 区分 | 可否 | 詳細 |
|---|---|---|
| Decode (CUDA/NVDEC) | ✅ | GPU 上 `uint8` RGB NCHW テンソル。バッチ（`get_frames_in_range`）、exact seek、`pts_seconds`、BT.601/709 + HDR 色変換。出力は常に 8bit RGB。FFmpeg 4–8 対応。 |
| Decode: カスタム CUDA ストリーム注入 | ❌ | 内部ストリームを自前管理。外部ストリーム注入は非公開。 |
| Encode (CUDA/NVENC) | ✅ | CUDA フレームに対し `hevc_nvenc`/`av1_nvenc`（`codec` 省略時は自動選択）。増分は `torchcodec.encoders.Encoder`。`codec='hevc'` のような素の名前はソフトウェアエンコーダ（libx265）に解決され CUDA フレームを拒否するため、必ず `*_nvenc` を使う。 |
| Encode: 10bit / P010 | ❌ | GPU エンコードは画素フォーマットが 8bit `nv12` 固定（`pixel_format` 指定不可）。これが唯一の確たるゲート。 |
| Encode: AV1 | ✅ | `av1_nvenc` で出力可。 |
| Encode: NVENC 設定 | ◯（一部） | `extra_options` で受理されるキー（cq/qmin/qmax/gop→g/lookahead→rc-lookahead/temporalaq→temporal-aq/aq→spatial-aq+aq-strength/nonrefp→nonref_p/maxbitrate→maxrate/vbvbufsize→bufsize、preset は専用引数）はマッピングして適用。rc/tuning_info/bref/vbvinit/initqp/lookahead_level/tflevel は torchcodec が拒否する。 |
| Encode: 色メタデータ | △ | torchcodec 自体は color primaries/transfer/matrix を書かないが、既存の remux（`remux_with_audio_and_metadata`、HEVC は VUI 書換）が出力ストリームに付与する。 |

decode は jasna が必要とする 8bit RGB を全入力で供給できる。
encode の唯一の確たるゲートは 10bit で、10bit 要求はネイティブへフォールバックする。

## アーキテクチャ

### 既存パスの無劣化（最優先制約）

ネイティブ経路は、挙動も性能も従来と一致させる。

- factory は `backend == native`（および AUTO で torchcodec が不適合か利用不可）のとき、既存の `NvidiaVideoReader` と `NvidiaVideoEncoder` を直接返す。アダプタで包まない。
- 能力判定と factory はコンストラクタ時に一度だけ実行する。フレーム毎のホットパス（`frames()` と `encode()`）には分岐も委譲も属性アクセスも足さない。
- 各パイプラインの改修は、生成箇所を factory 呼び出しに差し替えるだけにとどめ、decode と encode のループ本体は変更しない。

### 選択レイヤ `jasna/media/backend.py`

両バックエンドを知る唯一のモジュールである。
モジュール冒頭で torch、torchcodec、vali、nvc を import しない（遅延 import）。
これにより、GPU もネイティブライブラリも無い環境で判定ロジックを単体テストできる。

- `VideoBackend(str, Enum)`：`AUTO` と `NATIVE` と `TORCHCODEC`。
- `torchcodec_available()`：`find_spec` のみで判定する（安価で、例外を投げず、結果をキャッシュする）。インストール済みでも import に失敗するケースは構築時に捕捉し、AUTO ではネイティブへフォールバックする。
- `select_decoder_backend(...)`：torchcodec の decode は全入力を賄えるため、ゲートは可用性だけにする。
- `torchcodec_encoder_eligibility(...) -> (bool, reason)`：純粋判定。次の全条件を満たすときだけ torchcodec を選ぶ。
  1. `stream_mode is False`
  2. `codec` が `hevc` か `av1`
  3. 実効 bit_depth が `8`（`bit_depth if not None else (10 if metadata.is_10bit else 8)`）
  4. `--encoder-settings` がマッピング可能なキーのみ（`encoder_settings_mappable`）
  5. `color_range` が limited（JPEG/full は不可）
  6. `output_fps_multiplier == 1`
  - 色空間（BT.601/709/2020）はゲートしない。色は remux が付与するため、torchcodec + remux の色処理はネイティブと同一になる。
- `make_video_reader(...)` / `make_video_encoder(...)`：factory。ネイティブは既存クラスを直接返す。AUTO は torchcodec の構築や初期化に失敗したらネイティブへフォールバックし、`torchcodec` 強制時は例外を再送出する。

### torchcodec デコーダ `jasna/media/torchcodec_decoder.py`

`TorchcodecVideoReader(file, batch_size, device, metadata)` を `NvidiaVideoReader` と同一契約で実装する。
契約は、context manager であること、`frames(seek_ts=None)` が `(uint8[B,3,H,W] CUDA, list[int] PTS)` を yield することである。

- `VideoDecoder(file, device=str(device), dimension_order="NCHW", seek_mode="exact")` で構築する。
- `get_frames_in_range(next, stop)` で逐次バッチ読みし、`fb.data` をそのまま yield する。
- PTS は `int(round(pts_seconds / float(metadata.time_base)))` で求める（CPU に出るのは B 個の float だけで、画素は触らない）。厳密単調増加になるよう、衝突時は +1 する。
- torchcodec は内部ストリームを使うため、各バッチ yield の前に `torch.cuda.synchronize(device)` を入れ、下流の推論にフレームを確実に渡す（ネイティブのストリーム同期と同周期）。
- フォールバック契機は、import 失敗、`__enter__` 失敗、CPU フォールバックの検知である。CPU フォールバックは最初のバッチの `fb.data.device` で判定する（`decoder.metadata` に device 属性は無い）。

### torchcodec エンコーダ `jasna/media/torchcodec_encoder.py`

`TorchcodecVideoEncoder(...)` を `NvidiaVideoEncoder` と同一コンストラクタおよび FrameWriter 契約（`__enter__` / `encode(frame, pts)` / `__exit__`）で実装する。
コンストラクタで適格性を再チェックし、不適合なら `TorchcodecEncodeUnsupported(ValueError)` を送出する。

- 増分エンコードに `torchcodec.encoders.Encoder`（複数ストリーム）を使う。codec は hevc→`hevc_nvenc`、av1→`av1_nvenc`。
- 符号化はネイティブと同じく専用ワーカースレッドで行う（採用判断は付録Aを参照）。`__enter__` がワーカーを起動し、ワーカーが `Encoder()` → `add_video(height, width, frame_rate, device='cuda', codec=...)` → `open_file(中間mkv)` を実行する。`encode(frame, pts)` は呼び出し側（BlendEncode スレッド）で現在ストリームに `torch.cuda.Event` を記録し、`(frame, event)` を上限つきキューに積むだけ。ワーカーが FIFO 順に `event.synchronize()`（torchcodec は内部 CUDA ストリームを自前管理し外部注入できないため host 同期）→ LUT 適用 → `add_frames` を行い、`__exit__` の stop で `close()` する。`Encoder` のライフサイクルはすべてワーカー上に置き、内部ストリームのスレッドまたぎを避ける。torchcodec は CFR で pts を無視するため、ネイティブの pts 並べ替えバッファは持たず単一 FIFO で投入順を保つ。
- `__exit__` は中間 mkv を `remux_with_audio_and_metadata(中間, 出力, metadata, codec)` に通す。音声、色タグ、HEVC の VUI 書換はこの remux が担い、ネイティブと同一の色処理になる。
- タイミングは `frame_rate`（`video_fps_exact`）の CFR。ネイティブの「raw elementary + mkvmerge による明示タイムコード」と違い任意の per-frame PTS は持たないため、VFR はネイティブに残す。
- NVENC 設定は `_build_encode_params` が jasna の既定品質（cq=25, preset p5, qmin/qmax, temporal-aq, rc-lookahead 32, spatial-aq + aq-strength 8, gop、hevc は bf=4）をマッピング可能キーで構築し、`--encoder-settings` を上書きして `add_video(preset=..., extra_options=...)` に渡す。これにより torchcodec 出力の品質がネイティブに近づく。

### フォールバック

AUTO のデコードでは、torchcodec の初期化（NVDEC オープン、CPU フォールバック検知）が `__enter__` 時に起きるため、`make_video_reader` は `_FallbackVideoReader` アダプタを返す。
このアダプタは `__enter__` で torchcodec を試し、失敗したらネイティブを構築して入る。
`frames`/`__exit__` は実際に入ったリーダへ委譲するだけで、フレーム毎の追加処理は無い。
`native` 指定時はアダプタを介さず既存リーダを直接返す。

## CLI

```
--video-backend  {native,auto,torchcodec}            # 既定: native
--decode-backend {inherit,native,auto,torchcodec}    # 既定: inherit（=--video-backend）
--encode-backend {inherit,native,auto,torchcodec}    # 既定: inherit
```

- `native`：常に vali と PyNvVideoCodec を使う（従来挙動）。
- `auto`：torchcodec が利用可能かつ適格なら使い、そうでなければネイティブを使う。
- `torchcodec`：torchcodec を強制する。満たせない要求（10bit、streaming など）は早期にエラーにする。
- `--decode-backend` と `--encode-backend` はデコード側とエンコード側を個別に上書きする。`inherit` は `--video-backend` に従う。

`jasna`（`main.py`）と `jasna-framegen`（`framegen_cli.py`）の両方に追加する。
早期バリデーションとして、`torchcodec`（decode/encode いずれか）と streaming の組み合わせ、`encode-backend torchcodec` と `--bit-depth 10` の組み合わせをエラーにする。
frame-gen は常に CFR エンコードがネイティブのため、`jasna-framegen` では強制 `torchcodec` のエンコード側を `auto` に降格させ、デコードだけ torchcodec を使う。

## 依存とインストール

`pyproject.toml` に optional extra として追加する（ネイティブのみのインストールを壊さない）。

```toml
[project.optional-dependencies]
torchcodec = ["torchcodec>=0.14.0"]   # 0.13.0 は既知バグで不可
```

torch 2.12 + cu130 と整合させるため、uv で cu130 wheel index から入れる。jasna 本体と一緒に extra で入れる方法（ビルドガイドと同じフラグ）:

```powershell
uv pip install -e .[dev,torchcodec] `
    --extra-index-url https://download.pytorch.org/whl/cu130 `
    --index-strategy unsafe-best-match `
    --prerelease=allow
```

既存環境に torchcodec だけ追加する場合（`--no-deps` で pinned torch を乱さない）:

```powershell
uv pip install "torchcodec>=0.14.0" --no-deps --index-url https://download.pytorch.org/whl/cu130
```

- torchcodec は実行時に FFmpeg の共有ライブラリを dlopen する。jasna が要求する **ffmpeg 8 共有ビルド**（`avcodec-62`/`avformat-62`/`avutil-60`）が PATH にあれば解決できる。
- Windows の DLL 検索では、`packaging/windows_dll_paths.py` の whitelist に `torchcodec` を追加済み（不在時は no-op）。`torchcodec_decoder.py`/`torchcodec_encoder.py` は import 前に win32 で `os.add_dll_directory` を行う。
- `__main__.py` の `_preload_native_libs()` には無条件追加しない（optional のため）。
- 注意：Windows + cu130 + Python 3.13 の組み合わせには torchcodec の DLL ロード失敗の既往がある（torchcodec issue #1233）。検証環境（RTX 5080 / cu130 / ffmpeg 8.1）では再現しなかったが、`torchcodec_available()` は spec の有無のみを見るため、import 失敗は構築時に捕捉して AUTO はネイティブへ落とす。

## 性能と検証

実 1080p（31524 フレーム、rfdetr、復元あり、threaded encode）での native と torchcodec の比較は次のとおり（環境は緒言を参照。RTX 5080 / Windows）。

| 計測 | native | torchcodec | 備考 |
|---|---|---|---|
| デコード単体 | 2463 fps | 2164 fps | native が約 12% 速い（素材依存。合成では torchcodec が速いこともある） |
| エンコード単体（hevc 8bit、mux 込み） | 147 fps | 198 fps | torchcodec が約 1.35 倍（native は mkvmerge 明示タイムコード等も行う） |
| フル尺 総時間（復元あり） | 約 265 秒 | 約 258 秒 | ほぼ同等（差は試行間ばらつきの範囲。lada-yolo でも native 208秒 / torchcodec 201秒で同傾向） |
| 出力ビットレート（cq=25 等価設定） | 約 5.93 Mbps | 約 6.12 Mbps | torchcodec が約 +3.3%（エンコーダが違うため同一名目設定でもレート点がずれる） |
| 出力 SSIM（native vs torchcodec、全 31524 フレーム） | （基準） | All 0.954 / Y 0.937 | chroma U/V は約 0.99。下記「出力等価性」参照 |

デコードとエンコードのスループットは逆方向だが、いずれもパイプライン全体の数 % にすぎず、フル尺の総時間はバックエンドに実質依存しない（復元と検出が支配的）。
実モザイクを含むフル尺で検出 → 追跡 → 復元 → NVENC → remux が正しく動作し、モザイク領域は両 backend で復元され、フレーム数も一致する。ただし出力は native とビット等価ではなく、SSIM は All 0.954 / Y 0.937 である（次節）。

### 出力等価性とパイプラインの決定性

native パイプラインは決定的である。同一条件で2回流すと出力の動画ビットストリームは完全一致する（`ffmpeg -map 0:v -c copy -f md5` のパケット md5 一致、SSIM 1.0、ファイルサイズもバイト一致）。MKV のファイル単位 md5 は mkvmerge がランダムな segment-UID を書くため動画が同一でも変わるので、等価判定はパケット md5 か SSIM で行い、ファイルハッシュでは行わない。

native が決定的である以上、native と torchcodec の SSIM 0.954（All）/ 0.937（Y）は**パイプライン非決定性ではなく torchcodec 固有の差分**である。差は検出器に依らず（lada-yolo 0.9543 / rfdetr 0.9542）全フレームでほぼ一様で、主因は**デコード画素差**（vali の明示 BT 変換 + ディザ vs torchcodec の内部変換。本書「色」参照。このパイプラインは非モザイク領域をデコード元のまま blend するため、画素差がフレーム全面に乗る）であり、encode 差（+3.3% bitrate）は二次的である。視覚的な破綻は確認されていないが（chroma 0.99）、出力はビット等価ではなく忠実度に一段の差があり、許容可否は用途で判断する。

> 旧版はこの SSIM を 0.9904 としていたが、同一 RTX 5080・同一クリップでの統制された再計測（全フレーム、native は決定的）では再現せず、実測値 0.954 に訂正した。

## 制限と注意点

- **デコードフレームの連続性（要 `.contiguous()`）**：torchcodec は NCHW を内部 HWC バッファの非連続ビューで返す（stride が C/W で交互）。検出の TensorRT エンジンは入力を連続 NCHW 前提で生メモリから読むため、非連続のまま渡すとガベージを読み**モザイクを 1 つも検出しない**（→ 復元スキップ）。`TorchcodecVideoReader` は `fb.data.contiguous()` で連続化してから yield する。バッチが満杯（パディング無し）のときだけ顕在化するため見落としやすく、`tests/test_torchcodec_decoder.py` の `is_contiguous()` アサーションで回帰を防ぐ。
- **10bit**：torchcodec の GPU エンコードは 8bit nv12 固定。10bit 要求はネイティブが担う。
- **VFR**：torchcodec エンコードは CFR（`video_fps_exact`）。任意の per-frame PTS を持たないため、VFR はネイティブに残す。
- **マッピング不可の NVENC 設定**：rc/tuning_info/bref/vbvinit/initqp/lookahead_level/tflevel を `--encoder-settings` で指定した場合はネイティブに回す。
- **色**：vali は明示的な BT.x 変換、range 変換、ディザを行うが、torchcodec は内部変換を使うため、デコード結果の画素はビット完全一致ではない（合成クリップの先頭フレームで平均絶対差 約 8.5）。出力の色メタデータは remux が付与するためネイティブと同一。
- **デコード性能**：実 1080p ではネイティブが約 12% 速い（素材依存、デコードは非ボトルネック）。
- **フレーム数（出力には波及せず）**：ネイティブ vali は raw デコードで数枚過剰に返す（ground truth 48 に対し 50、500 枚に対し 504 と可変）。torchcodec は常に正しい。実パイプラインの出力枚数は両者とも正しく、差は無い。
- **符号化スレッド化の判断（採用）**：torchcodec の符号化はネイティブと同じく専用ワーカースレッドで実行する。理由は、同期実行だと重い検出（rfdetr）で符号化がブレンド符号化段に露出して二重ボトルネック化し、ネイティブに負けるため（付録A）。スレッド化で rfdetr はネイティブ同等（192秒）に戻る。トレードオフとして軽い検出（lada-yolo）は 100秒から114秒へ回帰するが、それでもネイティブ（124秒）には勝つ。両検出器でネイティブ以上を保てることを優先して threaded を既定とした。出力はスレッド化前後で不変（ネイティブとの SSIM 0.954 も変わらない）。

## テスト

- `tests/test_video_backend_select.py`（GPU 不要）：デコーダ判定と、エンコーダ適格判定の全マトリクス（codec/bit-depth/設定マッピング可否/色域/frame-gen/streaming）。
- `tests/test_video_backend_fallback.py`（GPU 不要）：`_FallbackVideoReader` の `__enter__` 失敗時フォールバックと、factory の振り分け。
- `tests/test_torchcodec_decoder.py`（GPU ガード）：出力契約、PTS 厳密増加、ネイティブとの PTS 一致、画素近接、フレーム数の正しさ。
- `tests/test_torchcodec_encoder.py`（GPU ガード）：hevc/av1 の round-trip を ffprobe 検証、設定マッピングの構築、不適合要求の拒否。
- `tests/test_perf_regression.py`（GPU、`-m perf`）：壁時計（前後で `torch.cuda.synchronize()`）でネイティブと torchcodec のデコード スループットを比較。
- `tests/test_main_validation.py`：CLI ガード（torchcodec + streaming、encode torchcodec + 10bit、av1 許可）。
- 既存テスト：decode/encode の生成シームが factory に移ったため、`test_pipeline_threads.py` と `test_pipeline_run*.py` の patch 先を `make_video_reader`/`make_video_encoder` に更新済み。

## 付録A: 検出器による優劣の違い

検出を軽い lada-yolo と重い rfdetr に替えて、ネイティブと torchcodec を比較した。
測定は Linux / RTX 5080 / secondary なし / 1080p 31524 フレームで行った（本体の表は Windows / rfdetr で、OS も検出器も異なる）。
バックエンドの優劣は検出器によって逆転する。

```
                        lada-yolo               rfdetr
                     ネイティブ torchcodec   ネイティブ torchcodec
総時間(壁時計)          124s      101s          192s      197s
decode-detect 合計     123.9s    100.8s        191.8s    196.6s
blend-encode 実作業     70.1s     101.6s        150.5s    197.3s
  うち write           0.6s      60.2s          1.0s      112.6s
```

lada-yolo では torchcodec が約18%速く、rfdetr ではネイティブが約3%速い。
rfdetr で `--video-backend auto` は torchcodec を選ぶため（8bit HEVC は適格）、auto も198秒でネイティブに劣る。

### 内訳が backend 間で比較できない理由

タイミングは CPU の壁時計で積算しており、GPU が非同期に走るため、カテゴリの境目で何を同期するかで振り分けが変わる。
ネイティブはバッチ末にデコード用ストリームだけを、torchcodec はバッチごとにデバイス全体を同期する（`torch.cuda.synchronize(device)`）。
このため decode と detect の内訳は backend 間で比較できず、信頼できるのは段の合計と総時間である。

### なぜ lada-yolo では torchcodec が速いのか

検出が軽いと、デコード検出段がボトルネックになる。
本体のデコード単体ベンチではネイティブが約12%速い（2463 fps 対 2164 fps）のに、この段の合計は torchcodec が速い（123.9秒 対 100.8秒）。
単体で速いネイティブが合算で負けるのは、ネイティブのデコードが検出と SM を奪い合うためである。
ネイティブはデコード後に色変換を2パス行い、これらの SM カーネルが検出の TensorRT カーネルと競合する。
torchcodec は色変換が軽く、検出に回る SM が増えるので、ボトルネック段が23秒縮む。

### なぜ rfdetr では逆転するのか

検出が重いと、この SM 競合の解消が効かない。
rfdetr は GPU をほぼ飽和させるので、デコードが SM を空けても検出がすぐ埋め、デコード検出段の合計は torchcodec がむしろ微増する（191.8秒 対 196.6秒）。
torchcodec が連続化のためにかけるバッチごとのコピーとデバイス全体同期は、飽和した GPU では隠れず、わずかな上乗せになる。

決定打は符号化である。
torchcodec の符号化はブレンド符号化スレッド上で同期実行され、SM を使う前処理を伴う。
同じ符号化でも、GPU が空いている lada-yolo では60.2秒だが、飽和した rfdetr では112.6秒に膨らむ。
このためブレンド符号化段が197.3秒まで上がり、デコード検出段と並ぶ二重ボトルネックになって、総時間がネイティブを上回る。

### 優劣を決める要因

ネイティブの符号化は別スレッドへ渡すだけで、検出器によらず裏に隠れる（write は約1秒）。
torchcodec の符号化は同期実行で表に出るため、ブレンド符号化段がデコード検出段のボトルネックの下に収まるかどうかで結果が決まる。
lada-yolo ではデコード検出段が101秒へ下がり、符号化を含むブレンド符号化段が101.6秒でぎりぎり収まったので、デコードの短縮がそのまま勝ちになった。
rfdetr ではデコード検出段が下がらず、同期実行の符号化がブレンド符号化段を197秒へ押し上げたので、デコード側の利得が無いまま負けた。

### 符号化のスレッド化とその効果（実装・実測）

上の分析を受けて、torchcodec の符号化をネイティブと同じく専用ワーカースレッドへ移した（`encode()` は producer ストリームにイベントを記録してキューへ積むだけ、ワーカーが `event.synchronize()` 後に `add_frames`）。
出力は不変で、スレッド化前後で encode 結果は変わらず、ネイティブとの SSIM も 0.954 のままである。

効果は検出器で割れた（RTX 5080 / Linux、同一セッションの A/B 計測）。

```
                     ネイティブ  torchcodec inline  torchcodec threaded
lada-yolo  総時間       124s        100s              114s
rfdetr     総時間       192s        197s              192s
```

rfdetr では狙いどおり、ブレンド符号化段の write が112秒から約7秒に落ち、段が decode-detect の下に収まって総時間が192秒（ネイティブと同等）になった。
lada-yolo では逆に100秒から114秒へ回帰した。
decode-detect 段が100.3秒から114.3秒へ上がっており（decode/detect 両方が約7秒増）、検出が軽く GPU に余裕がある場合は、ワーカーの並行エンコードが検出と SM を取り合ってボトルネック段を押し上げる。
inline では符号化がブレンドスレッドの歩調で自然に詰まっていたぶん、かえって効率がよかったということである。

トレードオフだが、threaded は両検出器でネイティブ以上を保つ（rfdetr で同等、lada-yolo で114秒 < 124秒）ため、torchcodec の既定として threaded を採用した。

### OS 間の差（同一 RTX 5080・デュアルブート）

本体の「性能と検証」（Windows）と本付録（Linux）は**同一 RTX 5080 のデュアルブート**で計測している。したがって OS 間の差はハードではなく OS / ソフトスタック起因である。

同一クリップ・同一コード（threaded）でのフル尺 総時間:

```
              Linux     Windows(同一5080)
lada-yolo     ~124s     ~208s   (+68%)
rfdetr        ~192s     ~265s   (+38%)
```

Windows が 34〜68% 遅く、差は **launch-bound な TRT 段（検出と BasicVSR++ 復元）に集中**する（固定機能の decode/encode は据え置き）。検出が軽く launch 本数の多い lada-yolo ほど悪化幅が大きい。HAGS は ON でこのレバーは使い切っており、残差は WDDM の submit overhead と、`.win.engine`（OS 別の TRT エンジンビルド）が候補。対策は検出・復元の CUDA Graph 化と TRT バージョンの OS 間統一。

なお Windows threaded での native vs torchcodec は、lada-yolo が 208秒 対 201秒、rfdetr が 265秒 対 258秒で、**両検出器とも torchcodec ≤ native**（差は試行間ばらつきの範囲）。よって「重い検出器で auto が torchcodec を選んで悲観化する」懸念は、threaded 化後は Linux でも rfdetr 同等（192=192）であり、Windows では再現しない。inline 時代（本付録の最初の表）に固有の注意だった。

## 付録B: 優劣が GPU に依存する理由

torchcodec の優位が出るかどうかは GPU のエンジン構成に依存する。
ただし機序は「torchcodec のデコーダが強い」ではない。
デコードはどちらの backend も同じ NVDEC ハードウェアを使うため、その層に差はない。

### デコーダを2個動かす構成と NVDEC 数

このパイプラインはデコーダを2個同時に動かす。
デコード検出段の読み出しと、符号化段が原フレームを読み直す読み出しである。
RTX 5080 は NVDEC を2基持つので、2つのデコーダを別々のエンジンに載せられ、デコード同士は競合しない。

GeForce RTX 50系のエンジン数は次のとおり（SM は CUDA コア数を128で割った概算）。

```
GPU            NVENC  NVDEC   SM
RTX 5090         3      2     170
RTX 5080(測定)   2      2      84
RTX 5070 Ti      2      1      70
RTX 5070         1      1      48
RTX 5060 Ti      1      1      36
RTX 5060         1      1      30
```

RTX 5060 と RTX 5060 Ti は NVDEC が1基しかない。
2つのデコーダが1基を共有するため、デコード同士がエンジンの取り合いを起こす。
この制約はネイティブと torchcodec の両方にかかる。

### 飽和がもたらす逆転（付録Aからの含意）

rfdetr が RTX 5080 の SM を飽和させたときの挙動は、弱い GPU で軽い検出器を回したときの近似になる。
付録Aのとおり、飽和下では torchcodec の SM 競合の解消が効かず、同期実行の符号化が表に出て（112秒）、ネイティブに負けた。
弱い GPU は同じ負荷でも早く飽和するため、RTX 5060 では lada-yolo のような軽い検出でも rfdetr-on-5080 に近い振る舞いになりうる。

### RTX 5060 系での予測

弱い GPU で torchcodec が勝てるかは、SM の空きと符号化の露出の釣り合いで決まる。
SM 競合の解消が効くのは、検出が GPU を飽和させていないときに限る（付録Aで実測）。
RTX 5060 は SM が少なく早く飽和するので、この利得は出にくい。
さらに NVDEC が1基で2つのデコーダが競合し、同期実行の符号化が SM を奪われて膨らむ。
これらは torchcodec に不利な方向で重なる。

したがって「デコーダが弱いと恩恵が薄い」という見立ては、方向としては当たっているが、原因はデコーダ単体の性能ではない。
NVDEC が1基で2つのデコーダが競合し、SM 不足で検出と符号化の釣り合いが崩れることが機序である。
弱い GPU では、軽い検出でも torchcodec の優位が縮むか消える公算が高いが、最終的な優劣は釣り合い次第で、実測しないと決まらない。

### 予測を実測で確かめる

1. RTX 5060 系の実機で同じ条件を回し、段ごとの時間を比較する。特に torchcodec の符号化時間がデコード検出段を超えるかを見る。
2. GPU の各エンジンの利用率を取得し（`nvidia-smi dmon` など）、NVDEC が先に飽和するか SM が先かを判定する。
3. 符号化のスレッド化（付録Aで実装済み）が弱い GPU でどう出るかを見る。RTX 5080 では rfdetr を改善し lada-yolo を回帰させたので、SM が少なく早く飽和する 5060 系では、軽い検出でも回帰側に振れる可能性がある。

出典:
- [Video Encode and Decode GPU Support Matrix (NVIDIA Developer)](https://developer.nvidia.com/video-encode-decode-gpu-support-matrix)
- [NVIDIA RTX Blackwell GPU Architecture (PDF)](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
