# torchcodec バックエンド（実験的、vali / PyNvVideoCodec フォールバック付き）

`--video-backend` で選べる、torchcodec ベースの decode/encode 経路の設計をまとめる。

既存のネイティブ経路（`python_vali` デコーダと `PyNvVideoCodec` エンコーダ）はそのまま残す。
torchcodec が扱えるケースだけ torchcodec を使い、扱えないケースは自動的にネイティブへフォールバックする。
既定は `native` で、これは従来と完全に同一の挙動になる。
torchcodec はオプトインの実験的機能である。

**実行環境**：jasna 本体と同じ GPU 専用スタック（NVIDIA GPU、CUDA 13.x、torch 2.12.0+cu130、ffmpeg 8 の共有ビルドを PATH）に加え、optional 依存 `torchcodec>=0.14.0`（cu130 ビルド）を要する。
本書の数値は Windows 11 / Python 3.13.9 / torch 2.12.0+cu130 / CUDA 13.0 / RTX 5080 / ffmpeg 8.1 full-shared で計測した。

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

- 増分エンコードに `torchcodec.encoders.Encoder`（複数ストリーム）を使う。`__enter__` で `add_video(height, width, frame_rate, device='cuda', codec=...)` → `open_file(中間mkv)`、`encode` で LUT 適用後に `VideoStream.add_frames(frame.unsqueeze(0))`、`__exit__` で `close()`。codec は hevc→`hevc_nvenc`、av1→`av1_nvenc`。
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

実 1080p（31524 フレーム、rfdetr、復元あり）での native と torchcodec の比較は次のとおり（環境は緒言を参照）。

| 計測 | native | torchcodec | 備考 |
|---|---|---|---|
| デコード単体 | 2463 fps | 2164 fps | native が約 12% 速い（素材依存。合成では torchcodec が速いこともある） |
| エンコード単体（hevc 8bit、mux 込み） | 147 fps | 198 fps | torchcodec が約 1.35 倍（native は mkvmerge 明示タイムコード等も行う） |
| フル尺 総時間（復元あり） | 約 284 秒 | 約 275 秒 | ほぼ同等 |
| 出力 SSIM（native vs torchcodec） | （基準） | 0.9904 | 両方とも正しく復元 |

デコードとエンコードのスループットは逆方向だが、いずれもパイプライン全体の数 % にすぎず、フル尺の総時間はバックエンドに実質依存しない（復元と検出が支配的）。
実モザイクを含むフル尺で検出 → 追跡 → 復元 → NVENC → remux が正しく動作し、出力は native と SSIM 0.990 で等価、フレーム数も一致する。

## 制限と注意点

- **デコードフレームの連続性（要 `.contiguous()`）**：torchcodec は NCHW を内部 HWC バッファの非連続ビューで返す（stride が C/W で交互）。検出の TensorRT エンジンは入力を連続 NCHW 前提で生メモリから読むため、非連続のまま渡すとガベージを読み**モザイクを 1 つも検出しない**（→ 復元スキップ）。`TorchcodecVideoReader` は `fb.data.contiguous()` で連続化してから yield する。バッチが満杯（パディング無し）のときだけ顕在化するため見落としやすく、`tests/test_torchcodec_decoder.py` の `is_contiguous()` アサーションで回帰を防ぐ。
- **10bit**：torchcodec の GPU エンコードは 8bit nv12 固定。10bit 要求はネイティブが担う。
- **VFR**：torchcodec エンコードは CFR（`video_fps_exact`）。任意の per-frame PTS を持たないため、VFR はネイティブに残す。
- **マッピング不可の NVENC 設定**：rc/tuning_info/bref/vbvinit/initqp/lookahead_level/tflevel を `--encoder-settings` で指定した場合はネイティブに回す。
- **色**：vali は明示的な BT.x 変換、range 変換、ディザを行うが、torchcodec は内部変換を使うため、デコード結果の画素はビット完全一致ではない（合成クリップの先頭フレームで平均絶対差 約 8.5）。出力の色メタデータは remux が付与するためネイティブと同一。
- **デコード性能**：実 1080p ではネイティブが約 12% 速い（素材依存、デコードは非ボトルネック）。
- **フレーム数（出力には波及せず）**：ネイティブ vali は raw デコードで数枚過剰に返す（ground truth 48 に対し 50、500 枚に対し 504 と可変）。torchcodec は常に正しい。実パイプラインの出力枚数は両者とも正しく、差は無い。

## テスト

- `tests/test_video_backend_select.py`（GPU 不要）：デコーダ判定と、エンコーダ適格判定の全マトリクス（codec/bit-depth/設定マッピング可否/色域/frame-gen/streaming）。
- `tests/test_video_backend_fallback.py`（GPU 不要）：`_FallbackVideoReader` の `__enter__` 失敗時フォールバックと、factory の振り分け。
- `tests/test_torchcodec_decoder.py`（GPU ガード）：出力契約、PTS 厳密増加、ネイティブとの PTS 一致、画素近接、フレーム数の正しさ。
- `tests/test_torchcodec_encoder.py`（GPU ガード）：hevc/av1 の round-trip を ffprobe 検証、設定マッピングの構築、不適合要求の拒否。
- `tests/test_perf_regression.py`（GPU、`-m perf`）：壁時計（前後で `torch.cuda.synchronize()`）でネイティブと torchcodec のデコード スループットを比較。
- `tests/test_main_validation.py`：CLI ガード（torchcodec + streaming、encode torchcodec + 10bit、av1 許可）。
- 既存テスト：decode/encode の生成シームが factory に移ったため、`test_pipeline_threads.py` と `test_pipeline_run*.py` の patch 先を `make_video_reader`/`make_video_encoder` に更新済み。
