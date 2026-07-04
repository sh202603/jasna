# 変更点サマリ: `modi`（`v0.7.2+modi`）vs upstream main

このブランチが upstream（`main`）に対して加えている変更の要約です。`modi` は upstream main `278ab09`（`v0.7.2`）にリベース済み。

- ブランチ: `modi`（fork `https://github.com/sh202603/jasna.git`）
- バージョン: `0.7.2+modi`（upstream main `278ab09` ベース = `v0.7.2`）
- 規模: upstream main 上に 6 コミットへ集約（`git diff --shortstat upstream/main..modi`）。

> **ベースを `v0.7.2` へ更新。** 以下のコミット別の突き合わせ記録は当初の `v0.6.2` 期リベース向けに書かれたものだが、それらの upstream 修正（デコーダ stream 同期、分離畳み込み、validate-model-name、onnx export、trt load）は現在のベースにも引き継がれており（upstream は v0.7.0 リリース時に `main` を force-push して SHA を書き換えたが、機能は維持）、収束関係は今も成立する。`v0.7.1` ベースで既に入っていたのは upstream の **サポーター向けモデル**（SD 1.5 画像復元、unet-4x）とモデル暗号化/Nuitka 周りで、本フォークはそのコードを **inert（動作しない形）** のまま同梱する（公開ソースからは復号も実行もできない。README のスコープ注記を参照）。ビルドは upstream に合わせて PyInstaller から Nuitka へ移行したが、パッケージングツールは非公開のため、公開での利用経路はソースからの実行（`docs/BUILDING_*` を参照）。
>
> **`v0.7.2` ベースでの新規分**（2026-06-18 取り込み）: 上流の**エクスポート後アクション**（`--post-export-action shutdown|command`, `--post-export-command`）、**検出モデルレジストリ / モデルファクトリ**（`jasna/mosaic/detection_registry.py`）、**出力パターン**のファイル名テンプレート（`--output-pattern "{original}_restored.mp4"`）、CLI/GUI のフォルダ処理改善（画像+動画を合算した `[i/N]` カウンタ）、起動時間の最適化（`restorer/__init__.py` の遅延 import）、GUI/Windows 修正。いずれも上記の inert なサポーターモデル同梱と競合せず、サポーターの `jasna/protection` サブモジュールのポインタも追従（`827c6cb → 1506724`、公開フォークでは空のまま）。modi 独自の CLI フォルダ毎ファイルバナーは上流に**吸収された**（`2f3a87e`、より良い合算カウンタ付き）ため、本リベースで drop した（§6 参照）。

> upstream に取り込まれた変更はこのリベースで drop 済み: **分離畳み込みブラー**（§3、upstream `f8c4048`）、**torch 2.12 スタック**、そして `v0.6.2` で新たに入った **issue #2 のカクつきを直すデコーダの stream 同期**（`cf33bcf` "sync decoder cuda stream"、§5）。これにより、本ブランチ独自のデコーダ同期バリアと opt-in の `--decoder nvdec` バックエンドは**冗長になったため削除**し、別系統の一次リストアのバッファ保持修正のみを残した（§5）。**BT.601 カラースペース**も upstream が独自実装したため突き合わせ済み（§2）。upstream `v0.6.2` は **HAGS チェックを削除**し、**`onnx` を依存に追加**している。
>
> `v0.6.2` 後の upstream 3 コミット（2026-06-09 取込）も同様に突き合わせ済み: `b55b501`「validate model name」は modi 未改変ファイルのためそのまま取込。`5b3ca34`「ultralytics onnx export fix」（`CUDA_VISIBLE_DEVICES` の退避/復元）は本ブランチの YOLO CPU export 修正（§1）と収束。upstream の退避/復元を採用しつつ、フローズンビルド対策の `half=False, device="cpu"` は本ブランチ側を維持。`6545b78`「linux trt load fix」（nvvfx より先に pip `tensorrt_libs` をプリロード）は本ブランチの既存修正（§1）と収束。関数名は upstream の `_preload_tensorrt_runtime` を採用し、実装は本ブランチの防御版（`find_spec` + try/except、Windows は DLL ロード順で対応済みのため no-op）を維持。upstream 追加のテストは両ファイルとも modi 実装に対してそのまま通る。

変更は大きく以下の層に分かれます。

---

## 1. 基盤（ビルド/ランタイムの改修）

upstream main に対するビルド環境とランタイムの更新。いずれも**ブランチのソースへ直接適用済み**で、クローン後そのままビルドできる（唯一の例外は venv に当てる mmengine パッチ、後述）。

- **ビルド/ランタイム改善**
  `model_weights/` の自動解決（`jasna/model_weights_resolver.py` 新規）。upstream も frozen バイナリ向けの `engine_paths.model_weights_dir()` を追加したため、upstream の呼び出し箇所を本リゾルバへ委譲して統一（env 上書き、パッケージ親フォールバック、ロギングを持つ superset を維持）。ほか TRT ベンチの新 API 追従、Windows 専用 DLL ロード補助、PyInstaller spec の調整など。
- **YOLO 検出エクスポート修正**: 検出 ONNX を CPU でエクスポートし、凍結（PyInstaller）バイナリで CUDA error 100 を回避する。ultralytics の GPU エクスポートは TensorRT より先に torch CUDA を初期化して初期化順序を壊すため。エクスポート後に `CUDA_VISIBLE_DEVICES` を復元。エンジンは引き続き fp16 でビルドされる。（`onnx` 自体は upstream の依存になった。）
- **Linux GUI / RTX-VSR 修正**
  モーダルダイアログの空表示修正、RTX Super-Res の TensorRT バージョン衝突（`libnvinfer.so.10` の先読み）修正。
- **ビルドガイド整備**: `docs/BUILDING_{LINUX,WINDOWS}_{ja,en}.md`。本ブランチはソース適用済みのためクローン後そのままビルド可。各ガイドの付録は変更内容の内訳を参照用に記述。

> mmengine 用の `patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff` は venv 内パッケージへ当てるもので、ビルド手順（各ガイド §8.1）で別途適用します（これが唯一同梱されるパッチ）。

---

## 2. 新機能（modi）: 出力フォーマットの柔軟化

従来 HEVC / 10bit(P010) / BT.709 固定だった GPU エンコード出力を、lada-ex 相当に拡張。

> upstream も独自に **BT.601 対応**（`chw_rgb_to_p010_bt601_limited` の狭い実装 + HEVC VUI 書き換え）と **`.cube` LUT** 機能を追加した。リベースでは本ブランチの汎用 superset（`chw_rgb_to_surface`、BT.601/709/2020 × NV12/P010）を維持しつつ、**upstream の HEVC VUI ビットストリーム書き換え（`-bsf:v hevc_metadata`、AV1 では適用しない）と `.cube` LUT を取り込んで共存**させた。

- **AV1 出力**: `--codec av1`（ファイル出力のみ。ストリーミング非対応、NVENC AV1 は B フレーム無効）。
- **8bit(NV12) / 10bit(P010) 出力**: `--bit-depth {auto,8,10}`。既定 `auto` はソース連動（10bit→P010、それ以外→NV12）。
- **BT.601 / BT.709 / BT.2020 色空間保持**: ソースの色空間に応じた limited-range RGB→YUV 変換を選択し、出力コンテナにマトリクス＋原色＋transfer を付与（BT.2020 は `bt2020nc` / `bt2020` / `bt2020-10`）。

### 実装の要点
- `jasna/media/rgb_to_p010.py`: 汎用変換 `chw_rgb_to_surface(frame, colorspace, bit_depth)`（NV12/P010 × BT.601/709/2020）。
- `jasna/media/__init__.py`: `Colorspace` enum と `VideoMetadata.yuv_colorspace`（av の Colorspace では表現できない BT.2020 を保持）。ffprobe からの色空間判定を拡張。
- `jasna/media/video_encoder.py`: codec＋ビット深度から `fmt`/`profile`/B フレーム/一時ファイル拡張子（`.hevc`/`.obu`）を決定。remux の色メタデータを matrix/primaries/transfer で正しく付与。出力コンテナは出力拡張子で決定（`.mkv`→Matroska、`.mp4`/`.mov`→MP4/MOV、AV1-in-MP4 可）し、mkvmerge で中間化後 ffmpeg で最終 remux（音声結合、`-c:v copy`、`.mp4`/`.mov` は `+faststart`）。
- `jasna/pipeline.py` / `jasna/streaming_pipeline.py`: 早期ガードを「BT.709 以外を拒否」→「フルレンジのみ拒否」に変更（BT.601/709/2020 を全許可）。フルレンジ判定は ffprobe の `color_range` が `pc` または `jpeg`（ffprobe は `pc` を報告するため、`pc` を取りこぼさないよう修正済み）。
- CLI（`jasna/main.py`）と GUI（`jasna/gui/`）に `--codec` / `--bit-depth` を追加。
- ドキュメント: `docs/CODECS_AND_COLORSPACE_{ja,en}.md`、各 README に1行追記。
- テスト: `tests/test_rgb_to_surface.py` 新規ほか、AV1/NV12/BT.601/BT.2020、バリデーションのケースを追加。

### 実機検証（ffmpeg 8 / mkvmerge v97）
HEVC 8/10bit（709）、BT.601、BT.2020、AV1（8/10bit）、`--bit-depth` 双方向 override をすべて確認（出力の codec/pix_fmt/色タグを ffprobe で検証、デコード成功）。追加で **10bit BT.601/709/2020 → P010**、**HEVC 10bit の `Main 10` プロファイル**、**`.mp4` への AV1-in-MP4 出力**、**フルレンジ(`pc`)入力の拒否発火** も確認。ユニットテストは全パス（フルレンジ `pc` 検出の回帰テスト `test_color_range_pc` を追加）。上記マトリクス（`--codec` × `--bit-depth` × BT.601/709/2020、`--bit-depth` 双方向、フルレンジ拒否）は **Windows / Linux 両環境**（RTX 5060 Ti）で、**CLI、GUI 連続処理、凍結(PyInstaller)バイナリ**にて確認済み（lada-yolo 検出 + RTX Super-Res、BT.2020 も VLC で色破綻なしを含む）。

### 既知の制限
- AV1 はファイル出力のみ（ストリーミング不可）。
- HDR トランスファ特性（PQ/HLG）は非保持（マトリクス＋原色のみ）。
- フルレンジ（JPEG/PC range）入力は非対応（検出時に拒否）。
- AV1 muxing は OBU→mkvmerge（中間段）→ ffmpeg remux（最終）。古い mkvmerge では IVF 化や ffmpeg 直 mux が必要な場合あり。

---

## 3. パフォーマンス修正: ブレンドマスク生成の分離畳み込み化（upstream 取り込み済み）

> **本項は本ブランチ独自の差分ではない:** upstream `v0.6.2` が同等実装（`f8c4048` "Reapply separable conv"）を持つため、リベース時に重複コミットは drop 済み。以下は経緯の記録。

upstream の `create_blend_mask`（`jasna/tracking/blending.py`）は、フレーム高に比例した大きな一様カーネル（1080p で 61×61、4K で 121×121、超解像時はさらに増大）の密な `conv2d` box blur を **2回**（dilation + falloff）実行していた。`O(K²)` のため解像度や検出数に応じて重くなる。一様カーネルは分離可能（`K×K = (1×K) ⊛ (K×1)`）なので 1D 畳み込み 2 パスに置換すると `O(K²)` → `O(2K)`、出力はビット同等。本ブランチと upstream はこれを独立に、ほぼ同一アルゴリズムで実装したため、リベースでは upstream 版を採用した。

---

## 4. 新機能（modi）: フレームレート倍化（フレーム生成）

AI補間フレームを挿入して出力フレームレートを上げる: `--frame-gen {none,2x,4x}`（ファイル出力のみ。AV1 と同様にストリーミング非対応）。

- **なぜ二次リストアではなく新ステージか**: 二次リストア（`unet-4x` / `tvai` / `rtx-super-res`）は 256×256 のモザイククロップを処理しフレーム数を変えない。フレーム生成は全解像度の出力フレームに対しフレーム数と PTS を**増やす**ため、パイプラインの `FrameWriter` を薄くラップするデコレータ（`FrameGenWriter`）として挿入する。パイプライン本体とエンコーダは無改造。
- **バックエンド**（`--frame-gen-backend {rife,rtx}`）。`FrameGenerator` プロトコルで差し替え可能:
  - `rife`（既定）: ニューラル補間（RIFE）。CUDA で**現在利用可能**。TorchScript チェックポイント（推奨、自己完結）か、同梱 RIFE 4.6 `IFNet`（`jasna/models/rife/`）への state_dict を読み込む。重みは `--frame-gen-model-path` か `model_weights/rife.pth`。
  - `rtx`: **NVIDIA RTX Video Frame Generation**。RTX Spark と同時に Python ホイール + ComfyUI ノードとして発表されたが、`nvidia-vfx`（1.2.0 は `VideoSuperRes` のみ公開）に**未出荷**。アダプタ（`jasna/framegen/rtx_frame_generator.py`）は将来の Effect を探し、出荷までは明示エラーにする。出荷後は推論呼び出し 1 箇所で有効化できる。
- **PTS 計算**（`FrameGenWriter`）: 連続する 2 フレームごとに、実フレームを出力後 `M-1` 枚の補間を `pts_k = prev_pts + round((curr_pts - prev_pts) * k / M)` で挿入。出力タイミングは PTS 駆動（mkvmerge タイムコード）なので、PTS 挿入が 2x/4x を生む。音声は元のタイムコードを保持するため尺と同期は不変。総フレーム数は `(N-1)*M + 1`。非単調 PTS の区間は補間をスキップ。NVENC の `fps`/`gop` は倍率でスケールし GOP/レート制御を正す（`NvidiaVideoEncoder(output_fps_multiplier=...)`）。
- CLI（`jasna/main.py`）と GUI（`jasna/gui/`）に `--frame-gen` / `--frame-gen-backend` を追加。
- スタンドアロン `jasna-framegen` CLI（`jasna/framegen_cli.py`、新規 `console_script`）: **復元済み動画にフレーム生成だけ**（2x/4x）を適用する（検出も復元もしない）。同じ NVDEC/NVENC + mkvmerge 経路と `FrameGenWriter` を再利用する薄いドライバで、`jasna.pipeline` や `jasna.protection` を一切 import しない（`tests/test_framegen_cli_protection_free.py` で担保）。2パス運用（公式バイナリで復元 → ここでアップコンバート）を可能にする。フォルダ入出力 + `--output-pattern`（`{original}` テンプレート）に対応し、`media_files.classify_folder` / `folder_output_path` を再利用。動画のみ（画像はスキップ）、バッチ全体で generator を 1 つ共有。テスト: `tests/test_framegen_cli_{driver,device,folder,folder_device,protection_free}.py`。
- 変換ツール: `make_rife_torchscript.py`（Practical-RIFE → TorchScript、`Model.inference` 委譲で版差吸収、CUDA では fp16 トレースが既定で失敗時は fp32 自動フォールバック、RIFE 4.25 で動作確認）。手順は `docs/FRAME_GENERATION_{ja,en}.md`。
- テスト: `tests/test_frame_gen_writer.py`（PTS と枚数、GPU 不要）と `tests/test_rife_frame_generator.py`（パディング + 補間形状 + TorchScript ロード + fp16 既定/fp32 フォールバック、CUDA 限定）。
- 実機検証（RTX、Windows、ソース実行）: 30fps、10661 フレームの入力 → `--frame-gen 2x` で 60fps、21321 フレーム（=`(N-1)*2+1`）、尺と音声同期が不変であることを `ffprobe` で確認。

### 注意と制限
- v1 は RIFE を blend-encode スレッド上で全解像度で PyTorch 実行（TensorRT 化や専用ステージ化は将来課題）。
- RIFE は既定で fp16（`--fp16` に追従）。同梱 IFNet は warp のグリッドを flow の dtype で生成して `grid_sample` の dtype 一致要求を満たす。float32 グリッドを焼き込んだ外部 TorchScript チェックポイントは初期化時プローブで検出し fp32 へ自動フォールバック。実測（RTX 5060 Ti、1080p、`--frame-gen 2x`、lada-yolo-v4、エンドツーエンド）: fp32 16.5fps → **fp16 31.4fps（約1.9倍）**。fp16/fp32 出力間の PSNR は平均約 50dB（見た目同一）。
- RIFE 重みは同梱しない。チェックポイントを各自用意し、ライセンス（Practical-RIFE の学習済み重みは非商用条項）を確認すること。

---

## 5. バグ修正: Linux でのモザイク領域「チャンク移動」アーティファクト（issue #2 / upstream #158）

Linux で処理した動画で、de-mosaic（モザイク除去）領域だけ再生がガクつき周囲とずれて見える（"chunk movement"）症状。検出器、二次リストア、RTX Super-Res、MPS、VRAM などの設定に依存せず、upstream にも存在した（#158 で報告）。

> **upstream `v0.6.2` が主因を修正した。** `v0.6.2` の目玉である `cf33bcf`（"sync decoder cuda stream"）が、デコーダの stream 同期（ドライバレベルの blocking stream 上に作った CUDA `ExternalStream` と、decode→convert→copy の各 handoff まわりの `synchronize()` バリア）を再導入する。これは本ブランチが従来 `torch.cuda.synchronize()` バリアで直していたのと同じデコーダのサーフェス使い回しレースなので、**本ブランチのデコーダ修正と opt-in の `--decoder nvdec` バックエンドは冗長になったため削除**した。既定経路のカクつきの底は `v0.6.2` が解消する。

### 本ブランチで残す: 一次リストアのバッファ保持レース
upstream が**対処していない**もう1つの独立系統はここで残す:

- `jasna/restorer/restoration_pipeline.py`: TRT runner の永続出力バッファの view を保持したまま後段（別スレッド）で uint8 へコピーアウトしており、その前に次クリップの推論が同バッファを上書きすると、復元領域が別クリップのデータになる。`primary_raw = self.restorer.raw_process(...).clone()`（一次スレッドで同 stream 上、次推論より前にコピーアウト）で修正。

この clone は、デコーダの stream 同期だけでは覆えないリストア側のレースを塞ぐ。

---

## 6. フォルダ一括処理（modi）

upstream v0.7.0 がフォルダ入力（`--input <dir> --output <dir>`、画像→動画の順に処理）を追加した。その上に、フォルダ一括時に効くフレーム生成のフィックスが1つ残る:

> **毎ファイル進捗バナー（現在は上流）。** modi は以前、フォルダバッチで各動画の前に `[i/N] Processing <in> -> <out>` を表示していた（`d5c801f`）。上流 `v0.7.2` が同一のバナーを追加し（`2f3a87e`）、画像+動画を合算した `[current/total]` カウンタに拡張したため、modi のコミットは冗長として本リベースで **drop**。履歴として記載。

- **一括処理での frame-gen**: RIFE generator は1回だけ構築され全動画で共有されるが、`FrameGenWriter.close()` が1本目の後にその**借用**generator を close（モデル解放 → `_model = None`）していたため、2本目が `'NoneType' object is not callable` でクラッシュしていた。writer は generator を close しないようにし、構築した側（CLI のフォルダループ / GUI のジョブ）が一括処理後に1回だけ close してライフサイクルを所有する。パイプラインが借用 restoration pipeline を扱うのと同じ方針。検証: 2本動画フォルダ + `--frame-gen 2x` で両方処理され、各 `(N-1)*2+1` フレーム。

## 7. 新機能（modi）: torchcodec バックエンド（実験的、vali / PyNvVideoCodec フォールバック付き）

`python_vali`（decode）と `PyNvVideoCodec`（encode）の代わりに使える実験的な `torchcodec>=0.14.0` バックエンドを追加した（オプション依存、既定 off）。既定は `native`（従来挙動）で、`--video-backend {native,auto,torchcodec}` と、個別上書きの `--decode-backend`/`--encode-backend {inherit,...}` で選択する。`auto` は適用できる場面で torchcodec を使い、それ以外はネイティブへフォールバックする。

torchcodec のデコードは全入力を 8bit RGB で代替でき、エンコードは 8bit の HEVC/AV1（`*_nvenc`）に対応する。色空間（BT.601/709/2020）は既存の remux が付与し、NVENC 設定はマッピング可能なキー（cq/qmin/qmax/gop/lookahead/temporalaq/aq/nonrefp/maxbitrate/vbvbufsize と preset）を `extra_options` に変換して適用する。**ネイティブが担うのは 10bit、マッピング不可の設定、frame-gen、streaming のみ**。デコード速度は素材依存（合成では torchcodec が速く、実 1080p ではネイティブが約 12% 速いが、デコードは非ボトルネック）。実装は `jasna/media/backend.py`（選択レイヤ）、`torchcodec_decoder.py`、`torchcodec_encoder.py`。設計と能力マトリクスは `docs/TORCHCODEC_BACKEND_{ja,en}.md` を参照。

## 8. 新機能（modi）: cuDNN FP8 復元バックエンド（実験的、TensorRT フォールバック付き）

BasicVSR++ の **upsample** サブエンジンを cuDNN graph API の FP8 畳み込みで置き換える実験的バックエンド。lada-ex `feat/fp8-recon`（`lada/models/basicvsrpp/fp8_recon.py`、AGPL-3.0。lada が reconstruction と呼ぶステージは同一の部分ネットワーク）からの移植。`--fp8-recon` で opt-in（GUI やサブプロセス経路にも届くよう `JASNA_FP8_RECON=1` へブリッジ）、既定 off は従来挙動。FP8 対応 GPU（sm89 以上）、fp16 モード、新依存 `nvidia-cudnn-frontend`（cuDNN ランタイム >= 9.17 は torch cu130 wheel に同梱。win32 は compile glue 用に `triton-windows` も入り、inductor が動かない場合は eager へ恒久降格）を要し、構築に失敗したら警告を出して TensorRT エンジンにフォールバックする。

有効時は TensorRT upsample エンジンを**ロード自体しない**（`load_sub_engines(..., load_upsample=False)`）ので、ロード時アリーナ（既定 b90 プロファイルで実測 2210 MB）が確保されない。エンジンファイルはフォールバック先として引き続きビルド・保持される。RTX 5080 (sm120) での実測: ステージレイテンシは TRT FP16 エンジン比 1.45〜1.56 倍（T=60 で 8.54 → 5.46 ms）、FP8 常駐は約 220 MB（純減 −1991 MB）、ステージ出力は FP32 参照比 PSNR 64 dB。パイプラインでは 480p〜4K のクリップで VRAM peak が 0.9〜1.7 GB 下がる一方、e2e fps は不変（計測した全構成で律速は検出側）。出力は FP16 エンジンと目視で区別できず（SSIM 0.983〜0.993）、走行間でビット決定的（md5 一致、FP16 と同等）。統合で2件の問題を発見し修正した: `import cudnn` の前に torch 同梱 cuDNN の lib ディレクトリを `LD_LIBRARY_PATH` の先頭に置く（nvvfx がサブライブラリ欠落の 9.7 ディスパッチャ入りディレクトリを先頭に追記するため、rtx-super-res を先に構築すると cudnn-frontend がそれを掴んで `cudnnCreate` で abort していた）、warmup で**全** T バケットを事前ビルドする（jasna のクリップ長は [1, max_clip_size] で自由に分布し、lada のチャンク長固定と違い遅延ビルドが復元ステージ内に混入するため）。実装は `jasna/restorer/fp8_upsample.py` と `basicvsrpp_sub_engines.py` の注入・フォールバック・close 対応。A/B ゲートベンチは `jasna --benchmark --benchmark-filter fp8`（パイプライン A/B と出力比較はリポジトリ外のローカルハーネスで実施）。Windows の frozen build での `cudnn` 同梱は未検証の開放事項（失敗しても TRT フォールバックで実走は継続）。設計と全計測は `docs/FP8_RECON_{ja,en}.md` を参照。
