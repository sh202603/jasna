# 変更点サマリ: `modi`（`v0.9.1+modi`）vs upstream main

このブランチが upstream（`main`）に対して加えている変更の要約です。`modi` は upstream main `a7cdaf8`（`v0.9.1` リリースタグ）にリベース済み。

- ブランチ: `modi`（fork `https://github.com/sh202603/jasna.git`）
- バージョン: `0.9.1+modi`（upstream main `a7cdaf8` ベース = `v0.9.1` タグ）

> **`v0.9.1` ベースでの新規分**（2026-07-30 取り込み）: 上流は v0.9.0 と v0.9.1 を連続リリースした（計 88 コミット）。主要点: **rfdetr-v6 世代の検出器**（デフォルト `rfdetr-v6` @576px/閾値 0.35、4K 向け `rfdetr-v6-large`、VR 向け `rfdetr-vr-v1` + スタジオ別ルーティング）と **dynamic-batch TensorRT エンジン**（`.bs1-N` 命名。固定バッチ機は `TrtRunner` 内で self-pad）、**VR 投影の全面再設計**（新設 `vr_projection.py` によるモザイク領域単位の raw/fisheye/gnomonic conditioning。全フレーム fisheye warp は消滅）、**CUDA カーネル融合 4 種**（RGB→NV12/P010、bilateral denoise、`.cube` LUT、検出前処理。ビルド済み `.fatbin` を同梱）、**VALI NVDEC デコードバックエンドの復活**（`DECODE_BACKEND` in-code トグルで VALI → PyAV-hw → PyAV-sw のエスカレーション。Kruk2 氏の corruption-tolerant な `python_vali` フォーク wheel が前提で、素の PyPI wheel では PyAV にフォールバック）、**`--sharpen`**（エンコード前 CAS）、**`--scene-detection`**、ソース連動の**ビットレート上限**、pts 単調クランプ、そして**本フォークの提案（[#244](https://github.com/Kruk2/jasna/issues/244)）から採用された `--fmp4` と StreamingEncoder の broken-pipe 修正**（upstream `cda1068`/`229403f`）。v0.9.1 ではさらに **preprocess/upsample sub-engines の固定バッチ化**（`preprocess_b60` / `upsample_dyn_b30`。`--max-clip-size` 変更での再コンパイルが不要になり、ベース VRAM は解像度に応じて 1.5〜3.5GB 減）、**loop body の CUDA graphs 既定 ON**（kill switch `JASNA_TRT_CUDAGRAPHS=0`）、色変換のゼロアロケーション化（`media/yuv_scratch.py`）、Windows 静的 DPI scaling（`customtkinter==6.0.0` 固定）。本フォーク側では 3 系統を**上流実装採用のため drop**: fmp4 実装一式 + streaming 修正（#244 経由で採用）、TRT 静的 shape bind / YOLO チャンク分割の OOB 修正（同じ OOB を upstream が dynamic エンジン + `TrtRunner` self-pad で修正、`c662ed5`+`c79fb88`）、VR fisheye の mask 合成ゴースト修正（対象の全フレーム warp 自体が消滅。新しい領域単位 delta 合成でのゴースト再発有無は再評価予定）。適合対応: `--video-backend` は上流のデコード連鎖の**上位層**として再定義（`native` = 上流の VALI→PyAV エスカレーション。`torchcodec` は従来どおり）、`make_video_encoder` は `--sharpen`/`--fmp4` をパススルー（torchcodec エンコーダでは CAS を適格性ゲート、fmp4 を警告）、**`--fp8-recon` は固定バッチ upsample 前提に再構成**（cuDNN グラフのバケットはクリップサイズでなく固定 `UPSAMPLE_BATCH=30` 上限に）。PyAV wheel は PyAV main からの再ビルドが必須（エンコーダが `CudaContext(cuda_stream=...)` で CUDA stream を渡すため）で、TRT エンジンは初回に全再コンパイルとなる。
>
> **`d7a99bd` ベースでの新規分**（2026-07-23 取り込み）: 上流は 4 つの構造リファクタを行った — ①**ドキュメントの per-language 化**（`docs/{en,ja,zh}/` にトピック別ファイル、README から旧セクションを移設）、②**GUI ロケールの言語別分割**（`gui/locales/{en,zh,ja,ko,th}.py`）と **CLI ヘルプ共有テーブル**（`jasna/cli_help.py`、GUI ツールチップが参照）、③**CLI/GUI 共通の composition root**（`session_config.py` / `session_factory.py`）、④**SettingsPanel のセクションモジュール分割**（`gui/settings_sections/`）。加えて **VR 空間メタデータ注入の削除**（`media/spatial_metadata.py` ごと撤去 — 本フォークは追従し、旧 `--mp4-fast-start` の VR180 フォールバックも不要化）、vendored mmagic の inference-only 化、`patch_frozen_torch` のエントリポイント移動、**時間方向の検出フィルタ**（`--max-detection-gap` / `--min-detection-duration`）、RTX Super-Res の入力サイズ設定、AV1 の NVDEC デコード修正、セグメント用ワーキングディレクトリ設定。本フォークの対応: **`--mp4-fast-start` を `--fmp4` に改名**（後方互換なし。preset キーも `fmp4`。upstream PR 用 `feat/fmp4` ブランチと同一仕様）。NVENC 非対応オプションの黙殺修正は upstream `af9374b` に取り込まれたため **drop**。modi の GUI 追加（フレーム生成、動画バックエンド、FP8、fmp4）はセクションモジュールへ移植し、flashvsr-inline は `SessionConfig` 拡張 + `session_factory._build_secondary_restorer` 分岐として、フレーム生成/動画バックエンドは `build_pipeline` のパススルー引数として composition root に載せ替えた。modi 独自ドキュメントも `docs/{en,ja}/` の小文字トピック名へ再配置（zh 版は作らない）。
>
> **`v0.8.1` ベースでの新規分**（2026-07-19 取り込み）: 上流は **AMD ROCm 対応（実験的、Linux）** を追加した — `jasna/accelerator.py`（vendor/capabilities 判定）、検出の onnxruntime-migraphx 実行（`mosaic/migraphx_runner.py`）、AMF ハードウェアデコード/エンコード、ROCm 向け cumsum ベース box blur（`_prefix_box_blur`）。それに伴い **pyproject の依存が extras に分割**され（`nvidia` = torch cu130 + TensorRT + nvidia-vfx / `amd` = torch 2.9.1 + migraphx）、ソース実行の dev インストールは `uv pip install -e .[dev,nvidia]` になった。`requires-python` は **>=3.12** に緩和。本フォークが報告した **NVENC 入力ピッチ整列バグ（[Kruk2/jasna#230](https://github.com/Kruk2/jasna/issues/230)）は upstream `bb6e36e` で修正**され、Linux ドライバ最小 580 化も upstream `041eded` が実装したため、対応するローカル 2 コミット（256B 整列版ピッチ修正 / os_utils のプラットフォーム分岐）は**このリベースで drop し upstream 版を採用**した（§10）。ほか GUI 区間プレビューの HiDPI 修正（#229）、リリースビルド整備（分割 Linux アーカイブ、libnvJitLink.so.13 同梱、Windows AMD リリース）。
>
> **`v0.8.0` ベースでの新規分**（2026-07-18 取り込み）: 上流はメディア層を **PyAV（NVDEC/NVENC）へ全面移行**し（`python_vali` / `PyNvVideoCodec` をランタイムから撤去、**mkvmerge 依存も削除**）、**AV1/H264 エンコード**・**BT.601/709/2020 × full/limited × 8/10-bit** の色空間/ビット深度対応・**ソフトウェアデコードフォールバック**・**アナモルフィック（SAR）対応**を実装した。さらに**区間エディター + スマートレンダリング**（`--segments`）、**復元プレビュー再生**、**マスク提案エディタ**（明示的な Submit 時のみ、フレーム+マスクを暗号化して上流の Cloudflare Worker へ送信）、**VR180 復元**（`--vr-mode`）、**高 FPS リターゲット**（`--retarget-high-fps`、60→30 間引き）、多数の Linux GUI 修正が入った。ドライバ要件は Windows 610+ / Linux 580+、依存に `av>=18,<19`（GPU パスは PyAV 18.1.0 相当の current_ctx API が必要。18.1.0 公開まで upstream main `61e4aa8` からのビルド wheel を使う）。
>
> このリベースの方針は **重複機能は upstream 優先**: 本フォークの「柔軟な出力」（旧 §2）は upstream 実装に全面置換して drop（`--bit-depth` フラグも廃止、下記 §2）。torchcodec バックエンドは新メディア層に合わせて意味論を再定義した（§7）。
>
> **ベースを `v0.7.2` へ更新。** 以下のコミット別の突き合わせ記録は当初の `v0.6.2` 期リベース向けに書かれたものだが、それらの upstream 修正（デコーダ stream 同期、分離畳み込み、validate-model-name、onnx export、trt load）は現在のベースにも引き継がれており（upstream は v0.7.0 リリース時に `main` を force-push して SHA を書き換えたが、機能は維持）、収束関係は今も成立する。`v0.7.1` ベースで既に入っていたのは upstream の **サポーター向けモデル**（SD 1.5 画像復元、unet-4x）とモデル暗号化/Nuitka 周りで、本フォークはそのコードを **inert（動作しない形）** のまま同梱する（公開ソースからは復号も実行もできない。README のスコープ注記を参照）。ビルドは upstream に合わせて PyInstaller から Nuitka へ移行したが、パッケージングツールは非公開のため、公開での利用経路はソースからの実行（`docs/{en,ja}/building_*` を参照）。
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
  `model_weights/` の自動解決（`jasna/model_weights_resolver.py` 新規）。upstream も frozen バイナリ向けの `engine_paths.model_weights_dir()` を追加したため、upstream の呼び出し箇所を本リゾルバへ委譲して統一（env 上書き、パッケージ親フォールバック、ロギングを持つ superset を維持）。ほか TRT ベンチの新 API 追従、PyInstaller spec の調整など。旧 `video_decoder.py` の Windows 向け `python_vali`/CUDA DLL ロード補助は、upstream v0.8.0 の PyAV 移行（vali 撤去）で不要になったため drop。
- **YOLO 検出エクスポート修正**: 検出 ONNX を CPU でエクスポートし、凍結（PyInstaller）バイナリで CUDA error 100 を回避する。ultralytics の GPU エクスポートは TensorRT より先に torch CUDA を初期化して初期化順序を壊すため。エクスポート後に `CUDA_VISIBLE_DEVICES` を復元。エンジンは引き続き fp16 でビルドされる。（`onnx` 自体は upstream の依存になった。）
- **Linux GUI / RTX-VSR 修正**
  RTX Super-Res の TensorRT バージョン衝突（`libnvinfer.so.10` の先読み）修正。旧来のモーダルダイアログ空表示対策（遅延 `grab_set`）は、upstream v0.8.0 の `wait_visibility()` ベースの X11 対応 + Linux GUI 修正群で置換されたため drop。
- **ビルドガイド整備**: `docs/{en,ja}/building_{linux,windows}.md`。本ブランチはソース適用済みのためクローン後そのままビルド可。各ガイドの付録は変更内容の内訳を参照用に記述。

> mmengine 用の `patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff` は venv 内パッケージへ当てるもので、ビルド手順（各ガイド §5.1）で別途適用します（これが唯一同梱されるパッチ）。

---

## 2. 出力フォーマットの柔軟化（upstream v0.8.0 に全面吸収）

> **本項は本ブランチ独自の差分ではなくなった。** modi は v0.7.2 期に AV1 / 8-10bit / BT.601-2020 出力（`--codec` / `--bit-depth`、`chw_rgb_to_surface` 汎用変換、mkvmerge+ffmpeg remux）を独自実装していたが、upstream v0.8.0 が PyAV ベースの新エンコーダで**同等以上**を実装した: `--codec {hevc,h264,av1}`、BT.601/709/2020 × full/limited × NV12/P010 変換一式（`rgb_to_nv12.py` / `rgb_to_p010.py`）、H.273 色タグの PyAV インライン付与、フルレンジ入力対応、mkvmerge 廃止。リベース方針（upstream 優先）に従い modi 実装は drop した。以下は帰結。

- **`--bit-depth` フラグは廃止**。ビット深度は upstream 仕様どおり codec で決まる（hevc/av1 → 10-bit P010、h264 → 8-bit NV12。`--segments` のスマートレンダリング断片のみ `match_input_bit_depth` で 8-bit ソースに追従）。
- modi の `Colorspace` enum / `VideoMetadata.yuv_colorspace` / `chw_rgb_to_surface` / mkvmerge ベースの remux は削除（upstream は `av>=18` で BT.2020 をネイティブ表現できる）。
- 旧 `docs/CODECS_AND_COLORSPACE_{ja,en}.md` と `tests/test_rgb_to_surface.py` は削除（upstream の `test_rgb_to_nv12.py` / `test_rgb_to_p010.py` / `test_video_encoder_mux.py` が置き換え）。
- upstream に無かった modi 差分でこの領域に残るのは **frame-gen 連携の出力 fps 配線**（§4）のみ。

---

## 3. パフォーマンス修正: ブレンドマスク生成の分離畳み込み化（upstream 取り込み済み）

> **本項は本ブランチ独自の差分ではない:** upstream `v0.6.2` が同等実装（`f8c4048` "Reapply separable conv"）を持つため、リベース時に重複コミットは drop 済み。以下は経緯の記録。

upstream の `create_blend_mask`（`jasna/tracking/blending.py`）は、フレーム高に比例した大きな一様カーネル（1080p で 61×61、4K で 121×121、超解像時はさらに増大）の密な `conv2d` box blur を **2回**（dilation + falloff）実行していた。`O(K²)` のため解像度や検出数に応じて重くなる。一様カーネルは分離可能（`K×K = (1×K) ⊛ (K×1)`）なので 1D 畳み込み 2 パスに置換すると `O(K²)` → `O(2K)`、出力はビット同等。本ブランチと upstream はこれを独立に、ほぼ同一アルゴリズムで実装したため、リベースでは upstream 版を採用した。

---

## 4. 新機能（modi）: フレームレート倍化（フレーム生成）

AI補間フレームを挿入して出力フレームレートを上げる: `--frame-gen {none,2x,4x}`（ファイル出力のみ。ストリーミング非対応、`--segments` スマートレンダリングとも併用不可 — コピー区間とフレームレートが一致しなくなるため起動時に拒否する）。

- **なぜ二次リストアではなく新ステージか**: 二次リストア（`unet-4x` / `tvai` / `rtx-super-res`）は 256×256 のモザイククロップを処理しフレーム数を変えない。フレーム生成は全解像度の出力フレームに対しフレーム数と PTS を**増やす**ため、パイプラインの `FrameWriter` を薄くラップするデコレータ（`FrameGenWriter`）として挿入する。パイプライン本体とエンコーダは無改造。
- **バックエンド**（`--frame-gen-backend {rife,rtx}`）。`FrameGenerator` プロトコルで差し替え可能:
  - `rife`（既定）: ニューラル補間（RIFE）。CUDA で**現在利用可能**。TorchScript チェックポイント（推奨、自己完結）か、同梱 RIFE 4.6 `IFNet`（`jasna/models/rife/`）への state_dict を読み込む。重みは `--frame-gen-model-path` か `model_weights/rife.pth`。
  - `rtx`: **NVIDIA RTX Video Frame Generation**。RTX Spark と同時に Python ホイール + ComfyUI ノードとして発表されたが、`nvidia-vfx`（1.2.0 は `VideoSuperRes` のみ公開）に**未出荷**。アダプタ（`jasna/framegen/rtx_frame_generator.py`）は将来の Effect を探し、出荷までは明示エラーにする。出荷後は推論呼び出し 1 箇所で有効化できる。
- **PTS 計算**（`FrameGenWriter`）: 連続する 2 フレームごとに、実フレームを出力後 `M-1` 枚の補間を `pts_k = prev_pts + round((curr_pts - prev_pts) * k / M)` で挿入。出力タイミングは PTS 駆動なので、PTS 挿入が 2x/4x を生む。音声は元のタイムコードを保持するため尺と同期は不変。総フレーム数は `(N-1)*M + 1`。非単調 PTS の区間は補間をスキップ。エンコーダの fps/GOP は、v0.8.0 の upstream エンコーダが持つ `output_fps` パラメータに **ソース fps × 倍率** を渡して正す（旧 `output_fps_multiplier` パラメータは廃止。`--retarget-high-fps` の間引き後 fps にも倍率が乗る）。
- CLI（`jasna/main.py`）と GUI（`jasna/gui/`）に `--frame-gen` / `--frame-gen-backend` を追加。GUI のエンコード設定には RIFE 重みパス（`--frame-gen-model-path` 相当）の入力欄もある。
- スタンドアロン `jasna-framegen` CLI（`jasna/framegen_cli.py`、新規 `console_script`）: **復元済み動画にフレーム生成だけ**（2x/4x）を適用する（検出も復元もしない）。同じ NVDEC/NVENC 経路と `FrameGenWriter` を再利用する薄いドライバで、`jasna.pipeline` や `jasna.protection` を一切 import しない（`tests/test_framegen_cli_protection_free.py` で担保）。2パス運用（公式バイナリで復元 → ここでアップコンバート）を可能にする。フォルダ入出力 + `--output-pattern`（`{original}` テンプレート）に対応し、`media_files.classify_folder` / `folder_output_path` を再利用。動画のみ（画像はスキップ）、バッチ全体で generator を 1 つ共有。テスト: `tests/test_framegen_cli_{driver,device,folder,folder_device,protection_free}.py`。
- 変換ツール: `scripts/make_rife_torchscript.py`（Practical-RIFE → TorchScript、`Model.inference` 委譲で版差吸収、CUDA では fp16 トレースが既定で失敗時は fp32 自動フォールバック、RIFE 4.25 で動作確認）。手順は `docs/{ja,en}/frame_generation.md`。
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

## 7. 新機能（modi）: torchcodec バックエンド（実験的、native = PyAV フォールバック付き）

ネイティブ（v0.8.0 以降は PyAV NVDEC/NVENC）の代わりに使える実験的な `torchcodec>=0.15.0` バックエンド（オプション依存、既定 off）。既定は `native`（従来挙動）で、`--video-backend {native,auto,torchcodec}` と、個別上書きの `--decode-backend`/`--encode-backend {inherit,...}` で選択する。

**v0.8.0 リベースでの意味論変更**: 上流のネイティブエンコーダは HEVC/AV1 を常に 10-bit（P010）で出力するようになり、8-bit nv12 専用の torchcodec エンコードでは出力パリティが成立しない。そのため:

- **decode**: 従来どおり `auto` で torchcodec を優先し、失敗時はネイティブへフォールバック。ただし `--retarget-high-fps` の frame stride はネイティブリーダー専用（強制指定時はエラー、auto はネイティブへ）。
- **encode**: **`auto` では選択されない**（出力ビット深度が黙って変わるのを防ぐ）。`--encode-backend torchcodec` の強制指定時のみ動作し、対象は **8-bit ソース + HEVC/AV1 + マッピング可能な NVENC 設定**に限る（出力は 8-bit。streaming / `--segments` / frame-gen / fps 変更は拒否）。色空間（BT.601/709/2020）は encoder 内蔵の ffmpeg copy-remux が H.273 コードポイント（+ HEVC は `hevc_metadata` BSF で VUI 書き換え）で付与し、旧 mkvmerge ヘルパーへの依存は撤去した。
- b2471d5 由来の**観測可能バックエンド**（起動ログに `[decode: ..., encode: ...]` を表示、リーダー/エンコーダの `backend` 属性）と**エンコード専用ワーカースレッド**は維持。

NVENC 設定のマッピング（cq/qmin/qmax/gop/lookahead/temporalaq/aq/nonrefp/maxbitrate/vbvbufsize と preset → `extra_options`）は従来どおり。実装は `jasna/media/backend.py`（選択レイヤ）、`torchcodec_decoder.py`、`torchcodec_encoder.py`。GUI ではエンコード設定の「ビデオバックエンド」ドロップダウン（decode/encode の両方に適用）として選択できる。設計と能力マトリクスは `docs/{ja,en}/torchcodec_backend.md` を参照（v0.8.0 の意味論変更は本節が正）。

## 8. 新機能（modi）: cuDNN FP8 復元バックエンド（実験的、TensorRT フォールバック付き）

> **v0.9.1 ベースでの更新:** 上流は upsample ステージを固定 `UPSAMPLE_BATCH=30` のチャンク実行に変えた（`upsample_dyn_b30` エンジン）ため、cuDNN グラフのバケットもクリップサイズでなく 30 上限になった — グラフ数が減り、warmup は `--max-clip-size` に依存しない。以下のアリーナ/レイテンシ/VRAM の数値は b90 時代の構成での実測であり歴史記録として残す。上流の固定バッチ化自体が TRT アリーナの VRAM をかなり回収するので、FP8 のレバーは縮む見込みで、再測定を予定している。

BasicVSR++ の **upsample** サブエンジンを cuDNN graph API の FP8 畳み込みで置き換える実験的バックエンド。lada-ex `feat/fp8-recon`（`lada/models/basicvsrpp/fp8_recon.py`、AGPL-3.0。lada が reconstruction と呼ぶステージは同一の部分ネットワーク）からの移植。`--fp8-recon` で opt-in（GUI やサブプロセス経路にも届くよう `JASNA_FP8_RECON=1` へブリッジ。GUI では詳細設定の「FP8復元（実験的）」トグル）、既定 off は従来挙動。FP8 対応 GPU（sm89 以上）、fp16 モード、新依存 `nvidia-cudnn-frontend`（cuDNN ランタイム >= 9.17 は torch cu130 wheel に同梱。win32 は compile glue 用に `triton-windows` も入り、inductor が動かない場合は eager へ恒久降格）を要し、構築に失敗したら警告を出して TensorRT エンジンにフォールバックする。

有効時は TensorRT upsample エンジンを**ロード自体しない**（`load_sub_engines(..., load_upsample=False)`）ので、ロード時アリーナ（既定 b90 プロファイルで実測 2210 MB）が確保されない。エンジンファイルはフォールバック先として引き続きビルド・保持される。RTX 5080 (sm120) での実測: ステージレイテンシは TRT FP16 エンジン比 1.45〜1.56 倍（T=60 で 8.54 → 5.46 ms）、FP8 常駐は約 220 MB（純減 −1991 MB）、ステージ出力は FP32 参照比 PSNR 64 dB。パイプラインでは 480p〜4K のクリップで VRAM peak が 0.9〜1.7 GB 下がる一方、e2e fps は不変（計測した全構成で律速は検出側）。出力は FP16 エンジンと目視で区別できず（SSIM 0.983〜0.993）、走行間でビット決定的（md5 一致、FP16 と同等）。統合で2件の問題を発見し修正した: `import cudnn` の前に torch 同梱 cuDNN の lib ディレクトリを `LD_LIBRARY_PATH` の先頭に置く（nvvfx がサブライブラリ欠落の 9.7 ディスパッチャ入りディレクトリを先頭に追記するため、rtx-super-res を先に構築すると cudnn-frontend がそれを掴んで `cudnnCreate` で abort していた）、warmup で**全** T バケットを事前ビルドする（jasna のクリップ長は [1, max_clip_size] で自由に分布し、lada のチャンク長固定と違い遅延ビルドが復元ステージ内に混入するため）。実装は `jasna/restorer/fp8_upsample.py` と `basicvsrpp_sub_engines.py` の注入・フォールバック・close 対応。A/B ゲートベンチは `jasna --benchmark --benchmark-filter fp8`（パイプライン A/B と出力比較はリポジトリ外のローカルハーネスで実施）。Windows でも検証済み（Windows 11 / 同一 RTX 5080。A/B ベンチが同等数値で全ゲート通過、1080p フルパイプライン走行も正常完了、glue は `triton-windows` 経由でコンパイル）。Windows の frozen build での `cudnn` 同梱は引き続き未検証の開放事項（失敗しても TRT フォールバックで実走は継続）。設計と全計測は `docs/{ja,en}/fp8_recon.md` を参照。

---

## 9. 新機能（modi）: FlashVSR 二次復元（実験的、オフライン3段 + inline）

一次の BasicVSR++ が大きなモザイク領域・接写・4K でぼやけさせるテクスチャを、拡散 one-step VSR の [FlashVSR](https://github.com/OpenImagingLab/FlashVSR)（fork [`lihaoyun6/FlashVSR_plus`](https://github.com/lihaoyun6/FlashVSR_plus)、4x 固定）で補う二次復元。256px の一次復元クロップを 1024px へ拡大して再 blend する。FlashVSR は独自ライセンスのサードパーティモデルで、checkout・重み・専用 venv は利用者が用意し `--flashvsr-repo` で渡す（jasna のサポーターモデルとは無関係、同梱なし）。FlashVSR venv は **uv-managed standalone Python 必須**（system Python は Triton の Sparse_SageAttention カーネルを JIT できない）。**2 つのモード**がある。

- **`--secondary-restoration flashvsr`（オフライン3段）**: `dump → FlashVSR → reblend` を独立サブプロセスで順に実行し、各段の終了で VRAM を全解放する。FlashVSR の tiny モードは単体 12–16 GB を要し一次（~9 GB）と 16 GB カード上で同時常駐できないため、ピーク VRAM が時間的に重ならないようプロセス分割する。中間 bundle（256px + 非圧縮 1024px、モザイク多めの長尺では数百 GB 級）をディスクに持ち、`--flashvsr-bundle-dir` で段階再開できる。**12 GB 級 GPU でも動作**。encode は 2 回（Phase 1 の捨て + Phase 3 の最終）。実装 `jasna/restorer/flashvsr_offline.py` + `flashvsr_phase2_driver.py`、サブプロセス分岐 `jasna/__main__.py`（`--flashvsr-phase`）。

- **`--secondary-restoration flashvsr-inline`（inline 単一パス）**: FlashVSR を通常のストリーミングパイプラインに二次復元として挟む。**中間ファイル・ディスクゲート・二重 encode が無い**。FlashVSR の **tiny-long**（O(1)、フレーム数非依存で定常 ~11.9 GB）を使い、fp8-recon（§8）で ~1.6 GB に絞った一次と**同時常駐**させて単一パスを実現する。同期 `SecondaryRestorer` が FlashVSR venv worker を resident spawn し、length-prefixed の RGB プロトコルで 256px→1024px をやり取りする（`imageio.get_writer` を差し替えロスレスにテンソル捕獲、小 clip は next_8n5 パディングで吸収し厳密に T 枚返す）。offline と同一の制約（clip 32 上限・frame-gen off）を強制し、同時常駐予算のため **fp8-recon を自動有効化**する。VRAM 天井対策として、DiT 推論を横短冊に分割して羽根合成する **`--flashvsr-tiles {1..4}`**（strip tiled-dit、inline 専用。`2` で attn マスク 0.39x / 壁時計 ~1.25x）を持つ。**16 GB カード + patched checkout が前提**。実装 `jasna/restorer/flashvsr_inline_secondary_restorer.py` + `flashvsr_inline_worker.py`。

**tiny-long パッチ（同梱）**: inline が使う tiny-long は fork の実装が第 2 チャンクで壊れる既知バグ（`8192 vs 4096` エラー）があり、2 箇所のチャンク跨ぎキャッシュ clear（per-chunk の `LQ_proj_in.clear_cache()` / `TCDecoder.clean_mem()`）を無効化する修正が要る（ループ前の一度きりのリセットは残す）。fork `lihaoyun6/FlashVSR_plus` は**不活性**のため上流化せず、パッチを [`patches/flashvsr_plus_tinylong_multichunk_fix.patch`](../../patches/flashvsr_plus_tinylong_multichunk_fix.patch) に同梱（利用者が checkout へ `git apply`）。restorer は起動時に checkout がパッチ済みかを検査し、未パッチなら明示エラーで停止してオフライン（tiny、無パッチ可）を案内する。これで唯一同梱されるパッチは mmengine 用（§1）と本パッチの 2 つになる。

**制約**: 両モードともファイル出力専用（`--stream` / 画像入力 非対応）、`--frame-gen` 非対応（フレーム生成は別パス）。オフラインはさらに v0.8.0 新機能の `--retarget-high-fps`（Phase 1 の frame stride が Phase 3 の再ブレンド索引とずれる）・`--segments`・VR 処理（`--vr-mode sbs`/`sbs-fisheye`、`auto` の VR 検出時）と併用不可で、起動時に拒否する。inline は FlashVSR（~15 crop-fps）律速で、モザイクが多い区間はその速度になる。

**検証**: ユニット `tests/test_flashvsr_{offline,inline}.py` + `test_main.py`（GPU 無し CI、full suite リグレッション 0）。実機（RTX 5080 sm120）: オフライン/inline とも E2E 完走・出力フレーム数 = 入力。Linux では inline は 852x480 で combined VRAM peak 14780 MiB、A/B（primary-only 比）でモザイク領域のみが変化することを確認。1080p は物理天井際（~15.8 GB）まで上がり、`vram_offloader`（キューフレームを RAM へ退避）と worker の `expandable_segments` が圧を吸収して完走する。Windows（同カード、`expandable_segments` 非対応）でもフル長 480p/1080p の E2E を確認: 1080p は tiles `1` で GPU 全体ピーク 15.9 GB（天井まで <0.4 GB）、`--flashvsr-tiles 2` で 14.2 GB / 壁時計 +25% となり、短冊境界のシーム（バンディング、色調ずれ）は不検出。Windows 16 GB の 1080p 常用は tiles `2` を推奨。設計と全計測は `docs/{ja,en}/flashvsr.md`。

---

## 10. バグ修正: NVENC 入力ピッチの整列（upstream 報告 → v0.8.1 で修正済み）

> **本項は本ブランチ独自の差分ではなくなった。** 本フォークが [Kruk2/jasna#230](https://github.com/Kruk2/jasna/issues/230) として報告し、upstream v0.8.1 の `bb6e36e`（"fix(encoder): align NVENC input pitch"）が修正した。以下は経緯の記録。

upstream v0.8.0 の PyAV エンコード経路は、RGB→YUV 変換結果のテンソルを `VideoFrame.from_dlpack` でゼロコピーのまま NVENC に登録する。このときの行ピッチは自然幅（W × 要素サイズ）になり、**16 バイト整列しない幅**（480p 定番の 852 幅 = P010 ピッチ 1704 B など）では NVENC のドライバ側カーネルが `cudaErrorMisalignedAddress` を起こす。共有 CUDA コンテキストが毒されるため decode/restore/blend/encode の全スレッドが同時にクラッシュし、プロセスはハングする（実測: Linux 595.71.05 / RTX 5080。幅 852/854/860 で再現、856/864/1280/1920/3840 は通過。コーデック不問、TRT・MPS・エンジン世代は無関係と切り分け済み）。

本フォークはまず 256 B 整列のステージングバッファ版をローカル実装して報告した。upstream 修正 `bb6e36e` は同じ構造（幅広バッファ + `[:, :width]` の strided view）で **16 B 整列**を採用し、パディングのゼロ埋めとテスト（`test_video_encoder_unit.py` の `_align_yuv_pitch` ユニット + `test_video_encoder_mux.py` の幅 852/854/860 × hevc/h264/av1 隔離プロセスプローブ）を追加している。v0.8.1 リベースでローカル版は drop し upstream 版を採用。v0.8.1 の AMD 対応後は NVIDIA 経路（`vendor is NVIDIA`）でのみ `_align_yuv_pitch` が呼ばれる（AMF はホストコピー経由でピッチ問題自体がない）。

**検証**（256B ローカル版・v0.8.0 時点）: 幅マトリクス 852/854/856/860/864/1280/1920 × hevc/h264 全通過、`tests/test_video_encoder_mux.py` 27 件パス、852x480（10661 フレーム一致）と 3840x2160 のフルパイプライン完走。**upstream 16B 版はリベース時に同型プローブ（上記 upstream テスト）を RTX 5080 実機で再検証済み。**
