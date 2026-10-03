# フレーム生成（フレームレート倍化）の使い方

`--frame-gen {none,2x,4x}` で出力動画のフレームレートを2倍/4倍にする（ファイル出力のみ、`--stream`非対応）。
中間フレームをAI補間で生成し、元のPTS間に新しいPTSを挿入する。音声は元のタイムコードを保持するため尺と同期は不変。

バックエンドは `--frame-gen-backend {rife,rtx}`：
- `rife`（既定）: PyTorch 上の RIFE ニューラル補間。対応 GPU 全般で動く。重みを別途用意する（後述）。
- `rtx`: `nvidia-vfx` 0.2.0.0 以降の NVIDIA RTX Video Frame Generation。重み不要、RIFE の約 10 倍速で補間精度もわずかに高いが、**RTX 40 シリーズ（Ada）以降限定**。[RTX Video Frame Generation バックエンド](#rtx-video-frame-generation-バックエンドrtx)を参照。

RIFEバックエンドは2つのチェックポイント形式を自動判別する。**TorchScript形式を推奨**（アーキテクチャと重みを内包し、確実に動作する）。`flownet.pkl`のstate_dictを直接置く方法もあるが、同梱IFNetとのキー一致に依存するため確実ではない。

---

## TorchScript重みの作成手順（推奨）

### 1. Practical-RIFE を取得し重みを配置（初回のみ）

```powershell
git clone https://github.com/hzwer/Practical-RIFE
```

**重みとモデルコードは git clone には含まれない**。README のモデル一覧から **RIFE 4.x のモデルパッケージ**を
Google Drive / 百度网盘で手動ダウンロードし、展開した `*.py`（IFNet実装）と `flownet.pkl` を `<repo>\train_log\` に置く
（README: "Download a model ... and put *.py and flownet.pkl on train_log/"）。結果として `train_log\` に
`RIFE_HDv3.py`, `IFNet_HDv3.py`, `flownet.pkl` が揃う状態にする。

> バージョン: 上流は現在 **v4.25 を推奨**（**v4.25 で動作確認済み**）。変換スクリプトは各版の `Model.inference` に委譲して
> 版固有の `scale_list` を使い、timestep 規約（スカラ版/全解像度マップ版）は変換時に**自動判別**するため、他の 4.x でも概ね通る。
> 実際の互換性は次の `--validate` で確認する。

### 2. 変換スクリプトを実行（jasna の venv を使用）

リポジトリの `scripts/make_rife_torchscript.py` を使う。jasna の venv を使うことで torch のバージョンが実行環境と一致する。

```powershell
.\.venv\Scripts\python.exe scripts/make_rife_torchscript.py `
    --rife-repo C:\path\to\Practical-RIFE `
    --output model_weights\rife.pth `
    --validate
```

- **fp16 が既定**（CUDA 時。`--no-fp16` で fp32 トレース、CPU は常に fp32）。バックエンドの既定 fp16 と揃えるためで、fp16 トレース品は dtype 昇格により **fp32 パイプラインでもそのまま動く**。逆に fp32 トレース品は float32 の warp グリッドがグラフに焼き込まれるため、fp16 パイプラインではバックエンドの自動 fp32 フォールバックが発火する（動作はするが fp16 の恩恵を受けない）。fp16 トレースが失敗または非有限値を出した場合はスクリプトが自動で fp32 に切り替える。
- `--validate` を付けると、保存後に**トレース時と異なる解像度**で再ロードして形状、値域、中点ブレンドを検証する（汎化の確認）。
- `--size`（既定256）はトレース解像度。RIFEはスケール相対の補間と実行時生成のwarpグリッドで構成されるため、他解像度にも概ね汎化する。
- 出力は既定で `model_weights\rife.pth`。バックエンドが探す既定パスなのでそのまま使える。別の場所に置く場合は実行時に `--frame-gen-model-path <パス>` を指定する。

### 3. 実行

```powershell
.\.venv\Scripts\python.exe -m jasna --input in.mp4 --output out2x.mkv --frame-gen 2x
.\.venv\Scripts\python.exe -m jasna --input in.mp4 --output out4x.mkv --frame-gen 4x
```

### 4. 確認

```powershell
ffprobe out2x.mkv
```

- `nb_frames` / `avg_frame_rate` が約2倍（4xなら約4倍）
- `Duration` が元動画と同じ（尺不変）
- 音声が同期している

---

## RTX Video Frame Generation バックエンド（`rtx`）

`--frame-gen-backend rtx` は RTX Video Effects SDK の Video Frame Generation effect を使う（`nvidia-vfx` 0.2.0.0 = VFX SDK 1.3.0。`pyproject.toml` で固定している版）。モデルは SDK に内蔵されているので、ダウンロードも変換も要らない。

**要件**

- GPU: **Ada（RTX 40 シリーズ）以降**（compute capability 8.9 以上）。Linux では Hopper も可。Turing（RTX 20）と Ampere（RTX 30）は SDK が対応しない。jasna は起動時に compute capability を確認し、満たさなければ `rife` を使うよう案内するエラーにする。
- ドライバ: Linux は **570.190 以降、580.82 以降、590.44 以降のいずれか**。Windows は **570.65 以降**。jasna 自身の Linux 最低要件（580）は 580.0〜580.81 も通すが、その範囲では SDK 側が effect のロードを拒否する。
- venv に `nvidia-vfx>=0.2.0.0`。この固定より前に作った venv は 0.1.0.1 でフレーム生成 effect を持たない。`uv pip install nvidia-vfx==0.2.0.0` で更新する（PyPI 上の殻パッケージが pypi.nvidia.com から実 wheel を取ってくる）。

**使い方**

```bash
jasna --input in.mp4 --output out2x.mkv --frame-gen 2x --frame-gen-backend rtx
jasna --input in.mp4 --output out4x.mkv --frame-gen 4x --frame-gen-backend rtx --frame-gen-rtx-mode high
jasna-framegen --input restored.mkv --output out2x.mkv --factor 2x --backend rtx
```

`--frame-gen-rtx-mode {low,medium,high}`（`jasna-framegen` では `--rtx-mode`）で SDK の品質モードを選ぶ。既定の `medium` が最もバランスがよく、`high` は最大でも 0.2 dB の差に対して約 6 倍遅い。`--frame-gen-model-path` と `--fp16` は RIFE 専用で、`rtx` では無視される。SDK の自動シーン切替検出は有効のままなので、カット点をまたいだ補間は起きない。

**実測（RTX 5080、nvidia-vfx 0.2.0.0、2026-10-03）**

生成 1 枚あたりの時間（合成した移動対）:

| 解像度 | `low` | `medium` | `high` | RIFE 4.25 fp16 |
| --- | ---: | ---: | ---: | ---: |
| 1920x1080 | 0.61 ms | 1.53 ms | 8.99 ms | 10.1 ms |
| 3840x2160 | 1.56 ms | 2.51 ms | 10.6 ms | 48.3 ms |

補間精度（`assets/test_clip1`、フレーム i と i+2 から i+1 を予測して正解との PSNR。2x 実運用より動きが大きい条件）:

| 手法 | 1080p | 2160p |
| --- | ---: | ---: |
| RIFE 4.25 fp16 | 42.85 dB | 41.41 dB |
| RTX `low` | 42.50 dB | 41.93 dB |
| RTX `medium` | 43.57 dB | 43.01 dB |
| RTX `high` | 43.74 dB | 42.94 dB |

effect ロード後の追加 VRAM: 1080p `medium` +0.55 GB（`high` +1.24 GB、RIFE fp16 +0.75 GB）、2160p `medium` +2.06 GB（RIFE +2.85 GB）。エンドツーエンドでは、300 フレームの 1080p テストクリップに `jasna-framegen --factor 2x` をかけた壁時計時間が `rtx` 2.3 秒、`rife` 5.8 秒（デコード、エンコード、起動込み）。

任意のフレームサイズを受け付け（RIFE のような 64 の倍数へのパディングは不要）、サイズが変わると約 15 ms で再ロードするので、解像度が混在するフォルダ一括処理でも generator を 1 つ共有できる。

**目視確認（2026-10-03、Linux）**: 1080p テストクリップの 2x 出力を `rtx` と `rife` で見比べ、RTX 側に残像、エッジの破綻、シーン切替の誤ブレンドは見られなかった（両出力間の PSNR は 47.6 dB で、近いが同一ではない）。

**Windows の実測（2026-10-03、RTX 5080、ドライバ 616.92、Windows 11）**: 0.2.0.0 の Windows wheel（436 MB、`uv pip install nvidia-vfx==0.2.0.0` で pypi.nvidia.com から取得）で同一コードがそのまま動く。実 SDK を呼ぶユニットテストが合格し、1080p テストクリップの `jasna-framegen --factor 2x` は `rtx` 4.3 秒、`rife` 23.5 秒（この環境の `rife.pth` は warp grid が float32 で焼き込まれた TorchScript のため fp32 fallback）。出力はどちらも 599 枚 60 fps で、両出力間の PSNR は 47.8 dB（Linux の 47.6 dB と同水準）。フルパイプライン（rtx-super-res + `--rtx-strength 0.6 --rtx-highbitrate` + 2x `rtx`）も 599 枚 60 fps で完走し、`--fp8-recon` との併用でも cuDNN の取り合いは起きなかった。

**未確認**: Turing/Ampere で SDK が返す正確なエラー（jasna 側の capability 判定が先に効く）。

---

## 2パス運用: `jasna-framegen`（スタンドアロン）

`jasna-framegen` は、**復元済み動画にフレーム生成だけ**を適用する独立コマンド（モザイク検出も BasicVSR++ 復元も走らせない）。次の用途に使う：

- **1パス目**で復元した動画（公式 jasna バイナリ、または `--frame-gen` なしの `jasna`）に対し、重い復元を再実行せず後から 2x/4x を足したいとき。
- factor やコーデックだけ変えて素早く再エンコードしたいとき。
- バッチ処理のためフレーム生成を本パイプラインから分離したいとき。

統合版 `--frame-gen` と同じ NVDEC/NVENC + mkvmerge 経路を再利用するので、音声と色メタは引き継がれ、タイミングは上記同様 PTS 駆動。`model_weights/rife.pth`（手順1〜2）が同じく必要で、protection / サポーターコードには一切触れない。

```bash
# パス1: 復元のみ（frame-gen なし）。2回目のエンコードで世代劣化を重ねないよう、
# 準ロスレスな中間ファイルにする（例: 高品質な cq）:
jasna --input in.mp4 --output restored.mkv --encoder-settings cq=16
# (または公式バイナリで restored.mkv を作成)

# パス2: フレーム生成のみ
jasna-framegen --input restored.mkv --output out2x.mkv --factor 2x
jasna-framegen --input restored.mkv --output out4x.mkv --factor 4x
```

主なオプション: `--factor {2x,4x}`、`--backend {rife,rtx}`、`--model-path <rife.pth>`（RIFE）、`--rtx-mode {low,medium,high}`（RTX）、`--codec {hevc,av1}`、`--encoder-settings <k=v,...>`、`--device cuda:0`、`--no-fp16`（RIFE）。出力品質は既定で jasna のエンコーダプロファイル（cq=25）。`--encoder-settings` で上書き可能。全オプションは `jasna-framegen --help`。確認は手順4と同じ（`ffprobe` でフレームレートが約2x/4x、尺不変、音声同期）。

### フォルダ一括 + 命名規則

`--input` がフォルダの場合、`--output` は出力フォルダとして扱われ、中の全動画を 1 つの RIFE モデル（1 回だけ構築して再利用）で処理する。フレーム生成は動画専用なので、フォルダ内の画像はスキップされる。出力ファイル名は `--output-pattern` で制御（`jasna` 本体と同じ意味）: `{original}` は入力 stem、既定は `{original}_out`（各入力の拡張子を維持）。

```bash
# in_dir/ の全動画を 2x にして out_dir/ へ（既定名: <name>_out.<ext>）
jasna-framegen --input in_dir --output out_dir --factor 2x

# 命名カスタム例: clip.mkv -> clip_2x.mkv
jasna-framegen --input in_dir --output out_dir --factor 2x --output-pattern "{original}_2x.mkv"
```

フォルダ実行ではファイルごとに `[i/N] name -> out` を表示し、色域非対応のファイルはスキップして継続する。`--output-pattern` が 2 つの入力を同じ出力に割り当てる（または入力を上書きする）場合は事前にエラーになる。

---

## トラブルシュート

- **`RTX Video Frame Generation is not available in the installed nvidia-vfx 0.1.0.1 ...`**: venv が 0.2.0.0 固定より前のもの。`uv pip install nvidia-vfx==0.2.0.0` で更新するか、`rife` を使う。
- **`RTX Video Frame Generation requires an NVIDIA Ada (RTX 40 series) or newer GPU ...`**: SDK の effect は Turing/Ampere で動かない。`--frame-gen-backend rife` を使う。
- **Linux のドライバ 580.0〜580.81 で `rtx` が SDK 内部で失敗する**: effect には 580.82 以降（または 570.190 以降 / 590.44 以降）が必要。ドライバを更新する。
- **`RIFE weights not found: ...`**: `model_weights\rife.pth` が無い。手順1〜2で作成するか `--frame-gen-model-path` を指定。
- **`RIFE state_dict loaded non-strictly (missing=.., unexpected=..)`**: `flownet.pkl` を直接置いた場合に出る警告で、同梱IFNetとキーが合っていない。補間結果が壊れるので **TorchScript方式に切り替える**。
- **変換スクリプトの import エラー**: `--rife-repo` が Practical-RIFE のチェックアウト（`train_log/` を含む）を指しているか確認。別バージョンで `flownet.forward` の戻り値が異なる場合は、`scripts/make_rife_torchscript.py` の `RifeTorchScriptWrapper.forward` を調整する。

## 注意（ライセンス）

`scripts/make_rife_torchscript.py` 自体は公開可能。ただし **RIFE のモデルコードと重み（`flownet.pkl` / 生成した `rife.pth`）は Practical-RIFE 由来で非商用条項がある**。再配布前に上流ライセンスを確認すること。https://github.com/hzwer/Practical-RIFE

## 補足（実装メモ）

- RIFEは**デフォルトでfp16**で動く（パイプラインの `--fp16` に追従。`--fp16` 無効時はfp32）。同梱IFNetのwarpはサンプリンググリッドをflowと同じdtypeで生成するため、`grid_sample` のdtype一致要求をfp16でも満たす。外部のTorchScriptチェックポイントがfloat32グリッドを内部に焼き込んでいる場合は、初期化時のプローブ推論で検出して**自動的にfp32へフォールバック**する（警告ログが出る。動作は継続）。出力はどちらでもuint8経由で往復するため画質経路は不変。
- 実測の高速化（RTX 5060 Ti、1080p、`--frame-gen 2x`、lada-yolo-v4、エンドツーエンド）: fp32チェックポイントで16.5fps → **fp16で31.4fps（約1.9倍）**。fp16/fp32出力間のPSNRは平均約50dBで見た目は同一であり、画質を理由にfp32を選ぶ必要はない。
- 補間はblend-encodeスレッド上で全解像度実行される（v1）。TRT化と専用スレッド化は将来拡張。
- `rtx` バックエンド（`jasna/framegen/rtx_frame_generator.py`）は、`rtx-super-res` が `VideoSuperRes` を包むのと同じ作法で nvvfx の `VideoFrameGeneration` を包む: CHW uint8 → contiguous な float32 [0,1] → `run_at_timestep(prev, cur, t, stream_ptr=<現在の torch ストリーム>)` → clone（SDK は出力バッファを 1 つ使い回す）→ uint8。effect は入力サイズに束縛されるため、最初のフレーム対から遅延ロードし、サイズが変わると再ロードする。
