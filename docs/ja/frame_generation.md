# フレーム生成（フレームレート倍化）の使い方

`--frame-gen {none,2x,4x}` で出力動画のフレームレートを2倍/4倍にする（ファイル出力のみ、`--stream`非対応）。
中間フレームをAI補間で生成し、元のPTS間に新しいPTSを挿入する。音声は元のタイムコードを保持するため尺と同期は不変。

バックエンドは `--frame-gen-backend {rife,rtx}`：
- `rife`（既定）: ニューラル補間。**現在利用可能**。重みを別途用意する。
- `rtx`: NVIDIA RTX Video Frame Generation。SDK（`nvidia-vfx`のフレーム生成Effect）が未出荷のため**現状は明示エラー**。出荷後に有効化予定。

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

主なオプション: `--factor {2x,4x}`、`--backend {rife,rtx}`、`--model-path <rife.pth>`、`--codec {hevc,av1}`、`--bit-depth {auto,8,10}`、`--encoder-settings <k=v,...>`、`--device cuda:0`、`--no-fp16`。出力品質は既定で jasna のエンコーダプロファイル（cq=25）。`--encoder-settings` で上書き可能。全オプションは `jasna-framegen --help`。確認は手順4と同じ（`ffprobe` でフレームレートが約2x/4x、尺不変、音声同期）。

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

- **`RTX Video Frame Generation is not available ...`**: `--frame-gen-backend rtx` は未出荷。`rife` を使う。
- **`RIFE weights not found: ...`**: `model_weights\rife.pth` が無い。手順1〜2で作成するか `--frame-gen-model-path` を指定。
- **`RIFE state_dict loaded non-strictly (missing=.., unexpected=..)`**: `flownet.pkl` を直接置いた場合に出る警告で、同梱IFNetとキーが合っていない。補間結果が壊れるので **TorchScript方式に切り替える**。
- **変換スクリプトの import エラー**: `--rife-repo` が Practical-RIFE のチェックアウト（`train_log/` を含む）を指しているか確認。別バージョンで `flownet.forward` の戻り値が異なる場合は、`scripts/make_rife_torchscript.py` の `RifeTorchScriptWrapper.forward` を調整する。

## 注意（ライセンス）

`scripts/make_rife_torchscript.py` 自体は公開可能。ただし **RIFE のモデルコードと重み（`flownet.pkl` / 生成した `rife.pth`）は Practical-RIFE 由来で非商用条項がある**。再配布前に上流ライセンスを確認すること。https://github.com/hzwer/Practical-RIFE

## 補足（実装メモ）

- RIFEは**デフォルトでfp16**で動く（パイプラインの `--fp16` に追従。`--fp16` 無効時はfp32）。同梱IFNetのwarpはサンプリンググリッドをflowと同じdtypeで生成するため、`grid_sample` のdtype一致要求をfp16でも満たす。外部のTorchScriptチェックポイントがfloat32グリッドを内部に焼き込んでいる場合は、初期化時のプローブ推論で検出して**自動的にfp32へフォールバック**する（警告ログが出る。動作は継続）。出力はどちらでもuint8経由で往復するため画質経路は不変。
- 実測の高速化（RTX 5060 Ti、1080p、`--frame-gen 2x`、lada-yolo-v4、エンドツーエンド）: fp32チェックポイントで16.5fps → **fp16で31.4fps（約1.9倍）**。fp16/fp32出力間のPSNRは平均約50dBで見た目は同一であり、画質を理由にfp32を選ぶ必要はない。
- 補間はblend-encodeスレッド上で全解像度実行される（v1）。TRT化と専用スレッド化は将来拡張。
