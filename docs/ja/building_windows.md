# Building Jasna for Windows

Windows で Jasna のセットアップを行い、**ソースから実行**する手順。

> **ブランチとベース。** 本ガイドは `0.11.1+modi`（upstream `2fc3e899` = v0.11.1 タグ + docs コミット 1 つ）の手順です。GPU スタックは `pyproject.toml` の upstream のピンに追従しています（torch 2.12.1+cu130 / torchvision 0.27.1+cu130 / torch-tensorrt 2.12.0 / tensorrt 10.16.1.11 / comfy-kitchen 0.2.36 / triton-windows 3.7.1 / ultralytics 8.4.96 以上）。本ブランチが足しているのは `nvidia-vfx==0.2.0.0`、`nvidia-cudnn-frontend`、任意の `nvidia-rtx` extra だけです。ブランチが upstream に加えている変更とその理由は [changes_vs_upstream.md](changes_vs_upstream.md) にあります。
>
> **検証。** 最後に実機検証したのは **v0.10.0 ベース**（2026-08-13、Windows 11、RTX 5060 Ti）です。現在の `0.11.1+modi` ベースは **Windows では未再検証**です。その間の変化は[付録 B](#付録-b-本セットアップの履歴)にまとめてあり、Linux では検証済みです（[building_linux.md](building_linux.md)）。
>
> **パッケージング。** このフォークはソースから実行します。実験的な Nuitka スクリプト（`scripts\build_nuitka.py`）で、本ガイドの venv から `jasna.exe` + `jasna-framegen.exe` のスタンドアロン配布物を作れます（[frozen_build.md](frozen_build.md)、8 節）。LTX と支援者モデルは upstream の公式リリースでしか動きません。

---

## 1. 前提ソフトウェア

| 種別 | 要件 | winget ID / 入手元 | 備考 |
|---|---|---|---|
| OS | Windows 10/11 x64 | n/a | PowerShell 7+ 推奨 |
| Git | Git for Windows | `Git.Git` | リポジトリ取得用 |
| uv | 最新版 | `astral-sh.uv` | Python の自動管理を含む |
| Python | 3.13 | n/a (uv が自動管理) | `uv venv --python 3.13` 実行時に未インストールなら自動取得。`requires-python` は `>=3.12` だが、本ガイドの検証は 3.13 のみ（3.14 は cu130 GPU スタックとの組み合わせが未検証） |
| NVIDIA Driver | **610+** | NVIDIA 公式 or NVIDIA App | 起動チェックが Windows では 610 以上を要求。GPU は compute capability 7.5+ |
| ffmpeg / ffprobe | **v8**（実行時 CLI） | [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases) の `ffmpeg-n8.1-latest-win64-gpl-shared-*.zip` 推奨（下記） | 起動チェックが `ffprobe` のメジャーバージョン 8 を要求 |
| VS Build Tools | 2022 (Desktop development with C++) | `Microsoft.VisualStudio.2022.BuildTools` | **任意の VALI wheel ビルド（5.3 節）と Nuitka バンドル（[frozen_build.md](frozen_build.md)）の場合のみ** |

基本セットアップに CUDA Toolkit は不要で、コンパイルするものもありません。torch / tensorrt の wheel が CUDA ランタイムを同梱し、PyAV も wheel で入ります。拡張をコンパイルするのは任意の VALI バックエンドのビルド（5.3 節）だけで、その追加前提（VS Build Tools、CUDA Toolkit）はそこに記載しています。

### 1.1 ffmpeg の選択と PATH への追加

CLI（`ffmpeg.exe` / `ffprobe.exe`）だけなら gyan.dev の [release-full ビルド](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z)でも足ります（起動チェックのエラーメッセージが案内するのもこれ）。BtbN の **shared ビルド**を推奨するのは、同じ展開物で次の 2 つも賄えるためです:

- torchcodec バックエンド（オプション）が実行時に必要とする FFmpeg の DLL 群（`bin\` の `avcodec-62.dll` 等）
- 任意の VALI バックエンドのビルド（5.3 節）がリンクする `include\` と `lib\`

展開して `bin\` を PATH に追加します:

```powershell
$Workspace = "C:\jasna-dev"          # ← 任意のディレクトリ
# https://github.com/BtbN/FFmpeg-Builds/releases から
# ffmpeg-n8.1-latest-win64-gpl-shared-*.zip を取得し $Workspace\ffmpeg-n8.1-shared に展開

$env:PATH = "$Workspace\ffmpeg-n8.1-shared\bin;$env:PATH"
ffmpeg -version                      # ffmpeg version n8.1 (メジャー 8) が表示されること
ffprobe -version
```

恒久化したい場合はユーザー環境変数の `Path` に `bin\` を追加する（新セッションから有効）。

### 1.2 winget で一括インストール

```powershell
winget install --id Git.Git       -e --source winget
winget install --id astral-sh.uv  -e --source winget

# 任意の VALI wheel ビルド（5.3 節）または Nuitka バンドルの場合のみ:
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --silent `
  --override "--wait --quiet --add ProductLang En-us --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

NVIDIA ドライバ（610+）は NVIDIA 公式または NVIDIA App から導入する。

---

## 2. リポジトリのクローン

クローンするのは `jasna` だけです（5.3 節の VALI フォークは任意で、その節でクローンします）。

```powershell
$Workspace = "C:\jasna-dev"          # ← 任意のディレクトリ
mkdir $Workspace -Force | Out-Null
cd $Workspace

git clone -b modi https://github.com/sh202603/jasna.git   # modi ブランチ（本フォークの既定）
```

> 新しい PowerShell セッションでは再度 `$Workspace = "..."` を設定すること。

---

## 3. Python 仮想環境の作成

`jasna` リポジトリ配下に `.venv` を作る。以降の `uv pip` はすべてこの venv をアクティブにした状態で実行する。

```powershell
cd $Workspace\jasna
uv venv --python 3.13
.\.venv\Scripts\Activate.ps1
python --version                     # -> Python 3.13.x
```

uv の managed Python で問題ありません。

---

## 4. PyAV

ビルドするものはありません。GPU パスは PyAV の current_ctx API（torch が初期化した CUDA コンテキストを NVDEC/NVENC と共有する）と av 18.1 の CUDA stream 明示指定を使い、`pyproject.toml` は `av==18.1.0`（upstream の VALI wheel がリンクしているのと同じリリース）に固定しています。5 節のインストールで PyPI から解決されます。この wheel は十分新しい nv-codec-headers でビルドした FFmpeg 8 を同梱しており（`hevc_nvenc` は `lookahead_level` を受理）、BtbN ビルドも delvewheel 修復も関係しません。PATH に置く BtbN の `bin\` は CLI ツールと任意の torchcodec バックエンドのためだけのものです。

公式 wheel は Linux で検証済みです。Windows で最後に検証した環境は暫定の自前ビルド wheel を使っており、そのコミットは 18.1.0 に含まれています。その経緯と、venv に暫定 wheel が残っているかの見分け方は[付録 B](#付録-b-本セットアップの履歴)にあります。

---

## 5. jasna 本体のインストール

GPU スタックは extras に分かれている（`nvidia` = NVIDIA スタック、`amd` = ROCm 用）。NVIDIA ビルドでは `nvidia` extra が `torch==2.12.1+cu130` / `torchvision==0.27.1+cu130` / `torch-tensorrt==2.12.0` / `nvidia-vfx==0.2.0.0` / `comfy-kitchen==0.2.36` / `triton-windows==3.7.1.post27` を入れる。一部は標準の PyPI には存在しないため、**PyTorch の wheel インデックスを `--extra-index-url` で指定し、さらに 2 つのフラグが必要**:

```powershell
cd $Workspace\jasna
uv pip install -e .[dev,nvidia] `
    --extra-index-url https://download.pytorch.org/whl/cu130 `
    --index-strategy unsafe-best-match `
    --prerelease=allow
```

> **⚠️ Windows では、このコマンドはそのままだと解決に失敗します。** `[nvidia]` extra の `python_vali==4.8.8` ピンに PyPI の Windows wheel が存在しないためです（`No wheels with a matching platform tag` エラー）。回避手順: まず `[dev,torchcodec]`（`nvidia` 抜き）で venv を作り、5.3 節でフォーク wheel（4.8.8）をビルド・導入してから、上記の `[dev,nvidia,...]` を再実行します。導入済みの 4.8.8 がピンを満たすため、2 回目は解決に成功します（wheel を `dist\` に置いたまま `--find-links <ワークスペース>\vali\dist` を足して 1 回で解決させることもできます。実機確認済み）。これは最後の Windows 検証（2026-08-13）時点の PyPI の状態です。フォーク wheel なしでコマンドが解決するなら、PyPI に Windows wheel が公開されたということで、この回避は不要になっています。

各フラグの役割:

- `--extra-index-url https://download.pytorch.org/whl/cu130`: `torch+cu130` / `torchvision+cu130` / `torch-tensorrt+cu130` ホイールの取得先
- `--index-strategy unsafe-best-match`: `pyproject.toml` の `torch-tensorrt==2.12.0` を PyTorch インデックス上の `torch-tensorrt==2.12.0+cu130`（ローカルバージョン付き）で満たすために必要。デフォルトの first-index 戦略では拒否される
- `--prerelease=allow`: `torch-tensorrt` の推移依存 `nvidia-cuda-runtime-cu13==0.0.0a0` がプレリリース版のため必要

`[dev]` で `nuitka>=2.4`, `pytest`, `pytest-cov`, `scikit-build`, `cmake`, `ninja` が入る。`[nvidia]` で GPU スタック（torch cu130 / TensorRT / torch-tensorrt / nvidia-vfx）が入る — **指定を忘れると torch が入らず起動しない**。

**注: `nvidia-vfx` は 0.2.0.0（VFX SDK 1.3.0）に固定している。** PyPI 上のエントリは殻で、上記インストール中に `pypi.nvidia.com` から実 wheel（Windows x64 は 436 MB）を取ってくる。0.2.0.0 は `--frame-gen-backend rtx`（[frame_generation.md](frame_generation.md)）の `VideoFrameGeneration` effect を提供し、TensorRT を同梱せず、torch の CUDA 13 ランタイムと並んで自前の `cudart64_12.dll` を持つ。wheel の取得、DLL ロード、cudart 12/13 の共存、`--fp8-recon` との cuDNN 併用は Windows（RTX 5080）で確認済み。0.1.0.1 が入った古い venv は `uv pip install nvidia-vfx==0.2.0.0` でその場で更新できる（[付録 B](#付録-b-本セットアップの履歴)）。

**オプション: torchcodec バックエンド。** 実験的な torchcodec のデコード/エンコード経路（`--video-backend torchcodec`/`auto`）を使う場合は、`torchcodec` extra を追加し、同じフラグで `.[dev,nvidia,torchcodec]` を入れる:

```powershell
uv pip install -e .[dev,nvidia,torchcodec] `
    --extra-index-url https://download.pytorch.org/whl/cu130 `
    --index-strategy unsafe-best-match `
    --prerelease=allow
```

これで `torchcodec>=0.15.0` が入る。torchcodec は実行時に FFmpeg の DLL 群を必要とするため、BtbN shared ビルドの `bin\` を PATH に置く（1.1節）。通常のセットアップには不要で、既定の `native` バックエンド（PyAV）は torchcodec なしで動作する。詳細は [torchcodec_backend.md](torchcodec_backend.md)。

**補足: FP8 復元バックエンドに追加のインストール手順は不要。** 依存 `nvidia-cudnn-frontend` と（Windows では）`triton-windows` は `pyproject.toml` の通常依存で、上記コマンドで一緒に入る。cuDNN ランタイム（9.17 以上）は torch cu130 wheel に同梱済み。機能は実行時 opt-in（`--fp8-recon`、FP8 対応 GPU sm89 以上が必要）で、使えない環境では TensorRT エンジンにフォールバックする。Windows 11 + RTX 5080 で実機検証済み。`--log-level info` で走らせると `CudnnFP8Upsample: enabled` が出る（既定の `--log-level error` ではこの行は出ない）。詳細は [fp8_recon.md](fp8_recon.md)。

**任意: TensorRT-RTX フレーバー。** `nvidia` の代わりに `nvidia-rtx` extra を入れると、TensorRT スタックが TensorRT-RTX（JIT コンパイル）に切り替わり、エンジンビルドが分単位から秒単位になる。両 extra は 1 つの venv に共存できないため、専用 venv を使う。エンジンは `.rtx` タグ付きの名前（`.rtx.win`）でキャッシュされるため、両フレーバーで 1 つの `model_weights` ディレクトリを共有できる。mmengine パッチ（§5.1）はこの venv にも必要。注意: `python_vali` は PyPI に Windows wheel が無いため、ローカルビルドの wheel を `--find-links` で供給する（§5 参照）。詳細と実測値（Windows 11 / RTX 5060 Ti 検証済み）は [tensorrt_rtx.md](tensorrt_rtx.md)。


### 5.1 mmengine パッチの適用 (torch 2.6+ 対応)

`mmengine.runner.checkpoint` の `torch.load` 呼び出しに `weights_only=False` を追加する。torch 2.6 以降は既定が `weights_only=True` に変更されたため、これを当てないと既存の `.pth` チェックポイント読み込みが失敗する。

`patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff` を venv 内の `site-packages` に適用する。`patch.exe` が無くてもよいよう、Python 製の `patch` パッケージを一時的に入れて当てる。

```powershell
cd $Workspace\jasna

uv pip install patch
uv run --no-project python -m patch -p1 -d .venv\Lib\site-packages `
    patches\fix_loading_mmengine_weights_on_torch26_and_higher.diff
uv pip uninstall patch
```

確認:

```powershell
Select-String -Path .venv\Lib\site-packages\mmengine\runner\checkpoint.py -Pattern "weights_only=False"
# → 行が 1 件ヒットすれば適用済み
```

> このパッチは `uv pip install -e .[dev,nvidia]` で `mmengine` を入れ直すたびに上書きされて消える。再インストール後は再度当てる。

### 5.2 YOLO 検出モデルに追加パッケージは不要

YOLO 系検出モデル（`lada-yolo-*`、`zelefans-vr-yolo-v2`）は、TensorRT エンジンをビルドする前に ultralytics が ONNX へエクスポートする。`onnx` は通常依存として入っており、エクスポートは `simplify=False` で走るため `onnxslim` も `onnxruntime` も要らない。RF-DETR モデルは事前ビルド済みの `.onnx` を TensorRT が直接パースする。

### 5.3 オプション: VALI NVDEC デコードバックエンドのビルド（`python_vali` フォーク）

native デコードは、まず VALI NVDEC デコーダを試し、使えなければ PyAV へエスカレーションします（v0.10.0 からは `JASNA_DECODE_BACKEND` 環境変数〔`auto` / `vali` / `pyav-hw` / `pyav-sw`〕でこの連鎖を明示指定できます。modi の `--video-backend` はその上位層で、`native` を選んだときにこの連鎖が働きます）。フォーク専用 API `DecodeSingleSurfaceAsyncDetailed` を持つ wheel が無いと、リーダーは警告を出して毎回 PyAV にフォールバックします（本節なしでもすべて動作します）。Windows では `[nvidia]` extra の解決にも 4.8.8 の wheel が要るため（5 節の注記）、フォーク wheel のビルドが実質前提になります。

1 節に対する追加の前提:

- **VS Build Tools 2022**（ビルドは vcvars64 を通した cmd / Developer PowerShell セッションで行う）
- **CUDA Toolkit 13.x**（`nvcc`。vali の CMake は `CUDA_PATH` から toolkit を解決する）
- **FFmpeg 8 の shared ビルド**: vali の CMake は `FFMPEG_ROOT` を **`C:\Program Files\ffmpeg8` にハードコード**しているため、`include\`/`lib\`/`bin\` を持つ shared ビルド（gyan.dev の full-shared 等）をそこに置くか、junction を張る（`New-Item -ItemType Junction -Path "C:\Program Files\ffmpeg8" -Target <展開先>`）。BtbN ビルド（1.1 節）でも要件は同じ。
- cmake、ninja、scikit-build、`pkg_resources` 入りの setuptools（<80）: いずれも 5 節の `[dev]` で venv に導入済み。

```powershell
cd $Workspace
git clone https://codeberg.org/Kruk2/vali.git       # 既存 checkout があれば省略
cd vali
git checkout 0e5c01e      # = 4.8.8 ピン: DecodeSingleSurfaceAsyncDetailed、ASF/VC-1 タイムスタンプ修正、decoder-open の GIL 解放（前回検証の f4a67f8 でも可）
git submodule update --init --recursive
```

ビルドは MSVC + CUDA の環境変数を通した **cmd セッション**で `setup.py bdist_wheel` を実行する（vcvars64 が PowerShell を汚さないよう bat 化が楽）:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CL=/utf-8"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d <ワークスペース>\vali
<ワークスペース>\jasna\.venv\Scripts\python.exe setup.py bdist_wheel
REM -> dist\python_vali-4.8.8-cp313-cp313-win_amd64.whl
```

Linux と違い、Windows の wheel は FFmpeg の DLL 群を `python_vali\` 内に同梱するため（CMake の `PYPI_BUILD=0` 既定）、delvewheel は不要です。導入と確認:

```powershell
uv pip install --force-reinstall --no-deps (Get-Item $Workspace\vali\dist\python_vali-4.8.8-*.whl)
python -c "import python_vali as v; print(hasattr(v.PyDecoder, 'DecodeSingleSurfaceAsyncDetailed'))"   # -> True
# jasna を --log-level info で実行すると "Using VALI NVDEC decoder for <file>" が出ます
```

> フォークコミット `f4a67f8`（= 4.8.8 ピン、2026-08-13、RTX 5060 Ti + CUDA 13.2）で Windows ビルド・パイプライン実走を検証済み。`0e5c01e` は Linux で検証済み（2026-10-16）で、Windows ビルドは未了。前回検証はフォークコミット `3ad0d54`（= 4.8.7 ピン、2026-07-30、RTX 5080 + CUDA 13.2 + gyan.dev 8.1 shared）: パイプラインの 2 つのデコードパスが両方 VALI を通り、同梱テストクリップの出力は PyAV デコード時と**ビットストリーム md5 完全一致**、av wheel および torchcodec バックエンドとの共存も確認（av は delvewheel 同梱、vali は自前同梱、torchcodec は PATH 解決で、それぞれ別コピーの FFmpeg DLL を持つが、Windows は DLL ごとにシンボル空間が独立しており衝突しない）。

---

## 6. モデルウェイトとアセットの配置

`$Workspace\jasna\model_weights\` に以下のファイルを置く（upstream のリリースパッケージから取り出す）:

- `lada_mosaic_restoration_model_generic_v1.2.pth`
- `rfdetr-v6.onnx`（v0.9.0 以降の既定検出器。4K 向け `rfdetr-v6-large.onnx` と VR180 向け `rfdetr-vr-v1.onnx` は任意、旧 `rfdetr-v5.onnx` も引き続き使える）
- `lada_mosaic_detection_model_v4_fast.pt`
- `lada_vr_mosaic_detection_model_v2_accurate.pt`（VR180 向け YOLO 検出モデル `zelefans-vr-yolo-v2` の実体）

ソースから実行すると Jasna は `model_weights\` を自動解決する（付録 A.1）ため、ここに置くだけでよい。

検出モデルは `model_weights\` にあるファイルから自動発見されるため、4 つ目が無くても他機能は動作する。その場合は `zelefans-vr-yolo-v2` がモデル一覧に現れず、VR180 の検出に使えないだけである。

テストクリップ 2 本は通常リポジトリに同梱され `$Workspace\jasna\assets\`（`test_clip1_1080p.mp4`, `test_clip1_2160p.mp4`）にある。無い場合は upstream リリースから抽出する。

### 6.1 オプション: RIFE フレーム補間モデル (`--frame-gen` 用)

フレームレート倍化 (`--frame-gen {2x,4x}`) を既定の `rife` バックエンドで使う場合のみ必要。RTX 40 シリーズ以降なら `--frame-gen-backend rtx`（5 節で入る `nvidia-vfx` wheel の NVIDIA RTX Video Frame Generation）は重みを一切必要としない（[frame_generation.md](frame_generation.md)）。RIFE 重みは**同梱されない**（非商用条項のため）ので、`scripts/make_rife_torchscript.py` で TorchScript チェックポイントを自分で作成する。

1. Practical-RIFE をクローンし、4.x のモデルパッケージ（**v4.25 で動作確認**）をダウンロードして `<repo>\train_log\` に `RIFE_HDv3.py`, `IFNet_HDv3.py`, `flownet.pkl` が揃う状態にする:
   ```powershell
   git clone https://github.com/hzwer/Practical-RIFE
   # README に従いモデルパッケージを Practical-RIFE\train_log\ にダウンロード・展開
   ```
2. プロジェクトの venv で TorchScript に変換する（`$Workspace\jasna` で実行）:
   ```powershell
   .\.venv\Scripts\python.exe scripts\make_rife_torchscript.py `
       --rife-repo C:\path\to\Practical-RIFE `
       --output model_weights\rife.pth --validate
   ```

**ソースから実行**する場合は `model_weights\rife.pth` を自動で参照する（他のウェイトと同じリゾルバ経由。または `JASNA_MODEL_WEIGHTS_DIR` をそれを含むフォルダに向ける）。詳細手順: [frame_generation.md](frame_generation.md)。

### 6.2 オプション: FlashVSR 二次復元（実験的）

FlashVSR（`--secondary-restoration flashvsr` / `flashvsr-inline`）は別リポジトリのチェックアウトと専用 venv を必要とする。チェックアウトには fork [`sh202603/FlashVSR_plus`](https://github.com/sh202603/FlashVSR_plus) を使う（inline に要る tiny-long の修正と `--flashvsr-accel` の高速化を含み、`uv sync` で依存が揃う）。Windows 対応済み。16 GB カードの 1080p では `--flashvsr-tiles 2` を使う（inline は `--max-clip-size`（既定 90）にそのまま従い、かつての 32 フレーム上限は無い）。セットアップ手順は [flashvsr.md](flashvsr.md) を参照。

---

### 6.3 オプション: SeedVR2 一次復元（実験的）

SeedVR2（`--restoration-model-name seedvr2`）は別リポジトリ `ComfyUI-SeedVR2_VideoUpscaler` のチェックアウトと専用 venv、`model_weights/` への LoRA（約 90 MB）配置を必要とする。パッチは不要で、jasna の venv には何も入らない。セットアップ手順は [seedvr2.md](seedvr2.md) を参照。

---

### 6.4 オプション: SwiftVR 二次復元と SwiftVR distill（実験的）

SwiftVR（`--secondary-restoration swiftvr` / `swiftvr-inline`。`--ltx-crop-vsr` の `swiftvr` バックエンドも同じ）は fork [`sh202603/SwiftVR`](https://github.com/sh202603/SwiftVR)（`modi` ブランチ）のチェックアウトと `uv sync` の専用 venv、約 20 GB のチェックポイントを必要とする。小さな蒸留モデル（`--secondary-restoration swiftvr-distill`。`--ltx-crop-vsr` の `swiftvr-distill` バックエンドも同じ）は [okatti/swiftvr-distill](https://huggingface.co/okatti/swiftvr-distill) の `swiftvr-distill.pt` を `model_weights\` に置くだけでよい。セットアップ手順と Windows での注意事項は [swiftvr.md](swiftvr.md) を参照。

---

## 7. ソースから実行 / 動作確認

1〜6節が完了したら、venv 内でソースチェックアウトから直接 Jasna を実行する:

```powershell
cd $Workspace\jasna
python -m jasna --version    # -> 0.11.1+modi
python -m jasna --help
jasna --input assets\test_clip1_1080p.mp4 --output $env:TEMP\out.mp4   # 短いクリップを処理
python -m jasna              # GUI を起動（引数なし）
```

> **GUI の起動は `python -m jasna`（引数なし）です。** console script の `jasna` は `jasna.main:main` に直結しており、引数なしでも GUI ディスパッチ（`jasna\__main__.py`）を通りません。frozen ビルド（`jasna.exe`）の引数なし起動だけが GUI に入ります。

確認事項:

- `ffmpeg` / `ffprobe`（v8）が `PATH` 上にあること（1.1節）。
- NVIDIA ドライバが **610 以上**であること（起動チェックが拒否します）。
- インストールパスが ASCII のみであること（起動時に強制。ユーザー名に非 ASCII を含む環境では展開先に注意）。

> **最初の処理実行は遅い。** 初回使用時に GPU 向け TensorRT エンジンがコンパイルされる（15〜60 分）。`model_weights\` 内のウェイト隣（`<モデル名>_sub_engines\` 等）にキャッシュされ、以降は再利用される。v0.9.1 ベース以降のエンジンキャッシュはそのまま有効である（エンジン層は v0.11.1 まで無変更）。`--max-clip-size` の変更で再コンパイルされることもない。

---

## 8. パッケージング / frozen ビルド

このフォークは独自の実験的 Nuitka ビルドスクリプト（`scripts\build_nuitka.py`）を同梱しており、本ガイドの venv から `jasna.exe` と `jasna-framegen.exe` から成るスタンドアロン配布物を生成できます。手順、バンドルの構成、制限は [frozen_build.md](frozen_build.md) にあります。スクリプトはブランチに追従しています（PyAV メディア層、rfdetr-v6 検出器、CUDA カーネル一式、外部復元器の worker スクリプト）が、ベースを上げるごとに Windows で再ビルドと動作確認が必要です。

upstream 自身のパッケージングツール（同じく Nuitka ベース）は**プライベートな submodule（`jasna/protection`）**にあり、このフォークには含まれません。この submodule に依存する機能（`unet-4x`、SD1.5 inpaint、ライセンス認証）は、このフォークの凍結ビルドでは動きません。

---

## トラブルシューティング

### PyAV / FFmpeg

- **エンコード開始直後に `ValueError: hevc_nvenc did not accept encoder option(s): ['lookahead_level']`**
  PyAV が古い nv-codec-headers の FFmpeg にリンクされている。公式 PyPI wheel（18.1 以降は対応 FFmpeg を同梱）では起きないため、`uv pip install --reinstall "av>=18.1"` で入れ直す。あえて自前ビルドする場合は付録 B.1 のとおり `INCLUDE` / `LIB` を BtbN ビルドに向ける。

- **`import av` が `ImportError: DLL load failed while importing _core` で失敗する**
  自前ビルドの wheel を delvewheel なしでインストールしている。Python 3.8+ は拡張モジュールの依存 DLL を PATH から探さないため、BtbN の `bin\` を PATH に置くだけでは解決しない。付録 B.1 の delvewheel repair を適用した wheel を入れ直すか、`uv pip install --reinstall "av>=18.1"` で公式 wheel に切り替える。

- **PyAV のビルドが `fatal error C1083: libavcodec/avcodec.h` で失敗する**
  FFmpeg の場所がコンパイラに伝わっていない。`PKG_CONFIG_PATH` は Windows では無視される（setup.py が pkg-config を呼ばない）。付録 B.1 のとおり `INCLUDE` / `LIB` に BtbN 展開物の `include` / `lib` を追記する。

- **エンコーダ/デコーダ初期化時に `current_ctx` 関連の `TypeError` 等で失敗する**
  av が 18.0.0 のまま（PyPI の旧リリース、またはバージョン表記が同じ `18.0.0` の暫定自前ビルド wheel）。`uv pip install "av>=18.1"` で入れ直す（4節）。

- **Jasna が起動拒否: ffprobe のバージョン違い**
  起動チェックは `ffprobe` の**メジャーバージョン 8** を要求する。v8 ビルドの `bin\` を `PATH` に追加する（1.1節）。

- **torchcodec バックエンドで FFmpeg DLL（`avcodec-62.dll` 等）が見つからない**
  torchcodec は実行時に FFmpeg の shared DLL を必要とする。BtbN shared ビルドの `bin\` を `PATH` に置く（1.1節）。

### jasna インストール

- **`uv pip install -e .[dev,nvidia]` が `No solution found ... no version of torch==2.12.0+cu130` で失敗する**
  PyTorch インデックスが未指定。`--extra-index-url https://download.pytorch.org/whl/cu130` を追加する。

- **`uv pip install -e .[dev,nvidia]` が `torch-tensorrt==2.12.0+cu130 ... unsatisfiable` で失敗する**
  uv のデフォルト index 戦略は first-index 限定で、`torch-tensorrt==2.12.0` を PyTorch インデックスの `2.12.0+cu130`（ローカルバージョン付き）で満たすことを拒否する。さらに推移依存 `nvidia-cuda-runtime-cu13==0.0.0a0` はプレリリース。`--index-strategy unsafe-best-match --prerelease=allow` を追加する（5節）。

### ランタイム

- **Jasna が `.pth`/`.onnx`/`.pt` で `FileNotFoundError`**
  `model_weights\` のファイル欠落、またはリゾルバが別の場所を見ている。3 つのウェイトを `$Workspace\jasna\model_weights\`（6節）に置くか、`JASNA_MODEL_WEIGHTS_DIR` をそれらが入ったフォルダに設定する。

- **起動時に NVIDIA ドライバのバージョンで拒否される**
  Windows の最低要件は **610** です。NVIDIA 公式からドライバを更新する。

- **初回起動が極端に遅い**
  異常ではない。TensorRT エンジン初回コンパイルに 15〜60 分かかる。`model_weights\` 内のウェイト隣にキャッシュされ 2 回目以降は高速化される。

---

## 付録 A: 本セットアップが依存するブランチ側の変更

以下はブランチのソースに適用済みの変更です。独立した `.patch` ファイルは同梱しません（同梱する 2 つのパッチ、mmengine 用と FlashVSR_plus 用は別パッケージに当てるものです。5.1 節と [flashvsr.md](flashvsr.md)）。ブランチの変更の全体は [changes_vs_upstream.md](changes_vs_upstream.md) にあります。

### A.1 `model_weights\` ディレクトリの自動解決

`jasna/model_weights_resolver.py` が `model_weights\` を次の順で探します: 環境変数 `JASNA_MODEL_WEIGHTS_DIR`、実行ファイルの隣（frozen ビルドのみ）、カレントディレクトリ、`jasna` パッケージの隣（ソースツリー）。最初に見つかったものを使い、どこを採用したかは `--log-level info` で起動時にログに出ます。CLI、検出モデルレジストリ、エンジンパス、GUI はハードコードの `Path("model_weights")` ではなくこれを経由するので、CLI は任意のディレクトリから実行でき、6 節はファイルを置くだけで済みます。

### A.2 torchcodec バックエンド向けの DLL 検索パス

Windows は拡張モジュールの依存 DLL を `PATH` から探さないため、torchcodec バックエンドのモジュール（`jasna/media/torchcodec_decoder.py`、`torchcodec_encoder.py`）は torchcodec パッケージのディレクトリと `CUDA_PATH\bin`（設定時）を `os.add_dll_directory()` で登録し、ブランチは upstream の `jasna/packaging/windows_dll_paths.py` に `torchcodec`、`tensorrt_rtx_libs`、同梱 `tools\`（frozen ビルドの FFmpeg 8 DLL）のディレクトリを足しています。native の PyAV 経路にはどれも不要です。torchcodec が素の名前（`avcodec-62.dll` など）でリンクする FFmpeg DLL は、`PATH` 上の shared ビルド（1.1 節）か frozen ビルドの `tools\` から供給する必要があります。

---

## 付録 B: 本セットアップの履歴

ベースごとのソース実行手順の変化と、本ガイドを Windows で検証した時点（新しい順）。不要になった手順は要約だけを残し、全文は本ファイルの git 履歴にあります。

- **v0.11.1 ベース（Windows では未検証）**: upstream が torch 2.12.1 / torchvision 0.27.1 を固定し、`comfy-kitchen==0.2.36` と `ultralytics>=8.4.96`（YOLO export に onnxruntime も onnxslim も要らなくなり、5.2 節は注記だけになった）を追加、`av==18.1.0` を厳密固定、デコードバックエンドの選択を `JASNA_DECODE_BACKEND` だけにした（in-code の `DECODE_BACKEND` トグルは廃止）。Linux ではこのベースを検証済み（[building_linux.md](building_linux.md)）で、VALI フォーク wheel も Kruk2 氏の `0e5c01e`（4.8.8 + ASF/VC-1 タイムスタンプ修正 + decoder-open の GIL 解放）で再ビルド済み。その Windows ビルドは未了。
- **nvidia-vfx 0.2.0.0 固定（2026-10-03。Windows 検証済み: RTX 5080、ドライバ 616.92）**: `--frame-gen-backend rtx` の `VideoFrameGeneration` effect のため、ブランチが `nvidia-vfx==0.2.0.0` を固定（upstream は無 pin）。0.2.0.0 は TensorRT を同梱せず（Linux で問題だった `libnvinfer` の衝突が消滅）、自前の `cudart64_12.dll` を持つ。それ以前の venv には 0.1.0.1 が入っている: `uv pip install nvidia-vfx==0.2.0.0`。
- **PyAV を PyPI から（2026-08-24。Windows では未再検証）**: av 18.1.0 が current_ctx API とエンコーダが使う CUDA stream 明示指定を含んで PyPI に公開され、B.1 の暫定自前ビルド wheel が不要になった。暫定 wheel は `18.0.0` を報告するため 5 節のインストールで置き換わる（すぐ替えるなら `uv pip install "av>=18.1"`）。`Test-Path .venv\Lib\site-packages\av.libs` が true なら暫定 wheel が残っている。公式 wheel は Linux で検証済み。Windows で最後に検証した環境（v0.10.0、2026-08-13）はコミット `f6f0a5e` から作った暫定 wheel を使っており、このコミットは 18.1.0 に含まれる。
- **v0.10.0 ベース（2026-08-13 検証: Windows 11、RTX 5060 Ti、ドライバ 610.88、Python 3.13.9、CUDA 13.2、ffmpeg 8.1）**: full pytest の失敗集合は素の v0.10.0 ベースラインと完全一致。CLI 実走 15 系統（`--cq`、章/字幕保存、`JASNA_DECODE_BACKEND`、post-export、近接ロック retarget、streaming 再起動を含む）と GUI スモーク（復元動画プレイヤーの VLC 経路と SoftwareClock 経路、キュー右クリック、A/B 比較、区間エディター）が合格。VALI フォーク wheel 4.8.8（`f4a67f8`）のビルドと検証（5.3 節）。このベースの新規: 環境変数 `JASNA_DECODE_BACKEND`、`python_vali==4.8.8` のピン（Windows では自前ビルドのフォーク wheel が `[nvidia]` 解決の前提になった。5 節）。
- **v0.9.1 ベース（2026-07-30 検証、RTX 5080）**: VALI NVDEC デコードバックエンドが任意の自前ビルド wheel として復活（フォークコミット `3ad0d54` = 4.8.7。両デコードパスが VALI を通り、出力は PyAV 実行とビットストリーム md5 一致、av wheel と torchcodec との共存も確認）。エンコーダが `CudaContext(cuda_stream=...)` で CUDA stream を渡すようになり、av 18.1.0 までは暫定 PyAV wheel が必須だった。TensorRT エンジンは一度全再コンパイル。
- **v0.8.1 ベース（2026-07-19 検証）**: GPU スタックが `nvidia` / `amd` extras へ（`.[dev,nvidia]` の由来）、`requires-python` は `>=3.12` に緩和。
- **v0.8.0 ベース（2026-07-18 検証、Windows 11、RTX 5080）**: upstream がメディア層を PyAV（NVDEC/NVENC）へ移し、`python_vali` / `PyNvVideoCodec` をランタイムから撤去。それらの C++ ビルドとともに、旧ガイドの前提だった CUDA Toolkit 13.2 と `CUDA_PATH` の固定、VS Build Tools 2022、`C:\Program Files\ffmpeg8\` への ffmpeg dev ビルド配置と junction 回避策（`FFMPEG_ROOT` のハードコードは vali のもので、5.3 節でだけ再登場する）、ビルド環境変数一式（`DISTUTILS_USE_SDK`、`CMAKE_GENERATOR`、`CL=/utf-8`、`FFMPEG_DIR`）と `setup-build-env.ps1`、`setuptools<80` の固定、MKVToolNix が不要になった。これらが環境に残っていても害はない（`CUDA_PATH` は付録 A.2 が参照する）。NVIDIA ドライバの最低要件は 610 に上がった。FP8 バックエンドの Windows 初検証もこのとき。

### B.1 暫定の PyAV wheel 自前ビルド（2026-07〜2026-08）

av 18.1.0 が未公開の間に必要だった手順で、ソースビルドが再び要る場合に備えて残す。リンク先は nv-codec-headers 12.2 以降でビルドした FFmpeg 8 でなければならない（BtbN の shared ビルドは条件を満たす。古いものでは全エンコードが開始直後に `ValueError: hevc_nvenc did not accept encoder option(s): ['lookahead_level']` で失敗する）。Developer PowerShell for VS 2022 で実行する。Windows 固有の要点は 3 つ: PyAV の setup.py は Windows では pkg-config を呼ばないので FFmpeg のヘッダとライブラリは MSVC の `INCLUDE` / `LIB` からしか伝わらない（無いと `fatal error C1083: libavcodec/avcodec.h`）。Python 3.8 以降は拡張モジュールの依存 DLL を `PATH` から探さないので FFmpeg DLL は delvewheel で wheel に同梱する（しないと `ImportError: DLL load failed while importing _core`）。できた wheel は `18.0.0` を報告するので、`av==18.1.0` のピンにより次の 5 節のインストールで置き換わる（使い続けるなら後で入れ直す）。

```powershell
cd $Workspace
git clone https://github.com/PyAV-Org/PyAV.git
cd PyAV
git checkout f6f0a5e                 # Linux/Windows 両実機で検証済み。v18.1.0 に包含
$env:INCLUDE = "$env:INCLUDE;$Workspace\ffmpeg-n8.1-shared\include"
$env:LIB     = "$env:LIB;$Workspace\ffmpeg-n8.1-shared\lib"
uv build --wheel                     # -> dist\av-18.0.0-cp311-abi3-win_amd64.whl
uv pip install delvewheel
delvewheel repair dist\av-18.0.0-cp311-abi3-win_amd64.whl `
    --add-path $Workspace\ffmpeg-n8.1-shared\bin -w dist\repaired
uv pip install --reinstall (Get-Item dist\repaired\av-*.whl)
python -c "import av; from av.video.frame import CudaContext; print(av.ffmpeg_version_info); print(hasattr(CudaContext, 'current_ctx'))"
```

delvewheel は DLL をハッシュ付き名で `av.libs\` に同梱するため、torchcodec がロードする FFmpeg DLL とはプロセス内で別コピーになるが、Windows は DLL ごとにシンボル空間が独立しており問題ない。
