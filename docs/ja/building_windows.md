# Building Jasna for Windows

Windows で Jasna のセットアップを行い、**ソースから実行**する手順。

> **検証状態（2026-07-30、`0.9.1+modi`・upstream `a7cdaf8` = v0.9.1 タグベース）**: 本ガイドの手順は Windows 11 + RTX 5080（driver 610.62、Python 3.13.9、CUDA 13.2、ffmpeg 8.1）で **a7cdaf8 ベースの `0.9.1+modi` を実機検証済み**です（full pytest の失敗集合は素の v0.9.1 ベースラインと一致、CLI 実走 9 系統 + AV1 入力 + GUI スモーク合格、VALI フォーク wheel の Windows ビルドも 5.3 節で検証済み。前回 d7a99bd ベースの検証は 2026-07-23）。Linux 側は [building_linux.md](building_linux.md) を参照してください。

> **本ガイドは `v0.9.1+modi` ブランチの手順です。** GPU スタック（**torch 2.12.0+cu130 / torchvision 0.27.0+cu130 / torch-tensorrt 2.12.0+cu130 / tensorrt 10.16.1.11**）は v0.7.2 期から変わらず、依存ピンは本ブランチの `pyproject.toml` に適用済みです。TensorRT が 10.16 系に留まるのは、`torch-tensorrt==2.12.0` が `tensorrt>=10.16.1,<10.17.0` を要求するためです（torch-tensorrt は TensorRT 11 に未対応）。

> **v0.8.0 でビルド手順は大幅に簡素化されました。** upstream v0.8.0 でメディア層が PyAV（NVDEC/NVENC）へ移行し、`python_vali` / `PyNvVideoCodec` の C++ ビルドが丸ごと不要になりました。これに伴い、旧ガイドの前提だった以下はすべて不要です:
>
> - CUDA Toolkit 13.2（と `CUDA_PATH` の固定）
> - VS Build Tools 2022（PyAV wheel を自前ビルドする場合を除く）
> - ffmpeg dev ビルドの `C:\Program Files\ffmpeg8\` への配置と junction 回避策（`FFMPEG_ROOT` ハードコードは vali のものでした）
> - ビルド環境変数一式（`DISTUTILS_USE_SDK`, `CMAKE_GENERATOR`, `CL=/utf-8`, `FFMPEG_DIR`）と `setup-build-env.ps1`
> - `setuptools<80` の固定
> - MKVToolNix（`mkvmerge`）
>
> 既存環境にこれらが残っていても害はありません（`CUDA_PATH` は torchcodec バックエンドの DLL 解決フォールバックが参照するため、残しておくと役に立つことがあります）。一方で **NVIDIA ドライバの要求が 610 以上に上がりました**（起動チェックが Windows では 610 未満を拒否します）。残る特殊手順は **PyAV wheel**（4節。PyAV 18.1.0 が PyPI に公開されるまでの暫定）だけです。
>
> **このブランチの主な追加機能:** RIFE による 2x/4x フレーム生成（[frame_generation.md](frame_generation.md)）、torchcodec バックエンド（[torchcodec_backend.md](torchcodec_backend.md)）、cuDNN FP8 復元（[fp8_recon.md](fp8_recon.md)）、FlashVSR 二次復元（[flashvsr.md](flashvsr.md)）。v0.7.2+modi にあった AV1 / 8bit / BT.601 と BT.2020 出力は upstream v0.8.0 に吸収されました。全差分は [changes_vs_upstream.md](changes_vs_upstream.md) を参照してください。

> **パッケージングについて:** このフォークは実験的な Nuitka ビルドスクリプト（`scripts\build_nuitka.py`）を同梱していますが、⚠️ **v0.8.0 のメディア層移行に未追従**です（旧 `python_vali` / `PyNvVideoCodec` の DLL 同梱を前提としたままで、現状では動作しない見込み）。詳細は [パッケージング / frozen ビルド](#8-パッケージング--frozen-ビルド)。

---

## 1. 前提ソフトウェア

| 種別 | 要件 | winget ID / 入手元 | 備考 |
|---|---|---|---|
| OS | Windows 10/11 x64 | n/a | PowerShell 7+ 推奨 |
| Git | Git for Windows | `Git.Git` | リポジトリ取得用 |
| uv | 最新版 | `astral-sh.uv` | Python の自動管理を含む |
| Python | 3.13 | n/a (uv が自動管理) | `uv venv --python 3.13` 実行時に未インストールなら自動取得。v0.8.1 で `requires-python` は `>=3.12` に緩和されたが、本ガイドの検証は 3.13 のみ（3.14 は cu130 GPU スタックとの組み合わせが未検証） |
| NVIDIA Driver | **610+** | NVIDIA 公式 or NVIDIA App | 起動チェックが Windows では 610 以上を要求。GPU は compute capability 7.5+ |
| ffmpeg / ffprobe | **v8**（実行時 CLI） | [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases) の `ffmpeg-n8.1-latest-win64-gpl-shared-*.zip` 推奨（下記） | 起動チェックが `ffprobe` のメジャーバージョン 8 を要求 |
| VS Build Tools | 2022 (Desktop development with C++) | `Microsoft.VisualStudio.2022.BuildTools` | **PyAV wheel を自前ビルドする場合のみ**（4節 (B)） |

CUDA Toolkit は不要になりました。torch / tensorrt の pip wheel が CUDA ランタイムを同梱しており、自前でコンパイルするネイティブ拡張は PyAV だけで、PyAV は CUDA を使いません（NVDEC/NVENC はリンク先 FFmpeg 側の機能です）。

### 1.1 ffmpeg の選択と PATH への追加

CLI（`ffmpeg.exe` / `ffprobe.exe`）だけなら gyan.dev の [release-full ビルド](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z)でも足ります（起動チェックのエラーメッセージが案内するのもこれ）。BtbN の **shared ビルド**を推奨するのは、同じ展開物で次の 2 つも賄えるためです:

- torchcodec バックエンド（オプション）が実行時に必要とする FFmpeg の DLL 群（`bin\` の `avcodec-62.dll` 等）
- PyAV wheel を自前ビルドする場合（4節 (B)）に必要な `include\`（ヘッダ）と `lib\`（MSVC 用インポートライブラリ `avcodec.lib` 等）

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

# PyAV wheel を自前ビルドする場合のみ (4節 (B)):
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --silent `
  --override "--wait --quiet --add ProductLang En-us --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

NVIDIA ドライバ（610+）は NVIDIA 公式または NVIDIA App から導入する。

---

## 2. リポジトリのクローン

クローンするのは `jasna` だけです。`vali` / `PyNvVideoCodec` は v0.8.0 のランタイムからは使われないため、チェックアウト不要になりました。

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

uv の managed Python で問題ありません（システム Python が必要だったのは旧ガイドの vali / PyNvVideoCodec ビルドで、v0.8.0 で廃止）。旧ガイドのビルドツール先入れ（`cmake ninja scikit-build "setuptools<80" wheel numpy`）も不要です。

---

## 4. PyAV wheel の導入（暫定手順。PyAV 18.1.0 公開までのつなぎ）

v0.8.0 の GPU パスは PyAV 18.1.0 で入る **current_ctx API**（torch が初期化済みの CUDA コンテキストを NVDEC/NVENC と共有する仕組み）を使いますが、18.1.0 は未公開で、PyPI の av 18.0.0 にはこの API がありません。

さらに、リンク先には **nv-codec-headers 12.2 以降でビルドされた FFmpeg 8** が必要です。古い nv-codec-headers の FFmpeg にリンクした PyAV では、`hevc_nvenc` に `lookahead_level` オプションが無いため、jasna の全エンコードが開始直後に `ValueError: hevc_nvenc did not accept encoder option(s): ['lookahead_level']` で失敗します（Linux の distro FFmpeg で実際に踏んだ症状。BtbN のビルドはこの条件を満たします）。

### (A) av 18.1.0 が PyPI に公開されていればそれを使う（最初に確認）

PyPI に av 18.1.0 が出ていれば本節の残りは不要です。PyPI のバイナリ wheel は対応済みの FFmpeg を同梱しているため、`uv pip install "av>=18.1"` するだけで済みます（5節のインストールでも解決されます）。

> **v0.9.1 ベースでの更新（2026-07-30、Windows 実機検証済み）:** v0.9.1 ベースのエンコーダは PyAV の CUDA stream 明示指定（`CudaContext(cuda_stream=...)`）を追加で必要とし、`61e4aa8` にはこの API がありません。PyAV **main** からビルドしてください（検証コミットは Linux/Windows とも `f6f0a5e`。以下のビルド手順自体は不変で、(B) は `f6f0a5e` で Windows 実機検証済み）。任意の VALI デコードバックエンド（フォーク `python_vali` wheel）の Windows ビルドは 5.3 節を参照。無くてもリーダーは PyAV にフォールバックし、全機能が動作します。

### (B) PyAV main から自前ビルド（実機検証済み）

CUDA stream API まで入った upstream main `f6f0a5e` をチェックアウトし、BtbN の shared ビルドにリンクして wheel を作ります。追加の前提は VS Build Tools 2022 だけです（**Developer PowerShell for VS 2022** セッションで実行する）。

> **pkg-config / pkgconf は不要です。** PyAV の setup.py は pkg-config を **Windows では呼びません**（非 Windows のみの分岐）。`PKG_CONFIG_PATH` を設定しても無視され、FFmpeg のヘッダ/ライブラリの場所は MSVC 標準の環境変数 `INCLUDE` / `LIB` からしか伝わりません。これを設定せずにビルドすると `fatal error C1083: libavcodec/avcodec.h: No such file or directory` で失敗します。

```powershell
cd $Workspace
git clone https://github.com/PyAV-Org/PyAV.git
cd PyAV
# v0.9.1 ベースは main からビルド（Linux 検証コミット f6f0a5e）。61e4aa8 には CUDA stream API がない
git checkout f6f0a5e

# cl.exe は INCLUDE を、link.exe は LIB を参照する（Developer PowerShell の既存値に追記）
$env:INCLUDE = "$env:INCLUDE;$Workspace\ffmpeg-n8.1-shared\include"
$env:LIB     = "$env:LIB;$Workspace\ffmpeg-n8.1-shared\lib"
uv build --wheel                     # -> dist\av-18.0.0-cp311-abi3-win_amd64.whl
```

**FFmpeg DLL の解決には delvewheel が必須です。** Python 3.8 以降の Windows は、拡張モジュール（`av\_core.pyd`）の依存 DLL を PATH から探しません（システムディレクトリ、.pyd 自身のフォルダ、`os.add_dll_directory()` 登録分のみ）。そのため BtbN の `bin\` を PATH に置くだけでは `ImportError: DLL load failed while importing _core` になります（実測）。delvewheel で FFmpeg DLL 群を wheel に同梱してからインストールします:

```powershell
uv pip install delvewheel
delvewheel repair dist\av-18.0.0-cp311-abi3-win_amd64.whl `
    --add-path $Workspace\ffmpeg-n8.1-shared\bin -w dist\repaired
uv pip install --reinstall (Get-Item dist\repaired\av-*.whl)

# 確認: BtbN の FFmpeg (n8.1.x) を認識し、current_ctx API があること
python -c "import av; from av.video.frame import CudaContext; print(av.ffmpeg_version_info); print(hasattr(CudaContext, 'current_ctx'))"
```

delvewheel は DLL をハッシュ付き名で `av.libs\` に同梱するため、torchcodec が実行時にロードする FFmpeg DLL とはプロセス内で別コピーになりますが、Windows は DLL ごとにシンボル空間が独立しており、Linux で問題になった同名ライブラリの衝突は起きません（スモークテスト一式で確認済み）。

> **注意: 自前ビルドの wheel はバージョン番号が PyPI の 18.0.0 と同じです。** `uv pip show av` では区別できず、取り違えると実行時に current_ctx 関連のエラーになります（トラブルシューティング参照）。5節の `uv pip install -e .[dev,nvidia]` は同版の av を置き換えない（実測確認済み）ため、本節を先に済ませておけばそのまま維持されます。導入済みかは `Test-Path .venv\Lib\site-packages\av.libs`（delvewheel 同梱の証拠）で判別できます。

---

## 5. jasna 本体のインストール

v0.8.1 で GPU スタックは extras に分割された（`nvidia` = NVIDIA スタック、`amd` = ROCm 用）。NVIDIA ビルドでは `nvidia` extra が `torch==2.12.0+cu130` / `torchvision==0.27.0+cu130` / `torch-tensorrt==2.12.0` / `nvidia-vfx` を入れる。これらは標準の PyPI には存在しないため、**PyTorch の wheel インデックスを `--extra-index-url` で指定し、さらに 2 つのフラグが必要**:

```powershell
cd $Workspace\jasna
uv pip install -e .[dev,nvidia] `
    --extra-index-url https://download.pytorch.org/whl/cu130 `
    --index-strategy unsafe-best-match `
    --prerelease=allow
```

> **⚠️ v0.9.1 ベースの Windows では、このコマンドはそのままだと解決に失敗します。** `[nvidia]` extra の `python_vali==4.8.7` ピンに PyPI の Windows wheel が存在しないためです（`No wheels with a matching platform tag` エラー）。回避手順: まず `[dev,torchcodec]`（`nvidia` 抜き）で venv を作り、5.3 節でフォーク wheel（4.8.7）をビルド・導入してから、上記の `[dev,nvidia,...]` を再実行します。導入済みの 4.8.7 がピンを満たすため、2 回目は解決に成功します（実機確認済み）。

各フラグの役割:

- `--extra-index-url https://download.pytorch.org/whl/cu130`: `torch+cu130` / `torchvision+cu130` / `torch-tensorrt+cu130` ホイールの取得先
- `--index-strategy unsafe-best-match`: `pyproject.toml` の `torch-tensorrt==2.12.0` を PyTorch インデックス上の `torch-tensorrt==2.12.0+cu130`（ローカルバージョン付き）で満たすために必要。デフォルトの first-index 戦略では拒否される
- `--prerelease=allow`: `torch-tensorrt` の推移依存 `nvidia-cuda-runtime-cu13==0.0.0a0` がプレリリース版のため必要

`[dev]` で `nuitka>=2.4`, `pytest`, `pytest-cov`, `scikit-build`, `cmake`, `ninja` が入る。`[nvidia]` で GPU スタック（torch cu130 / TensorRT / torch-tensorrt / nvidia-vfx。v0.8.1 で必須依存から分割）が入る — **指定を忘れると torch が入らず起動しない**。

**オプション: torchcodec バックエンド。** 実験的な torchcodec のデコード/エンコード経路（`--video-backend torchcodec`/`auto`）を使う場合は、`torchcodec` extra を追加し、同じフラグで `.[dev,nvidia,torchcodec]` を入れる:

```powershell
uv pip install -e .[dev,nvidia,torchcodec] `
    --extra-index-url https://download.pytorch.org/whl/cu130 `
    --index-strategy unsafe-best-match `
    --prerelease=allow
```

これで `torchcodec>=0.15.0` が入る。torchcodec は実行時に FFmpeg の DLL 群を必要とするため、BtbN shared ビルドの `bin\` を PATH に置く（1.1節）。通常のセットアップには不要で、既定の `native` バックエンド（PyAV）は torchcodec なしで動作する。詳細は [torchcodec_backend.md](torchcodec_backend.md)。

**補足: FP8 復元バックエンドに追加のインストール手順は不要。** 依存 `nvidia-cudnn-frontend` と（Windows では）`triton-windows` は `pyproject.toml` の通常依存で、上記コマンドで一緒に入る。cuDNN ランタイム（9.17 以上）は torch cu130 wheel に同梱済み。機能は実行時 opt-in（`--fp8-recon`、FP8 対応 GPU sm89 以上が必要）で、使えない環境では TensorRT エンジンにフォールバックする。v0.8.0+modi でも Windows 11 + RTX 5080 で実機検証済み（2026-07-18。`--log-level info` で `CudnnFP8Upsample: enabled` を確認。既定の `--log-level error` では有効化ログが出ない点に注意）。詳細は [fp8_recon.md](fp8_recon.md)。

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

### 5.2 ONNX パッケージ（YOLO 検出モデル用）

**YOLO 系検出モデル（lada-yolo-\* と zelefans-vr-yolo-v2）**を使う場合、ultralytics は TensorRT エンジンをビルドする前にモデルを ONNX へエクスポートする。これには自動では入らない 3 パッケージが必要（RF-DETR モデルは不要。事前ビルド済み `.onnx` を TensorRT が直接パースするため）:

```powershell
uv pip install onnx onnxslim onnxruntime
```

> **これらが未導入だと**、`--detection-model lada-yolo-v4` 実行時にエンジンコンパイルが `ERROR ONNX: export failure ... No module named 'onnx'` → `RuntimeError: Engine compilation subprocess failed` で中断する。venv に 3 パッケージを入れれば解消する。

### 5.3 オプション: VALI NVDEC デコードバックエンドのビルド（`python_vali` フォーク）

v0.9.1 ベースの native デコードは、まず VALI NVDEC デコーダを試し、使えなければ PyAV へエスカレーションします。フォーク専用 API `DecodeSingleSurfaceAsyncDetailed` を持つ wheel が無いと、リーダーは警告を出して毎回 PyAV にフォールバックします（本節なしでもすべて動作します）。Windows では `[nvidia]` extra の解決にも 4.8.7 の wheel が要るため（5 節の注記）、フォーク wheel のビルドが実質前提になります。

1 節に対する追加の前提:

- **VS Build Tools 2022**（4 節 (B) と同じ。ビルドは vcvars64 を通した cmd / Developer PowerShell セッションで行う）
- **CUDA Toolkit 13.x**（`nvcc`。vali の CMake は `CUDA_PATH` から toolkit を解決する）
- **FFmpeg 8 の shared ビルド**: vali の CMake は `FFMPEG_ROOT` を **`C:\Program Files\ffmpeg8` にハードコード**しているため、`include\`/`lib\`/`bin\` を持つ shared ビルド（gyan.dev の full-shared 等）をそこに置くか、junction を張る（`New-Item -ItemType Junction -Path "C:\Program Files\ffmpeg8" -Target <展開先>`）。BtbN ビルド（1.1 節）でも要件は同じ。
- cmake、ninja、scikit-build、`pkg_resources` 入りの setuptools（<80）: いずれも 5 節の `[dev]` で venv に導入済み。

```powershell
cd $Workspace
git clone https://codeberg.org/Kruk2/vali.git       # 既存 checkout があれば省略
cd vali
git checkout 3ad0d54      # = 4.8.7 ピン、DecodeSingleSurfaceAsyncDetailed あり
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
REM -> dist\python_vali-4.8.7-cp313-cp313-win_amd64.whl
```

Linux と違い、Windows の wheel は FFmpeg の DLL 群を `python_vali\` 内に同梱するため（CMake の `PYPI_BUILD=0` 既定）、delvewheel は不要です。導入と確認:

```powershell
uv pip install --force-reinstall --no-deps (Get-Item $Workspace\vali\dist\python_vali-4.8.7-*.whl)
python -c "import python_vali as v; print(hasattr(v.PyDecoder, 'DecodeSingleSurfaceAsyncDetailed'))"   # -> True
# jasna を --log-level info で実行すると "Using VALI NVDEC decoder for <file>" が出ます
```

> フォークコミット `3ad0d54`（= 4.8.7 ピン、2026-07-30、RTX 5080 + CUDA 13.2 + gyan.dev 8.1 shared）で Windows 実機検証済み: パイプラインの 2 つのデコードパスが両方 VALI を通り、同梱テストクリップの出力は PyAV デコード時と**ビットストリーム md5 完全一致**、av wheel および torchcodec バックエンドとの共存も確認（av は delvewheel 同梱、vali は自前同梱、torchcodec は PATH 解決で、それぞれ別コピーの FFmpeg DLL を持つが、Windows は DLL ごとにシンボル空間が独立しており衝突しない）。

---

## 6. モデルウェイトとアセットの配置

`$Workspace\jasna\model_weights\` に以下 4 ファイルを置く:

- `lada_mosaic_restoration_model_generic_v1.2.pth`
- `rfdetr-v5.onnx`
- `lada_mosaic_detection_model_v4_fast.pt`
- `lada_vr_mosaic_detection_model_v2_accurate.pt`（v0.8.0 で追加。VR180 向け YOLO 検出モデル `zelefans-vr-yolo-v2` の実体）

ソースから実行すると Jasna は `model_weights\` を自動解決する（付録 A.1）ため、ここに置くだけでよい。

検出モデルは `model_weights\` にあるファイルから自動発見されるため、4 つ目が無くても他機能は動作する。その場合は `zelefans-vr-yolo-v2` がモデル一覧に現れず、VR180 の検出に使えないだけである。

テストクリップ 2 本は通常リポジトリに同梱され `$Workspace\jasna\assets\`（`test_clip1_1080p.mp4`, `test_clip1_2160p.mp4`）にある。無い場合は upstream リリースから抽出する。

### 6.1 オプション: RIFE フレーム補間モデル (`--frame-gen` 用)

フレームレート倍化 (`--frame-gen {2x,4x}`) を使う場合のみ必要。RIFE 重みは**同梱されない**（非商用条項のため）ので、`scripts/make_rife_torchscript.py` で TorchScript チェックポイントを自分で作成する。

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

FlashVSR（`--secondary-restoration flashvsr` / `flashvsr-inline`）は別リポジトリのチェックアウトと専用 venv を必要とし、inline 用にはチェックアウトへの同梱パッチ適用も要る。Windows 対応済み（inline は v0.8.0+modi でも Windows の 16 GB カード + `--flashvsr-tiles 2` で再検証済み。v0.8.0 は inline 時に `--max-clip-size` を自動で 32 に抑えるため、v0.7.2 期より VRAM に余裕がある）。セットアップ手順は [flashvsr.md](flashvsr.md) を参照。

---

## 7. ソースから実行 / 動作確認

1〜6節が完了したら、venv 内でソースチェックアウトから直接 Jasna を実行する:

```powershell
cd $Workspace\jasna
python -m jasna --version    # -> 0.9.1+modi
python -m jasna --help
jasna --input assets\test_clip1_1080p.mp4 --output $env:TEMP\out.mp4   # 短いクリップを処理
python -m jasna              # GUI を起動（引数なし）
```

> **GUI の起動は `python -m jasna`（引数なし）です。** console script の `jasna` は `jasna.main:main` に直結しており、引数なしでも GUI ディスパッチ（`jasna\__main__.py`）を通りません。frozen ビルド（`jasna.exe`）の引数なし起動だけが GUI に入ります。

確認事項:

- `ffmpeg` / `ffprobe`（v8）が `PATH` 上にあること（1.1節）。`mkvmerge` は不要になりました。
- NVIDIA ドライバが **610 以上**であること（起動チェックが拒否します）。
- インストールパスが ASCII のみであること（起動時に強制。ユーザー名に非 ASCII を含む環境では展開先に注意）。

> **最初の処理実行は遅い。** 初回使用時に GPU 向け TensorRT エンジンがコンパイルされる（15〜60 分）。`model_weights\` 内のウェイト隣（`<モデル名>_sub_engines\` 等）にキャッシュされ、以降は再利用される。v0.7.2 から移行した場合は、既定 max-clip-size の変更（90）に伴い一部エンジンが再コンパイルされる（loop_body 系のキャッシュは名前が変わらないため再利用される）。

---

## 8. パッケージング / frozen ビルド

このフォークは実験的な Nuitka ビルドスクリプト（`scripts\build_nuitka.py`）を同梱しており、v0.7.2 期には単一の `jasna.exe` から成るスタンドアロン配布物を生成できました（[frozen_build.md](frozen_build.md)）。

⚠️ **このスクリプトは v0.8.0 のメディア層移行に未追従です。** 旧 `python_vali` / `PyNvVideoCodec` の DLL 同梱（CUDA NPP / nvJPEG 等）を前提としたままで、現状では動作しない見込みです。v0.8.0 対応は Windows 実機検証後に行います。

upstream 自身のパッケージングツール（同じく Nuitka ベース）は**プライベートな submodule（`jasna/protection`）**にあり、このフォークには含まれません。この submodule に依存する機能（`unet-4x`、SD1.5 inpaint、ライセンス認証）は、このフォークの凍結ビルドでは動きません。

---

## トラブルシューティング

### PyAV / FFmpeg

- **エンコード開始直後に `ValueError: hevc_nvenc did not accept encoder option(s): ['lookahead_level']`**
  PyAV が古い nv-codec-headers の FFmpeg にリンクされている。4節 (B) のとおり `INCLUDE` / `LIB` を BtbN ビルドに向けて wheel を作り直す。

- **`import av` が `ImportError: DLL load failed while importing _core` で失敗する**
  自前ビルドの wheel を delvewheel なしでインストールしている。Python 3.8+ は拡張モジュールの依存 DLL を PATH から探さないため、BtbN の `bin\` を PATH に置くだけでは解決しない。4節 (B) の delvewheel repair を適用した wheel を入れ直す。

- **PyAV のビルドが `fatal error C1083: libavcodec/avcodec.h` で失敗する**
  FFmpeg の場所がコンパイラに伝わっていない。`PKG_CONFIG_PATH` は Windows では無視される（setup.py が pkg-config を呼ばない）。4節 (B) のとおり `INCLUDE` / `LIB` に BtbN 展開物の `include` / `lib` を追記する。

- **エンコーダ/デコーダ初期化時に `current_ctx` 関連の `TypeError` 等で失敗する**
  av が PyPI の 18.0.0 のまま（4節の wheel 未導入、または依存再インストールで置き換わった）。4節の wheel を `uv pip install --reinstall` で入れ直す。

- **Jasna が起動拒否: ffprobe のバージョン違い**
  起動チェックは `ffprobe` の**メジャーバージョン 8** を要求する。v8 ビルドの `bin\` を `PATH` に追加する（1.1節）。`mkvmerge` のチェックは v0.8.0 で廃止された。

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
  Windows の最低要件は **610** です（v0.7.2+modi 期の 591.67+ から引き上げ）。NVIDIA 公式からドライバを更新する。

- **初回起動が極端に遅い**
  異常ではない。TensorRT エンジン初回コンパイルに 15〜60 分かかる。`model_weights\` 内のウェイト隣にキャッシュされ 2 回目以降は高速化される。

---

## 付録 A: 本ブランチのビルド/ランタイム改修の内容

本ブランチのソースへ適用済みの変更です（参照用の記述。独立した `.patch` ファイルは同梱しない）。

### A.1 `model_weights/` ディレクトリの自動解決

**問題**: 既定の `--detection-model-path` / `--restoration-model-path` のデフォルトが `Path("model_weights")` 相対パス。任意フォルダから Jasna を実行すると、`<CWD>/model_weights/...` を探して `FileNotFoundError`。

**修正**: 新規 `jasna/model_weights_resolver.py` が `model_weights/` フォルダを以下の優先順で自動的に探す。

1. **環境変数 `JASNA_MODEL_WEIGHTS_DIR` で指定したフォルダ**：ユーザーが明示した場所が最優先
2. **実行ファイルと同じフォルダの中の `model_weights\`**：パッケージ済みインストールの標準配置
3. **コマンドを実行したフォルダの中の `model_weights\`**：いま自分がいる場所
4. **jasna ソースの親フォルダの中の `model_weights\`**：`uv pip install -e .` でソースから動かしている開発者向け

最初に見つかったものを使う。どこを採用したかは `--log-level info` で起動時にログ表示される。`main.py`、`mosaic/detection_registry.py`、`engine_paths.py`、GUI（`gui/processor.py`, `gui/engine_preflight.py`）がハードコードの `Path("model_weights")` ではなく resolver を経由する。

### A.2 DLL ロード補助（torchcodec バックエンドのみ）

v0.8.0 で native 経路（PyAV）の DLL 補助は不要になりました。現在は torchcodec バックエンドのモジュール（`jasna/media/torchcodec_decoder.py` / `torchcodec_encoder.py`）が **Windows のみ**、torchcodec パッケージのディレクトリと `CUDA_PATH\bin`（設定時）を DLL 検索パスに登録します。

### A.3 BasicVSR++ ベンチの TRT API 修正

`jasna/benchmark/basicvsrpp_restoration.py` を新しい `_preprocess_engine` API（feat_extract + flow を 1 つの TRT エンジンに統合）に更新し、`--benchmark basicvsrpp` を動作させる。クロスプラットフォーム。

### 完全な diff

本ブランチの全変更は upstream との比較（`git diff upstream/main..modi`）で確認できる。
