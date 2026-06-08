# Building Jasna for Windows

Windows で Jasna のビルド依存をセットアップし、**ソースから実行**する手順。

> **本ガイドは `v0.7.1+modi` ブランチの手順です。** GPU スタック — **torch 2.12.0+cu130 / torchvision 0.27.0+cu130 / torch-tensorrt 2.12.0+cu130 / tensorrt 10.16.1.11** で、これらの依存ピンは本ブランチの `pyproject.toml` に適用済みです。TensorRT が **10.16** 系に留まるのは、`torch-tensorrt==2.12.0` が `tensorrt>=10.16.1,<10.17.0` を要求するためです（torch-tensorrt は TensorRT 11 に未対応）。
>
> **このブランチの新機能:** AV1 出力 / 8bit(NV12) 出力 / BT.601・BT.2020 色空間保持（[CODECS_AND_COLORSPACE_ja.md](CODECS_AND_COLORSPACE_ja.md) 参照）、および RIFE による 2x/4x フレーム生成（[FRAME_GENERATION_ja.md](FRAME_GENERATION_ja.md) 参照）。

> **パッケージングについて:** この公開フォークはネイティブ GPU 依存をビルドし、Jasna を**ソースから実行**します。frozen/パッケージ化されたバイナリを生成する **Nuitka のパッケージングツールは同梱していません** — そのツールは upstream と同様に**プライベート**です。ソース実行ではなくパッケージ済みバイナリが欲しい場合は upstream Kruk2/jasna の公式リリースを使ってください。詳細は [パッケージング / frozen ビルド](#10-パッケージング--frozen-ビルド)。

---

## 1. 前提ソフトウェア

> ⚠ **ffmpeg のインストール場所について先に注意**: `vali/src/CMakeLists.txt` が `FFMPEG_ROOT` を `C:/Program Files/ffmpeg8` にハードコードしており、`-DFFMPEG_ROOT=...` (CMAKE_ARGS) を渡しても上書きされません。そのため **ffmpeg dev ビルドは `C:\Program Files\ffmpeg8\` に展開するのが最も簡単** です。既に別の場所に置いている、または `C:\Program Files\` に書き込めない場合は、junction で見せかけるか vali のソースを 1 行編集する必要があります（詳細はセクション 5.2）。後から発覚するとビルドのやり直しになるため、ここで配置場所を決めておくこと。

| 種別 | 要件 | winget ID / 入手元 | 備考 |
|---|---|---|---|
| OS | Windows 10/11 x64 | — | PowerShell 7+ 推奨 |
| Git | Git for Windows | `Git.Git` | サブモジュール取得用 |
| uv | 最新版 | `astral-sh.uv` | Python の自動管理を含む (下記参照) |
| Python | 3.13+ | — (uv が自動管理) | uv が `uv venv --python 3.13` 実行時に未インストールなら自動取得するため、別途インストール不要 |
| CUDA Toolkit | **13.2** | NVIDIA 公式 ([developer.nvidia.com](https://developer.nvidia.com/cuda-downloads)) | 既定 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`。`CUDA_PATH` が 13.2 を指していること (セクション 3.1)。サイズが大きいため winget は非推奨 |
| NVIDIA Driver | 591.67+ (59x 系) | NVIDIA 公式 or GeForce Experience | GPU は compute capability 7.5+ |
| VS Build Tools | **2022** (Desktop development with C++) | `Microsoft.VisualStudio.2022.BuildTools` | `vali` / `PyNvVideoCodec` の C++ ビルドに必要 |
| ffmpeg / ffprobe | **v8 shared dev ビルド** | `Gyan.FFmpeg.Shared` | `include/`, `lib/`, `bin/` を含む dev ビルドを **`C:\Program Files\ffmpeg8\`** に展開（上記注意書きの通り、別パスを使う場合は workaround が必要）。`vali` / `PyNvVideoCodec` がリンク時に参照 (詳細はセクション 5)。v8 CLI は実行時にも必要 |
| MKVToolNix | `mkvmerge`（実行時） | [mkvtoolnix.download](https://mkvtoolnix.download/) | Jasna 実行時に `PATH` 上に必要。依存のビルド自体には不要 |

C++ コンパイルを伴う手順は **Developer PowerShell for VS 2022**、もしくは `vcvars64.bat` を読み込んだセッションで実行すること。

### 1.1 winget で一括インストール

CUDA Toolkit と NVIDIA Driver を除く前提ソフトウェアは winget でまとめて入れられる:

```powershell
winget install --id Gyan.FFmpeg.Shared              -e --source winget
winget install --id Git.Git                         -e --source winget
winget install --id astral-sh.uv                    -e --source winget
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --silent `
  --override "--wait --quiet --add ProductLang En-us --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

注意点:

- **ffmpeg**: `Gyan.FFmpeg.Shared` のインストール先は winget 既定の場所 (`C:\Users\<user>\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_*\ffmpeg-*-full_build-shared\` 等) になり、`C:\Program Files\ffmpeg8\` ではない。インストール後に `Get-ChildItem $env:LOCALAPPDATA\Microsoft\WinGet\Packages\ -Filter "Gyan.FFmpeg*" -Recurse -Directory` 等で実体パスを確認し、`C:\Program Files\ffmpeg8\` への junction を作る (セクション 5.2 (A)) のが楽。
- **Python は不要**: uv が `uv venv --python 3.13` 実行時に未インストールなら自動取得する。明示的に先入れしたい場合は `uv python install 3.13`。
- **CUDA Toolkit と NVIDIA Driver は手動インストール**。CUDA はバージョン固定 (13.2) が必要で winget の `Nvidia.CUDA` ではバージョン管理が煩雑なため、NVIDIA 公式からダウンロードする方が確実。
- **MKVToolNix は依存のビルドには不要** だが、Jasna を実際に実行する際には `mkvmerge` が `PATH` 上にある必要がある（MKVToolNix を入れて `PATH` に追加する）。

---

## 2. リポジトリのクローン

3 リポジトリを **同一の親ディレクトリ** 配下に並べる。以降の手順では作業ルートを `$Workspace` という PowerShell 変数で参照する。**自身の環境に合わせてパスを置き換える** こと（例では `C:\jasna-dev` を使う）。

```powershell
$Workspace = "C:\jasna-dev"          # ← 任意のディレクトリを指定
mkdir $Workspace -Force | Out-Null
cd $Workspace

git clone https://codeberg.org/Kruk2/vali.git
git clone https://codeberg.org/Kruk2/PyNvVideoCodec.git
git clone -b modi https://github.com/sh202603/jasna.git   # modi ブランチ（本フォークの既定・パッチ適用済み）

# vali はサブモジュール (extern/dlpack) を持つので初期化
cd vali
git submodule update --init --recursive
cd ..
```

レイアウト:

```
<Workspace>\
  vali\
  PyNvVideoCodec\
  jasna\        <- 以降の作業はここを起点
```

> 以降のセクションで `$Workspace` を使うコマンドを実行する場合、**新しい PowerShell セッションでは再度 `$Workspace = "..."` を設定する** こと（セクション 3.5 の helper スクリプトを使うと自動化できる）。

---

## 3. ビルド環境のセットアップ (環境変数 + VS Build Tools)

`vali` / `PyNvVideoCodec` の C++ ビルドのために、以下を **同じ PowerShell セッションで** セットアップする。以降のセクション 4 以降はこのセッションで実行する。

### 3.0 ビルド環境変数の早見表

セットアップ全体で使う環境変数の一覧。詳細は各セクション参照。**セクション 3.5 の helper スクリプトを使えば一括設定可能**。

| 環境変数 | 値の例 | 消費者 | 設定セクション |
|---|---|---|---|
| `CUDA_PATH` / `CUDA_HOME` | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2` | nvcc, vali CMake, jasna 実行時 | 3.1 |
| `PATH` (追加) | `<CUDA_PATH>\bin;<CUDA_PATH>\libnvvp;<ffmpeg>\bin` | shell | 3.1 / 3.2 |
| `DISTUTILS_USE_SDK` | `1` | setuptools | 3.4 |
| `CMAKE_GENERATOR` | `Ninja` | CMake | 3.4 |
| `CL` | `/utf-8` | MSVC `cl.exe` (PyNvVideoCodec ビルドで必須) | 3.4 |
| `FFMPEG_DIR` | `C:/Program Files/ffmpeg8` **(★フォワードスラッシュ)** | PyNvVideoCodec CMake | 5 / 7 |
| `CMAKE_ARGS` | `-DFFMPEG_ROOT=H:/ffmpeg` | (任意: 既定外パスで vali をビルドする場合) | 5.2 |

### 3.1 CUDA_PATH の固定

本プロジェクトでは **CUDA 13.2** を使う。`CUDA_PATH` を設定して nvcc / vali が正しい toolkit を拾えるようにし、また Jasna 実行時にネイティブライブラリが CUDA `bin/` ディレクトリを見つけられるようにする（セクション 6 / 付録 A.2）。

> 補足: `pyproject.toml` の `torch==2.12.0+cu130` は CUDA 13.0 ABI ビルドだが、CUDA 13.x はマイナーバージョン互換があり 13.2 ランタイム上でも動作する。

```powershell
$env:CUDA_PATH      = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:CUDA_HOME      = $env:CUDA_PATH
$env:PATH           = "$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:PATH"
```

確認:

```powershell
nvcc --version          # release 13.2 が表示されること
Get-Command nvcc | Select-Object Source
```

恒久化したい場合は `setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"` をユーザー環境変数として登録 (新セッションから有効)。

### 3.2 ffmpeg の PATH への追加 (Jasna 実行に必須)

ソースから Jasna を実行する際、v8 の `ffmpeg`/`ffprobe` CLI が `PATH` 上にある必要がある（同梱コピーは無い）。ffmpeg の `bin` ディレクトリを追加する:

```powershell
$env:PATH = "C:\Program Files\ffmpeg8\bin;$env:PATH"
ffmpeg -version          # ffmpeg version 8.x が表示されること
```

> ffmpeg を既定外の場所に置く場合の **ビルド時** の設定 (vali / PyNvVideoCodec) はセクション 5 にまとめて記述。

### 3.3 VS Build Tools 環境のロード

C++ コンパイラ (`cl.exe`) と Windows SDK を現在のセッションに取り込む。`vali` / `PyNvVideoCodec` の `pip install` で必須。

**PowerShell から (推奨):**

```powershell
& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1' `
    -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
```

> `Launch-VsDevShell.ps1` は既定で作業ディレクトリを VS のソースパスに移動する。`-SkipAutomaticLocation` で現在のディレクトリを維持する。

**cmd.exe から起動する場合:**

```cmd
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

> 別エディション (Community / Professional / Enterprise) を使っている場合は `BuildTools` の部分を読み替える。

確認:

```powershell
cl.exe                  # Microsoft (R) C/C++ Optimizing Compiler のバナーが出ること
where.exe cl.exe        # MSVC ツールセット配下のパスが返ること
```

### 3.4 ビルド補助の環境変数

scikit-build / setuptools が MSVC を確実に使うよう、以下を設定しておく。

```powershell
$env:DISTUTILS_USE_SDK = "1"        # setuptools に既存の MSVC 環境を尊重させる
$env:CMAKE_GENERATOR   = "Ninja"    # CMake のジェネレータを Ninja に固定 (任意)
$env:CL                = "/utf-8"   # 日本語 Windows (CP932) で C4819 → C2220 を避ける (PyNvVideoCodec 等で必須)
```

> **`$env:CL = "/utf-8"` の必要性**: 日本語版 Windows のシステムコードページは CP932 (Shift-JIS)。CUDA / ffmpeg のヘッダには UTF-8 で書かれた著者名 (アクセント記号付き等) が含まれ、CP932 では表現できないため MSVC が `C4819` 警告を出す。`PyNvVideoCodec` は `/WX` (警告をエラー扱い) でビルドするため、これが `error C2220` に昇格してビルド失敗する。`/utf-8` を渡すと MSVC がソースを UTF-8 として解釈し、警告自体が出なくなる。`CL` 環境変数はすべての `cl.exe` 呼び出し（`nvcc` の `-Xcompiler` 経由も含む）に効く。

### 3.5 セッションロード用ヘルパースクリプト (任意)

毎回コマンドを打つのが面倒な場合、以下を **`$Workspace\setup-build-env.ps1`** として保存し、ビルドの度に `. $Workspace\setup-build-env.ps1` で読み込む（または `cd $Workspace; . .\setup-build-env.ps1`）。

`$Workspace` をスクリプト自身の置き場所から自動算出するので、ワークスペースを移動・コピーしても書き換え不要。

```powershell
# setup-build-env.ps1
$Workspace                 = $PSScriptRoot          # スクリプトと同階層を Workspace とみなす

$env:CUDA_PATH             = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:CUDA_HOME             = $env:CUDA_PATH
$env:DISTUTILS_USE_SDK     = "1"
$env:CMAKE_GENERATOR       = "Ninja"
$env:CL                    = "/utf-8"                          # 日本語 Windows での MSVC 文字コード対策
$env:FFMPEG_DIR            = "C:/Program Files/ffmpeg8"        # PyNvVideoCodec CMake 用。★フォワードスラッシュ必須 (理由はセクション 7)
$env:PATH                  = "$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;C:\Program Files\ffmpeg8\bin;$env:PATH"

& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1' `
    -Arch amd64 -HostArch amd64 -SkipAutomaticLocation

# $Workspace を呼び出し側セッションにも公開
Set-Variable -Name Workspace -Value $Workspace -Scope Global
```

---

## 4. Python 仮想環境の作成

`jasna` リポジトリ配下に `.venv` を作る。以降の `pip` / `uv pip` はすべてこの venv をアクティブにした状態で実行する。

```powershell
cd $Workspace\jasna
uv venv --python 3.13
.\.venv\Scripts\Activate.ps1
```

ビルド系ツールを先に入れておく (`--no-build-isolation` で venv 側を参照させるため):

```powershell
uv pip install cmake ninja scikit-build "setuptools<80" wheel numpy
```

> `setuptools` 80 以降は `pkg_resources` が標準同梱から外され、PyPI 上にも独立した `pkg_resources` パッケージは存在しない。`vali` / `PyNvVideoCodec` の `setup.py` は `from pkg_resources import ...` を使うため、`setuptools<80` (現状 `79.0.1` 等) で固定する必要がある。

---

## 5. ffmpeg v8 (shared dev ビルド) の配置

`vali` のビルド（次セクション）が ffmpeg のヘッダ/インポートライブラリにリンクするため、**実行ファイルだけでなく `include/` と `lib/` を含む "shared dev ビルド" が必須**。同じ `bin/` が、実行時に `PATH` 上で必要な v8 の `ffmpeg`/`ffprobe` CLI も提供する（セクション 3.2）。

### 5.1 入手と展開

- 推奨入手元: [gyan.dev (Windows builds)](https://www.gyan.dev/ffmpeg/builds/) の "shared" 版（例: `ffmpeg-8.x-full_build-shared.7z`）あるいは [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases) の `*-win64-lgpl-shared-8.x.zip`
- 展開先: **`C:\Program Files\ffmpeg8\`**（後述のとおり `vali` の CMake がこのパスをハードコードしているため、他の場所に置く場合は workaround が必要）
- 展開後の構成（必須）:
  ```
  C:\Program Files\ffmpeg8\
    bin\        # ffmpeg.exe, ffprobe.exe, avcodec-*.dll, avformat-*.dll, ...
    include\    # libavcodec\, libavformat\, libavutil\, libswresample\, libswscale\
    lib\        # avcodec.lib, avformat.lib, avutil.lib, swresample.lib, swscale.lib
  ```

### 5.2 既定外の場所に置きたい場合

`vali/src/CMakeLists.txt` は `set(FFMPEG_ROOT "C:/Program Files/ffmpeg8")` を **ガード無しで** 実行するため、`-DFFMPEG_ROOT=...` (CMAKE_ARGS) を渡しても上書きされない。回避策は以下のいずれか:

**(A) Junction で見せかける（推奨、管理者権限が必要）**

```powershell
# 管理者 PowerShell から
New-Item -ItemType Junction -Path "C:\Program Files\ffmpeg8" -Target "H:\ffmpeg"
```

**(B) vali のソースを 1 行だけ条件付きに編集する**

```powershell
cd $Workspace\vali
(Get-Content src\CMakeLists.txt) -replace `
  '^set\(FFMPEG_ROOT "C:/Program Files/ffmpeg8"\)$', `
  "if(NOT DEFINED FFMPEG_ROOT)`r`n  set(FFMPEG_ROOT `"C:/Program Files/ffmpeg8`")`r`nendif()" |
  Set-Content src\CMakeLists.txt

$env:CMAKE_ARGS = "-DFFMPEG_ROOT=H:/ffmpeg"   # フォワードスラッシュ
```

### 5.3 ffmpeg ディレクトリを参照する 2 つのビルド時消費者

ffmpeg はビルド中に `vali` と `PyNvVideoCodec` がそれぞれ別の変数で参照する。既定パス (`C:\Program Files\ffmpeg8`) を使うなら表中の「既定動作」列で済むが、別パスに置く場合は **各消費者ごとに個別設定が必要** な点に注意。

| 消費者 | 参照する変数 | 既定動作 | 既定外パスを使う場合 |
|---|---|---|---|
| **`vali` CMake** | `FFMPEG_ROOT` (CMake) | `vali/src/CMakeLists.txt` がガード無しで `C:/Program Files/ffmpeg8` をハードコード | セクション 5.2 (junction か vali ソース 1 行編集) |
| **`PyNvVideoCodec` CMake** | `FFMPEG_DIR` (環境変数) | 未設定なら同梱 `external/ffmpeg/` にフォールバック (lib 無し → ビルド失敗) | `$env:FFMPEG_DIR = "C:/Program Files/ffmpeg8"` を設定 (★フォワードスラッシュ。詳細はセクション 7) |

> 実行時には Jasna が v8 の `ffmpeg`/`ffprobe` CLI を `PATH` 上に要求する — `bin/` ディレクトリを追加する（セクション 3.2）。

---

## 6. vali のビルドとインストール

> **前提**: セクション 3 (CUDA_PATH / VS Build Tools / 環境変数) + セクション 4 (venv 作成 + `setuptools<80`) + セクション 5 (ffmpeg dev ビルドが `C:\Program Files\ffmpeg8\` または junction で配置) が完了していること。

```powershell
cd $Workspace\vali
uv pip install . --no-build-isolation
```

確認:

```powershell
python -c "import os; os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'],'bin')); import python_vali; print(python_vali.__file__)"
```

---

## 7. PyNvVideoCodec のビルドとインストール

> **前提**: セクション 6 と同様 (CUDA / VS Build Tools / venv / ffmpeg dev ビルド) に加え、`$env:CL = "/utf-8"` (3.4)。

PyNvVideoCodec の CMake は **環境変数 `FFMPEG_DIR`** を読み、未設定時は同梱の `external/ffmpeg/` (ヘッダ/ソースのみ、`lib/` 無し) にフォールバックして `avcodec_library-NOTFOUND` 等で失敗する。ビルド前に必ず ffmpeg dev ビルドのトップディレクトリを **フォワードスラッシュで** 指定する:

```powershell
$env:FFMPEG_DIR = "C:/Program Files/ffmpeg8"   # ★フォワードスラッシュ必須
$env:CL         = "/utf-8"                     # 日本語 Windows 必須 (セクション 3.4 参照)
cd $Workspace\PyNvVideoCodec
uv pip install . --no-build-isolation
```

> **フォワードスラッシュが必須**: PyNvVideoCodec の install 段で CMake が `FFMPEG_DIR` の値を生成スクリプトに埋め込む際、バックスラッシュを含むパス (`C:\Program Files\ffmpeg8`) を quoted string として処理すると `\P` 等を不正エスケープとみなして `Syntax error in cmake code` で失敗する。

> `vali` の `FFMPEG_ROOT` (CMake 変数) とは別の変数名・別の渡し方なので注意。PyNvVideoCodec は環境変数のみを参照する。

> 日本語 Windows では `$env:CL = "/utf-8"` が無いと `error C2220: 警告がエラーとして扱われます` / `warning C4819` で失敗する。詳細はセクション 3.4 参照。

> 設定変更後にビルドし直す場合は古いキャッシュを破棄: `Remove-Item -Recurse -Force _skbuild`

確認:

```powershell
python -c "import PyNvVideoCodec; print(PyNvVideoCodec.__file__)"
```

---

## 8. jasna 本体のインストール

> **前提**: セクション 6 / 7 の `python_vali` / `PyNvVideoCodec` が venv にインストール済みであること。

`pyproject.toml` の依存に `torch==2.12.0+cu130` / `torchvision==0.27.0+cu130` / `torch-tensorrt==2.12.0` が含まれる。これらは標準の PyPI には存在しないため、**PyTorch の wheel インデックスを `--extra-index-url` で指定し、さらに 2 つのフラグが必要**:

```powershell
cd $Workspace\jasna
uv pip install -e .[dev] `
    --extra-index-url https://download.pytorch.org/whl/cu130 `
    --index-strategy unsafe-best-match `
    --prerelease=allow
```

各フラグの役割:

- `--extra-index-url https://download.pytorch.org/whl/cu130`: `torch+cu130` / `torchvision+cu130` / `torch-tensorrt+cu130` ホイールの取得先
- `--index-strategy unsafe-best-match`: `pyproject.toml` の `torch-tensorrt==2.12.0` を PyTorch インデックス上の `torch-tensorrt==2.12.0+cu130` (ローカルバージョン付き) で満たすために必要。デフォルトの first-index 戦略では拒否される
- `--prerelease=allow`: `torch-tensorrt` の推移依存 `nvidia-cuda-runtime-cu13==0.0.0a0` がプレリリース版のため必要

`[dev]` で `nuitka>=2.4`, `pytest`, `pytest-cov`, `scikit-build`, `cmake`, `ninja` が入る。

スモークテスト:

```powershell
jasna --help
```

### 8.1 mmengine パッチの適用 (torch 2.6+ 対応)

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

> このパッチは `uv pip install -e .[dev]` で `mmengine` を入れ直すたびに上書きされて消える。再インストール後は再度当てる。

### 8.2 ONNX パッケージ（YOLO 検出モデル用）

**YOLO（lada-yolo-\*）検出モデル**を使う場合、ultralytics は TensorRT エンジンをビルドする前にモデルを ONNX へエクスポートします。これには自動では入らない 3 パッケージが必要です。（RF-DETR モデルは不要 — 事前ビルド済み `.onnx` を TensorRT が直接パースするため。）

```powershell
cd $Workspace\jasna
uv pip install onnx onnxslim onnxruntime
```

ソースから実行する場合、ultralytics が初回エクスポート時に未導入ならこれらを自動ダウンロードしますが、事前に入れておくと初回 YOLO 実行時の待ちを回避できます。

> **これらが未導入だと**、`jasna --detection-model lada-yolo-v4 ...` 実行時にエンジンコンパイルが `ERROR ONNX: export failure ... No module named 'onnx'` → `RuntimeError: Engine compilation subprocess failed` で中断します。venv に 3 パッケージを入れれば解消します。

---

## 9. モデルウェイトとアセットの配置

`$Workspace\jasna\model_weights\` に以下 3 ファイルを置く:

- `lada_mosaic_restoration_model_generic_v1.2.pth`
- `rfdetr-v5.onnx`
- `lada_mosaic_detection_model_v4_fast.pt`

ソースから実行すると Jasna は `model_weights\` を自動解決する（付録 A.1）ため、ここに置くだけでよい。

テストクリップ 2 本は通常リポジトリに同梱され `$Workspace\jasna\assets\`（`test_clip1_1080p.mp4`, `test_clip1_2160p.mp4`）にある。無い場合は upstream リリースから抽出する。

### 9.1 オプション: RIFE フレーム補間モデル (`--frame-gen` 用)

フレームレート倍化 (`--frame-gen {2x,4x}`) を使う場合のみ必要。RIFE 重みは**同梱されない**（非商用条項のため）ので、`make_rife_torchscript.py` で TorchScript チェックポイントを自分で作成する。

1. Practical-RIFE をクローンし、4.x のモデルパッケージ（**v4.25 で動作確認**）をダウンロードして `<repo>\train_log\` に `RIFE_HDv3.py`, `IFNet_HDv3.py`, `flownet.pkl` が揃う状態にする:
   ```powershell
   git clone https://github.com/hzwer/Practical-RIFE
   # README に従いモデルパッケージを Practical-RIFE\train_log\ にダウンロード・展開
   ```
2. プロジェクトの venv で TorchScript に変換する（`$Workspace\jasna` で実行）:
   ```powershell
   .\.venv\Scripts\python.exe make_rife_torchscript.py `
       --rife-repo C:\path\to\Practical-RIFE `
       --output model_weights\rife.pth --validate
   ```

**ソースから実行**する場合は `model_weights\rife.pth` を自動で参照する（他のウェイトと同じリゾルバ経由。または `JASNA_MODEL_WEIGHTS_DIR` をそれを含むフォルダに向ける）。詳細手順: [docs/FRAME_GENERATION_ja.md](FRAME_GENERATION_ja.md)。

---

## 10. パッケージング / frozen ビルド

現時点で、このフォークから **パッケージ化された frozen バイナリを生成する公開手段はありません。** upstream はビルドを PyInstaller から **Nuitka** へ切り替えましたが、Nuitka のビルドスクリプトや手順は公開しておらず、実際のパッケージングツールは **プライベートな submodule（`jasna/protection`）** にあり、この公開フォークには含まれません。フォーク側の旧 PyInstaller ビルドスクリプト（`build_exe.py`, `jasna.spec`）は **削除済み** です。

したがって公開でサポートされる経路は **ソースから実行**（11節）です。パッケージ済みバイナリが必要なら upstream Kruk2/jasna の公式リリースを使ってください。

---

## 11. ソースから実行 / 動作確認

1〜9節が完了したら、venv 内でソースチェックアウトから直接 Jasna を実行する:

```powershell
cd $Workspace\jasna
jasna --version          # -> 0.7.1+modi
jasna --help
jasna --input assets\test_clip1_1080p.mp4 --output $env:TEMP\out.mp4   # 短いクリップを処理
jasna                    # GUI 起動（引数なし）
```

`model_weights\` は自動解決される（付録 A.1）。v8 の `ffmpeg`/`ffprobe` CLI と `mkvmerge` が `PATH` 上にあること（セクション 3.2・1）、および `CUDA_PATH` が CUDA 13.2 を指していてネイティブライブラリが CUDA `bin\` を見つけられること（付録 A.2）を確認する。

---

## トラブルシューティング

- **`uv pip install .` が CMake エラー / `cl.exe not found` で失敗する**
  セクション 3.3 の VS Build Tools ロードを実行していないセッションでビルドしている。`cl.exe` が PATH に通っているか `where.exe cl.exe` で確認。

- **`vali` ビルドが `AVCODEC_INCLUDE_DIRS / AVCODEC_LIB ... NOTFOUND` 等の CMake エラーで失敗する**
  ffmpeg の **shared dev ビルド** が `C:\Program Files\ffmpeg8\` に展開されていない、もしくは `bin/` だけしか入っていない。セクション 5 のとおり `include/`, `lib/`, `bin/` を含むディストリビューションを配置する。`vali/src/CMakeLists.txt` は `FFMPEG_ROOT` を `C:/Program Files/ffmpeg8` にハードコードしており **`-DFFMPEG_ROOT=...` (`CMAKE_ARGS`) を渡しても上書きされない**。既定外の場所に置きたい場合はセクション 5.2 の junction か、vali ソースの条件付き編集を行う。

- **`PyNvVideoCodec` ビルドが `avcodec_library-NOTFOUND` 等の CMake エラーで失敗する**
  PyNvVideoCodec は **環境変数 `FFMPEG_DIR`** を参照する。未設定時は同梱 `external/ffmpeg/` (lib なし) にフォールバックして失敗する。セクション 7 のとおり `$env:FFMPEG_DIR = "C:/Program Files/ffmpeg8"` をビルド前に設定する (フォワードスラッシュ必須)。`vali` の `FFMPEG_ROOT` (CMake 変数) とは別物。

- **`PyNvVideoCodec` ビルドの install 段で `Syntax error in cmake code ... Invalid character escape '\P'` で失敗する**
  `$env:FFMPEG_DIR` がバックスラッシュを含むパス (`C:\Program Files\ffmpeg8`) で設定されている。CMake の install スクリプトがこれを quoted string として処理しようとして `\P` を不正エスケープとみなす。フォワードスラッシュに修正 (`"C:/Program Files/ffmpeg8"`) し、`_skbuild` を削除して再ビルドする。

- **日本語 Windows で `PyNvVideoCodec` ビルドが `error C2220: 警告がエラーとして扱われます` / `warning C4819: 現在のコードページ (932) で表示できない文字` で失敗する**
  CUDA / ffmpeg のヘッダに含まれる UTF-8 文字 (著者名のアクセント記号等) を CP932 で解釈しようとして MSVC が C4819 警告を出し、`/WX` で error に昇格している。`$env:CL = "/utf-8"` をビルド前に設定すれば MSVC がソースを UTF-8 として解釈し、警告自体が出なくなる。セクション 3.4 のとおりビルドセッション全体で設定するのが楽。

- **`uv pip install -e .[dev]` が `No solution found ... no version of torch==2.12.0+cu130` で失敗する**
  PyTorch インデックスが未指定。`--extra-index-url https://download.pytorch.org/whl/cu130` を追加する。

- **`uv pip install -e .[dev]` が `torch-tensorrt==2.12.0+cu130 ... unsatisfiable` で失敗する**
  uv のデフォルト index 戦略は first-index 限定で、`torch-tensorrt==2.12.0` (pyproject 指定) を PyTorch インデックスの `2.12.0+cu130` (ローカルバージョン付き) で満たすことを拒否する。さらに推移依存 `nvidia-cuda-runtime-cu13==0.0.0a0` はプレリリース。`--index-strategy unsafe-best-match --prerelease=allow` を追加する (セクション 8 のコマンド参照)。

- **`vali` / `PyNvVideoCodec` のビルドが `ModuleNotFoundError: No module named 'pkg_resources'` で失敗する**
  `setuptools` 80 以降は `pkg_resources` を標準同梱しなくなったが、PyPI 上に独立した `pkg_resources` パッケージは存在しない。`uv pip install pkg_resources` も "No solution found" で失敗する。即時対処は `setuptools` をダウングレード:
  ```powershell
  uv pip install "setuptools<80"     # 例: 79.0.1
  ```
  その後 `uv pip install . --no-build-isolation` を再実行。セクション 4 の事前インストールで既に `"setuptools<80"` を pin していれば発生しない。

- **CMake が誤った CUDA バージョン (例: 13.0 など別の版) を検出する**
  セクション 3.1 の `CUDA_PATH` 設定を行っていないか、別バージョンが PATH 先頭にある。`Get-Command nvcc | Select-Object Source` で参照先を確認。

- **Jasna が `.pth`/`.onnx`/`.pt` で `FileNotFoundError`**
  `model_weights\` のファイル欠落、またはリゾルバが別の場所を見ている。3 つのウェイトを `$Workspace\jasna\model_weights\`（9節）に置くか、`JASNA_MODEL_WEIGHTS_DIR` をそれらが入ったフォルダに設定する。

- **Jasna が起動拒否: ffmpeg/ffprobe のバージョン違い、または mkvmerge 不在**
  Jasna は `ffmpeg`/`ffprobe` の **メジャーバージョン 8** と、`PATH` 上の `mkvmerge` を要求する。ffmpeg の `bin\` を `PATH` に追加し（セクション 3.2）、MKVToolNix を導入する（セクション 1）。

- **`import python_vali` で DLL ロードエラー**
  `CUDA_PATH` が CUDA 13.2 を指しているか確認。ネイティブライブラリには CUDA `bin\` を DLL 検索パスに登録する必要があり、Jasna はこれを自動で行う（付録 A.2）が、単体 import 時は先に `os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'],'bin'))` を実行する（`vali` 側 README 記載の仕様）。

- **初回起動が極端に遅い**
  異常ではない。TensorRT エンジン初回コンパイルに 15–60 分かかる。`~/.jasna/engines/` 配下にキャッシュされ 2 回目以降は高速化される。

---

## 付録 A: 本ブランチのビルド/ランタイム改修の内容

本ブランチのソースへ適用済みの Windows ビルド/実行環境向け修正の内訳（参照用の記述。独立した `.patch` ファイルは同梱しない）。

### A.1 `model_weights/` ディレクトリの自動解決

**問題**: 既定の `--detection-model-path` / `--restoration-model-path` のデフォルトが `Path("model_weights")` 相対パス。任意フォルダから Jasna を実行すると、`<CWD>/model_weights/...` を探して `FileNotFoundError`。

**修正**:

- **新規 `jasna/model_weights_resolver.py`**: `model_weights/` フォルダを以下の優先順で自動的に探す。
  1. **環境変数 `JASNA_MODEL_WEIGHTS_DIR` で指定したフォルダ** — ユーザーが「ここを使え」と明示した場所が最優先
  2. **実行ファイルと同じフォルダの中の `model_weights\`** — パッケージ済みインストールの標準配置がここ
  3. **コマンドを実行したフォルダの中の `model_weights\`** — `PowerShell` で `cd` してきた、いま自分がいる場所
  4. **jasna ソースの親フォルダの中の `model_weights\`** — `uv pip install -e .` でソースから入れて動かしている開発者向け

  最初に見つかったものを使う。どこを採用したかは `--log-level info` で起動時にログ表示される（同じ値が続けば重複ログは抑止）。
- **`jasna/main.py`**: `--restoration-model-path` のデフォルトを `""` に変更。未指定時にリゾルバ経由で解決。help 文に探索順を併記
- **`jasna/mosaic/detection_registry.py`**: `detection_model_weights_path()` / `discover_available_detection_models()` がリゾルバを利用
- **`jasna/engine_paths.py`**: `UNET4X_ONNX_PATH` をリゾルバから算出
- **`jasna/gui/processor.py`, `jasna/gui/engine_preflight.py`**: GUI 側のハードコード `Path("model_weights")` をリゾルバ呼び出しに置換

### A.2 Windows ランタイムでの DLL ロード補助

**問題**: Windows では DLL 検索パスが限定されており、`python_vali` のロード時に依存 DLL (CUDA 含む) が見つからないことがある。

**修正** (`jasna/media/video_decoder.py`):

```python
if sys.platform == "win32":
    _vali_spec = importlib.util.find_spec("python_vali")
    if _vali_spec and _vali_spec.origin:
        os.add_dll_directory(str(Path(_vali_spec.origin).parent))
    _cuda_path = os.environ.get("CUDA_PATH")
    if _cuda_path:
        _cuda_bin = os.path.join(_cuda_path, "bin")
        if os.path.isdir(_cuda_bin):
            os.add_dll_directory(_cuda_bin)
```

`python_vali` のロード前に、その隣にある DLL と CUDA `bin/` を DLL 検索パスに追加する。

### A.3 BasicVSR++ ベンチの新 TRT API 追従

**問題**: `jasna/restorer/basicvsrpp_sub_engines.py` は既に新しい `_preprocess_engine` API (feat_extract + flow を 1 つの TRT エンジンに統合) を使っているが、ベンチコード `jasna/benchmark/basicvsrpp_restoration.py` だけ古い `_feat_extract_engine` + 別 flow 計算の API を呼んでいた。`--benchmark basicvsrpp` 実行時に `AttributeError` で失敗する。

**修正** (`jasna/benchmark/basicvsrpp_restoration.py`):

- `split._feat_extract_engine` + 別途 `compute_flow` 呼び出しを、`split._preprocess_engine` 1 回の呼び出しに置換
- 不要になった `torch.nn.functional` import を削除

### 完全な diff

10 ファイル（更新 9 + 新規 1）、約 290 行。完全な内容は本ブランチを upstream と比較（`git diff upstream/main..modi`）して確認。
