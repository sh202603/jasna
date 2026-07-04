# Building Jasna for Linux

Linux で Jasna のビルド依存をセットアップし、**ソースから実行**する手順です。

> **本ガイドは `v0.7.2+modi` ブランチの手順です。** GPU スタック（**torch 2.12.0+cu130 / torchvision 0.27.0+cu130 / torch-tensorrt 2.12.0+cu130 / tensorrt 10.16.1.11**）で、これらの依存ピンは本ブランチの `pyproject.toml` に適用済みです。TensorRT が **10.16** 系に留まるのは、`torch-tensorrt==2.12.0` が `tensorrt>=10.16.1,<10.17.0` を要求するためです（torch-tensorrt は TensorRT 11 に未対応）。
>
> **このブランチの新機能:** AV1 出力 / 8bit(NV12) 出力 / BT.601 と BT.2020 の色空間保持（[CODECS_AND_COLORSPACE_ja.md](CODECS_AND_COLORSPACE_ja.md) 参照）、および RIFE による 2x/4x フレーム生成（[FRAME_GENERATION_ja.md](FRAME_GENERATION_ja.md) 参照）。

> **パッケージングについて:** この公開フォークはネイティブ GPU 依存をビルドし、Jasna を**ソースから実行**します。frozen/パッケージ化されたバイナリを生成する **Nuitka のパッケージングツールは同梱していません**。そのツールは upstream と同様に**プライベート**です。ソース実行ではなくパッケージ済みバイナリが欲しい場合は upstream Kruk2/jasna の公式リリースを使ってください。詳細は [パッケージング / frozen ビルド](#10-パッケージング--frozen-ビルド)。

> 本ガイドは **Ubuntu 26.04 LTS**（ffmpeg 8 / gcc 15 / glibc 2.43 を同梱）+ **CUDA 13.3** + RTX 50 シリーズ GPU + **ドライバ 595** で検証済みです。他のディストリでも動作しますが、パッケージ名や ffmpeg/CUDA の導入手順は異なります。

---

## 1. 前提ソフトウェア

> ⚠ **Linux 固有の注意点が 2 つあります:**
> 1. **ffmpeg はシステムの dev パッケージを使い、ダウンロードはしません。** この `vali` フォークは ffmpeg を自動ダウンロードしません（upstream の VALI README にはダウンロードすると書かれていますが、こちらには当てはまりません）。`apt` で ffmpeg-8 の **dev** ライブラリを入れ、それにリンクします。PyNvVideoCodec はさらに、それらを `FFMPEG_DIR` で特定のディレクトリ構成として渡す必要があります（5節）。
> 2. **システムの `python3.13` と、その `-dev` ヘッダに対してビルドする必要があります**（self-contained / managed Python は不可）。PyNvVideoCodec の pybind11 が Python の include ディレクトリを `/usr/include/python3.13` と解決し、これは `python3.13-dev` を入れたときだけ存在します。**Ubuntu 26.04 の既定 `python3` は 3.14** なので注意。3.14 では vali / PyNvVideoCodec の include パス解決と torch cu130 wheel が合わずビルドや解決に失敗します（`pyproject.toml` は `requires-python = ">=3.13,<3.14"` で 3.14 を明示的に拒否）。`python3.13` が標準リポジトリに無ければ deadsnakes 等の追加が必要で、venv を作る前に **`/usr/bin/python3.13` が存在することを確認**してください（`ls -l /usr/bin/python3.13`）。

| カテゴリ | 要件 | 入手元 | 備考 |
|---|---|---|---|
| OS | Ubuntu 26.04 x64（または同等） | n/a | 26.04 で検証。`apt` に ffmpeg 8 がある |
| ビルドツール | `build-essential`, `pkg-config` | apt | gcc/g++ 15 で問題なし。CUDA 13.3 の `nvcc` はホストコンパイラとして受け入れる |
| cmake / ninja | 最近の版 | apt **または** venv（4節） | `pip install` 時は venv 側が使われる。システム側は任意 |
| Python | **3.13 + 3.13-dev + 3.13-tk** | apt (`python3.13 python3.13-dev python3.13-tk`) | dev ヘッダは vali（`Development.Module`）と PyNvVideoCodec（pybind11）に必須。`python3.13-tk`（Tcl/Tk）は GUI の実行に必要で、無いと `tkinter`/`customtkinter` の import に失敗する |
| uv | 最新 | [astral.sh/uv](https://docs.astral.sh/uv/) | venv を管理 |
| CUDA Toolkit | **13.x**（13.3 で検証） | NVIDIA apt リポジトリ | toolkit のみ。ドライバは**置き換えない**。`nvcc` が 13.x を指すこと（3節） |
| NVIDIA ドライバ | 590+（59x 系列） | ディストリ / NVIDIA | GPU は compute capability 7.5+ |
| ffmpeg / ffprobe | **v8**（実行時）+ **dev ライブラリ**（ビルド時） | apt | 実行時 CLI はメジャー 8。ビルドには `libav*-dev`（1.1節） |
| MKVToolNix | `mkvmerge`（実行時） | apt (`mkvtoolnix`) | 実際に動画処理するときのみ必要。ビルド自体には不要 |

### 1.1 システムパッケージのインストール

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential pkg-config cmake ninja-build \
  python3.13 python3.13-dev python3.13-tk \
  libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev \
  libavfilter-dev libavdevice-dev \
  ffmpeg mkvtoolnix
```

`uv` が無ければ導入します:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.2 CUDA 13 toolkit のインストール

`vali` と `PyNvVideoCodec` のコンパイルに CUDA *toolkit*（`nvcc`、ヘッダ、`cudart`）が必要です。**toolkit のみ** を入れます。既存のドライバには触れません。

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-3
```

`/usr/local/cuda-13.3` にインストールされます。（別の Ubuntu 版なら `ubuntu2604` を `ubuntu2404` 等に置換。）

> `cuda-toolkit-13-3` パッケージは意図的にドライバを**引き込みません**。ドライバも置き換えたいのでない限り、`cuda` メタパッケージは入れないでください。

---

## 2. リポジトリのクローン

3 つのリポジトリを **同じ親ディレクトリ** に置きます。本ガイドではそれを `$WORKSPACE` と呼びます。

```bash
export WORKSPACE="$HOME/jasna-dev"     # ← 任意のディレクトリ
mkdir -p "$WORKSPACE" && cd "$WORKSPACE"

git clone https://codeberg.org/Kruk2/vali.git
git clone https://codeberg.org/Kruk2/PyNvVideoCodec.git
git clone -b modi https://github.com/sh202603/jasna.git   # modi ブランチ（本フォークの既定・3パッチ適用済み）

# vali はサブモジュール (extern/dlpack, cmake-modules) を持つので初期化
cd vali && git submodule update --init --recursive && cd ..
```

レイアウト:

```
$WORKSPACE/
  vali/
  PyNvVideoCodec/
  jasna/          <- 以降の作業ルート
```

---

## 3. ビルド環境変数 (CUDA)

ビルドする **同じシェル** で設定します。`vali` / `PyNvVideoCodec` のビルドが参照します。

```bash
export CUDA_PATH=/usr/local/cuda-13.3
export CUDAToolkit_ROOT=/usr/local/cuda-13.3
export PATH="$CUDA_PATH/bin:$PATH"
```

`nvcc` が 13.x を指すか確認（古い `nvcc` が `PATH` にあっても、上記の prepend が勝ちます）:

```bash
nvcc --version            # "release 13.x" と表示されること
which nvcc                 # /usr/local/cuda-13.3/bin/nvcc
```

### 3.0 ビルド環境変数の早見表

| 環境変数 | 値 | 消費者 | 節 |
|---|---|---|---|
| `CUDA_PATH` / `CUDAToolkit_ROOT` | `/usr/local/cuda-13.3` | vali CMake, nvcc | 3 |
| `PATH`（先頭追加） | `$CUDA_PATH/bin` | shell / nvcc | 3 |
| `VIRTUAL_ENV` | `$WORKSPACE/jasna/.venv` | `uv pip install` の対象 | 4 |
| `FFMPEG_DIR` | `$WORKSPACE/ffmpeg-prefix` | PyNvVideoCodec CMake | 5 / 7 |
| `CUDACXX` | `/usr/local/cuda-13.3/bin/nvcc` | PyNvVideoCodec CMake（CUDA 13 の `sm_52` 回避） | 7 |

---

## 4. Python 仮想環境の作成

`jasna` リポジトリ配下に `.venv` を、**システムの `python3.13` から**作成します（managed/standalone Python は不可。1節の注意参照）。

```bash
cd "$WORKSPACE/jasna"
ls -l /usr/bin/python3.13                       # 無ければ python3.13 / python3.13-dev を先に入れる（1.1節）
uv venv --python /usr/bin/python3.13 .venv
export VIRTUAL_ENV="$WORKSPACE/jasna/.venv"
python --version                                # -> Python 3.13.x（3.14 なら作り直し）
```

> ⚠ **`--python` は必ずパス（`/usr/bin/python3.13`）で渡すこと。** バージョン要求形式（`--python 3.13`）や `--python` 省略にすると、`/usr/bin/python3.13` が無い環境では uv が managed CPython（最新 = **3.14**）を取得したり、Ubuntu 26.04 の既定 `python3`（3.14）を拾ってしまいます。`python --version` が 3.14 を返したら `rm -rf .venv` してパス指定で作り直してください。

ビルド用ツールを venv に入れます（`--no-build-isolation` がこれらを使えるように）:

```bash
uv pip install cmake ninja scikit-build "setuptools<80" wheel numpy
```

> **`setuptools<80` の理由**: `setuptools` ≥ 80 は `pkg_resources` を同梱しなくなり、`vali` / `PyNvVideoCodec` は `setup.py` で `pkg_resources` を import します。80 未満に固定してください（~79.0.1 に解決）。

---

## 5. ffmpeg-8 をビルドに見せる

`vali` はシステムの ffmpeg-8 dev ライブラリを自動的に見つけます（CMake の既定探索が `/usr/include/x86_64-linux-gnu` + `/usr/lib/x86_64-linux-gnu` を拾う）。よって追加設定は不要で、1.1節の `apt` パッケージで足ります。

**PyNvVideoCodec は別です。** CMake が `include/` と `lib/x86_64/` サブディレクトリを持つ ffmpeg prefix を期待しますが、これは Ubuntu の multiarch レイアウトと **一致しません**。システムの ffmpeg-8 ファイルを指す小さな symlink prefix を作ります（ヘッダとライブラリのバージョンを揃えたまま）:

```bash
export FFMPEG_DIR="$WORKSPACE/ffmpeg-prefix"
rm -rf "$FFMPEG_DIR" && mkdir -p "$FFMPEG_DIR/include" "$FFMPEG_DIR/lib/x86_64"

for d in /usr/include/x86_64-linux-gnu/lib{av,sw}*; do
  ln -sfn "$d" "$FFMPEG_DIR/include/$(basename "$d")"
done
for so in /usr/lib/x86_64-linux-gnu/lib{avcodec,avformat,avutil,swresample,swscale,avfilter,avdevice}.so*; do
  ln -sf "$so" "$FFMPEG_DIR/lib/x86_64/$(basename "$so")"
done

ls "$FFMPEG_DIR/include"        # libavcodec, libavformat, ...
ls "$FFMPEG_DIR/lib/x86_64"     # libavcodec.so, libavcodec.so.62, ...
```

---

## 6. vali のビルドとインストール

> **前提**: 3節（CUDA 環境）+ 4節（venv + `setuptools<80`）+ 1.1節（ffmpeg dev ライブラリ）。

`vali` の `setup.py` は既に `-DCMAKE_CUDA_ARCHITECTURES=native` を渡し、`CUDA_PATH` から `nvcc` を固定するため、そのまま configure とビルドができます:

```bash
cd "$WORKSPACE/vali"
uv pip install . --no-build-isolation
```

ビルドは ffmpeg の `.so` をインストール済みパッケージに同梱します。import を確認します（torch を未ロードのときは、ローダに CUDA ランタイムを示す）:

```bash
LD_LIBRARY_PATH=/usr/local/cuda-13.3/targets/x86_64-linux/lib \
  python -c "import python_vali; print('vali OK', python_vali.__version__)"
```

> Jasna 実行時は `LD_LIBRARY_PATH` 不要です。torch (cu130) が同梱の CUDA 13 ランタイムを先にロードし、`vali`/`PyNvVideoCodec` を満たします。上記の `LD_LIBRARY_PATH` はネイティブライブラリを単体 import するときだけ必要です。

---

## 7. PyNvVideoCodec のビルドとインストール

> **前提**: 6節と同じ + `FFMPEG_DIR` prefix（5節）。

CUDA 13 のシステムでは追加の環境変数が 2 つ必要です:

- **`FFMPEG_DIR`**：5節の symlink prefix。
- **`CUDACXX`**：`CMAKE_CUDA_COMPILER` を先に設定し、CMake のコンパイラ識別プローブをスキップさせます。このプローブは `-arch=sm_52` をハードコードしており、**CUDA 13 では `sm_52` が削除**されています（`ptxas fatal: Value 'sm_52' is not defined`）。コンパイラを先に設定すればプローブを回避でき、プロジェクト既定のアーキ一覧（`75;80;86;89;120`）は CUDA 13 で有効です。

```bash
cd "$WORKSPACE/PyNvVideoCodec"
export FFMPEG_DIR="$WORKSPACE/ffmpeg-prefix"
export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
uv pip install . --no-build-isolation
```

確認:

```bash
LD_LIBRARY_PATH=/usr/local/cuda-13.3/targets/x86_64-linux/lib \
  python -c "import PyNvVideoCodec as p; print('PyNvVideoCodec OK', p.__version__)"
```

> 設定を変えて再ビルドする場合は、先に古いキャッシュを消します: `rm -rf _skbuild`。

---

## 8. jasna 本体のインストール

> **前提**: venv に `python_vali`（6節）と `PyNvVideoCodec`（7節）が入っていること。

`pyproject.toml` は `torch==2.12.0+cu130` / `torchvision==0.27.0+cu130` / `torch-tensorrt==2.12.0` に依存し、これらは既定の PyPI にありません。uv を PyTorch cu130 インデックスに向け、フラグを 2 つ追加します:

```bash
cd "$WORKSPACE/jasna"
uv pip install -e .[dev] \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

各フラグの理由:

- `--extra-index-url https://download.pytorch.org/whl/cu130`：`torch+cu130` / `torch-tensorrt+cu130` wheel の入手元。
- `--index-strategy unsafe-best-match`：`torch-tensorrt==2.12.0` をインデックスの local-version リリース `2.12.0+cu130` で満たせるようにする。
- `--prerelease=allow`：推移依存（`nvidia-cuda-runtime-cu13`）がプレリリース。

`[dev]` extra は `nuitka>=2.4`, `pytest`, `pytest-cov`, `scikit-build`, `cmake`, `ninja` を入れます。

**オプション: torchcodec バックエンド。** 実験的な torchcodec のデコード/エンコード経路（`--video-backend torchcodec`/`auto`）を使う場合は、`torchcodec` extra を追加し、同じフラグで `.[dev,torchcodec]` を入れます:

```bash
uv pip install -e .[dev,torchcodec] \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

これで `torchcodec>=0.14.0` が入ります。通常のビルドには不要で、既定の `native` バックエンドは torchcodec なしで動作します。詳細は [TORCHCODEC_BACKEND_ja.md](TORCHCODEC_BACKEND_ja.md)。

**補足: FP8 復元バックエンドに追加のインストール手順は不要です。** 依存 `nvidia-cudnn-frontend` は `pyproject.toml` の通常依存で、上記コマンドで一緒に入ります。cuDNN ランタイム（9.17 以上）は torch cu130 wheel に同梱済みです。機能自体は実行時 opt-in（`--fp8-recon`、FP8 対応 GPU sm89 以上が必要）で、使えない環境では TensorRT エンジンにフォールバックします。詳細は [FP8_RECON_ja.md](FP8_RECON_ja.md)。

### 8.2 ONNX パッケージ（YOLO 検出モデル用）

**YOLO（lada-yolo-\*）検出モデル**を使う場合、ultralytics は TensorRT エンジンをビルドする前にモデルを ONNX へエクスポートします。これには自動では入らない 3 パッケージが必要です。（RF-DETR モデルは不要。事前ビルド済み `.onnx` を TensorRT が直接パースするため）

```bash
uv pip install onnx onnxslim onnxruntime
```

ソースから実行する場合、ultralytics が初回エクスポート時に未導入ならこれらを自動ダウンロードしますが、事前に入れておくと初回 YOLO 実行時の待ちを回避できます。

スモークテスト:

```bash
python -m jasna --version     # -> 0.7.2+modi
python -m jasna --help
```

### 8.1 mmengine パッチの適用 (torch 2.6+ 対応)

`mmengine.runner.checkpoint` 内の `torch.load` 呼び出しに `weights_only=False` を追加します。torch 2.6 以降は既定が `weights_only=True` になり、既存の `.pth` チェックポイント読み込みが壊れるためです。diff は `patches/` に同梱。

```bash
cd "$WORKSPACE/jasna"
patch -p1 -d .venv/lib/python3.13/site-packages \
    < patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff

# 確認
grep -n "weights_only=False" .venv/lib/python3.13/site-packages/mmengine/runner/checkpoint.py
```

> `uv pip install -e .[dev]` を再実行すると `mmengine` が再インストールされ、このパッチが消えます。その後は再適用してください。

---

## 9. モデルウェイトとアセットの配置

`$WORKSPACE/jasna/model_weights/` に以下 3 ファイルを置く:

- `lada_mosaic_restoration_model_generic_v1.2.pth`
- `rfdetr-v5.onnx`
- `lada_mosaic_detection_model_v4_fast.pt`

ソースから実行すると Jasna は `model_weights/` を自動解決する（付録 A.1）ため、ここに置くだけでよい。

テストクリップ 2 本は通常リポジトリに同梱され `$WORKSPACE/jasna/assets/`（`test_clip1_1080p.mp4`, `test_clip1_2160p.mp4`）にある。無い場合は upstream リリースから抽出する。

### 9.1 オプション: RIFE フレーム補間モデル (`--frame-gen` 用)

フレームレート倍化 (`--frame-gen {2x,4x}`) を使う場合のみ必要。RIFE 重みは**同梱されない**（非商用条項のため）ので、`make_rife_torchscript.py` で TorchScript チェックポイントを自分で作成する。

1. Practical-RIFE をクローンし、4.x のモデルパッケージ（**v4.25 で動作確認**）をダウンロードして `<repo>/train_log/` に `RIFE_HDv3.py`, `IFNet_HDv3.py`, `flownet.pkl` が揃う状態にする:
   ```bash
   git clone https://github.com/hzwer/Practical-RIFE
   # README に従いモデルパッケージを Practical-RIFE/train_log/ にダウンロード・展開
   ```
2. プロジェクトの venv で TorchScript に変換する（`$WORKSPACE/jasna` で実行）:
   ```bash
   .venv/bin/python make_rife_torchscript.py \
       --rife-repo /path/to/Practical-RIFE \
       --output model_weights/rife.pth --validate
   ```

**ソースから実行**する場合は `model_weights/rife.pth` を自動で参照する（他のウェイトと同じリゾルバ経由。または `JASNA_MODEL_WEIGHTS_DIR` をそれを含むフォルダに向ける）。詳細手順: [docs/FRAME_GENERATION_ja.md](FRAME_GENERATION_ja.md)。

---

## 10. パッケージング / frozen ビルド

現時点で、このフォークから **パッケージ化された frozen バイナリを生成する公開手段はありません。** upstream はビルドを PyInstaller から **Nuitka** へ切り替えましたが、Nuitka のビルドスクリプトや手順は公開しておらず、実際のパッケージングツールは **プライベートな submodule（`jasna/protection`）** にあり、この公開フォークには含まれません。フォーク側の旧 PyInstaller ビルドスクリプト（`build_exe.py`, `jasna.spec`）は **削除済み** です。

したがって公開でサポートされる経路は **ソースから実行**（11節）です。パッケージ済みバイナリが必要なら upstream Kruk2/jasna の公式リリースを使ってください。

---

## 11. ソースから実行 / 動作確認

1〜9節が完了したら、venv 内でソースチェックアウトから直接 Jasna を実行します:

```bash
cd "$WORKSPACE/jasna"
jasna --version              # -> 0.7.2+modi
jasna --help
jasna --input assets/test_clip1_1080p.mp4 --output /tmp/out.mp4   # 短いクリップを処理
jasna                        # GUI を起動（引数なし）
```

`model_weights/` は自動解決され（付録 A.1）、`ffmpeg`/`ffprobe`（v8）と `mkvmerge` はシステムの `PATH` から使われます（1.1節）。

> **最初の処理実行は遅い。** 初回使用時に GPU 向け TensorRT エンジンがコンパイルされます（15〜60 分）。`model_weights/` 内のウェイト隣にキャッシュされ、以降は再利用されます。

---

## トラブルシューティング

### 環境 / Python

- **vali configure 失敗: `Could NOT find Python3 (missing: ... Development.Module)`**
  Python の開発ヘッダが無い。`python3.13-dev` を入れ、venv を `/usr/bin/python3.13` から作成（1.1 と 4 節）。managed/standalone Python は使わないこと。

- **PyNvVideoCodec generate 失敗: `pybind11::module includes non-existent path /usr/include/python3.13`**
  同じ原因：`python3.13-dev` 未導入、または venv を非システム Python から作った。`python3.13-dev` を入れ、`/usr/bin/python3.13` から venv を作り直す。

- **`vali` / `PyNvVideoCodec` ビルド失敗: `ModuleNotFoundError: No module named 'pkg_resources'`**
  `setuptools` ≥ 80 が `pkg_resources` を落とした。venv で `uv pip install "setuptools<80"`（4節）してから再ビルド。

- **GUI 起動時に `ModuleNotFoundError: No module named 'tkinter'`**
  GUI には Tcl/Tk が必要で、これは `python3.13-tk` パッケージです（注: `python3-tk` は別のインタプリタ向け）。venv を作成した `python3.13` 向けに導入してください: `sudo apt-get install -y python3.13-tk`。CLI は無くても動作し、GUI のみ必要です。

### CUDA

- **`nvcc --version` が違う版を表示**
  別の CUDA が `PATH` の先にある。`PATH="$CUDA_PATH/bin:$PATH"` を再 export（3節）し `which nvcc` を確認。

- **PyNvVideoCodec configure 失敗: `ptxas fatal: Value 'sm_52' is not defined for option 'gpu-name'`**
  CUDA 13 は Maxwell の `sm_52` ターゲットを削除したが、CMake のコンパイラ識別プローブがそれをハードコードしている。CUDA コンパイラを先に設定してプローブをスキップ: `export CUDACXX=/usr/local/cuda-13.3/bin/nvcc`（7節）。

- **gcc が「新しすぎる」**
  CUDA 13.3 の `nvcc` は gcc 15 を受け入れます（検証済み）。将来の toolkit がホスト gcc を拒否したら、`gcc-14 g++-14` を入れて `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14`（または `export NVCC_PREPEND_FLAGS='-ccbin g++-14'`）を追加。

### ffmpeg

- **vali configure 失敗: `AVCODEC_LIB ... NOTFOUND`**
  ffmpeg の `-dev` パッケージが未導入。`libav{codec,format,util}-dev libsw{resample,scale}-dev` を入れる（1.1節）。

- **PyNvVideoCodec が ffmpeg を見つけられない / ヘッダが違う**
  `FFMPEG_DIR` が未設定、またはレイアウトが誤り。`include/` と `lib/x86_64/` を持つ symlink prefix を作り直し（5節）、ビルド前に `export FFMPEG_DIR=...`。

### jasna インストール

- **`uv pip install -e .[dev]` が `no version of torch==2.12.0+cu130` で失敗**。`--extra-index-url https://download.pytorch.org/whl/cu130` を追加。
- **`torch-tensorrt==2.12.0+cu130 ... unsatisfiable`**。`--index-strategy unsafe-best-match --prerelease=allow` を追加（8節）。

### ランタイム

- **`import python_vali` が単体で CUDA `.so` ロードエラー**
  `import torch` の後に実行する（同梱の CUDA 13 ランタイムがロードされる）か、`LD_LIBRARY_PATH=/usr/local/cuda-13.3/targets/x86_64-linux/lib` を設定。Jasna 内では自動。

- **Jasna が `.pth`/`.onnx`/`.pt` で `FileNotFoundError`**
  `model_weights/` のファイル欠落、またはリゾルバが別の場所を見ている。3 つのウェイトを `$WORKSPACE/jasna/model_weights/`（9節）に置くか、`JASNA_MODEL_WEIGHTS_DIR` をそれらが入ったフォルダに設定する。

- **Jasna が起動拒否: ffmpeg/ffprobe のバージョン違い、または mkvmerge 不在**
  Jasna は `ffmpeg`/`ffprobe` の **メジャーバージョン 8** と、`PATH` 上の `mkvmerge` を要求します。`apt` で導入（1.1節）。

- **RTX Super-Res で `IRuntime::deserializeCudaEngine ... Serialization assertion stdVersionRead == kSERIALIZATION_VERSION failed` / `Version tag does not match`**
  `nvidia-vfx`（nvvfx）パッケージは自前の TensorRT（10.9）を同梱し `RTLD_GLOBAL` で読み込みますが、jasna のエンジンは TensorRT 10.16（`tensorrt_libs`）でビルドされています。両者は soname `libnvinfer.so.10` を共有するため、nvvfx 側が先にロードされると `torch-tensorrt` が 10.9 にバインドされ、10.16 製エンジンを読めません。修正は本ブランチで適用済みです（付録 B.2 参照。nvvfx の import 前に `tensorrt_libs` の `libnvinfer.so.10` を `RTLD_GLOBAL` で先読みして 10.16 のシンボルを優先）。

- **`--fp8-recon` と RTX Super-Res の併用で `Unable to load any of {libcudnn_graph.so.9.7.1, ...}` / `Cannot load symbol cudnnCreate` で中断する**
  上の TensorRT 衝突の cuDNN 版です。nvvfx は同梱 libs ディレクトリを `LD_LIBRARY_PATH` の先頭に追記しますが、そこにはサブライブラリを持たない cuDNN 9.7 ディスパッチャだけが置かれています。`cudnn-frontend` は `LD_LIBRARY_PATH` を最初に走査するため、この不完全なコピーを掴んでしまいます。修正は本ブランチで適用済みです（`jasna/restorer/fp8_upsample.py` が `import cudnn` の前に torch 同梱の完全な cuDNN の lib ディレクトリを `LD_LIBRARY_PATH` のさらに先頭へ置き、構築順に関係なく正常なディスパッチャへ解決させます）。

---

## 付録 A: 本ブランチのビルド/ランタイム改修の内容

本ブランチのソースへ適用済みの変更です（参照用の記述。独立した `.patch` ファイルは同梱しない）。クロスプラットフォームで、Windows 固有部分は `sys.platform == "win32"` でガードされ、Linux では無効です。

### A.1 `model_weights/` ディレクトリの自動解決

`jasna/model_weights_resolver.py` を追加し、`model_weights/` を優先順位で探索: `JASNA_MODEL_WEIGHTS_DIR` 環境変数 → 実行ファイルの隣 → カレントディレクトリ → ソースツリーの隣。`main.py`、`mosaic/detection_registry.py`、`engine_paths.py`、GUI（`gui/processor.py`, `gui/engine_preflight.py`）がハードコードの `Path("model_weights")` ではなく resolver を経由するようになります。これで CLI を任意のディレクトリから実行可能に。

### A.2 DLL ロード補助 (`jasna/media/video_decoder.py`)

CUDA / `python_vali` のディレクトリを DLL 検索パスに登録するが **Windows のみ**（`if sys.platform == "win32"`）。Linux では no-op。

### A.3 BasicVSR++ ベンチの TRT API 修正

`jasna/benchmark/basicvsrpp_restoration.py` を新しい `_preprocess_engine` API に更新し `--benchmark basicvsrpp` を動作させる。クロスプラットフォーム。

---

## 付録 B: 本ブランチの Linux GUI / RTX-VSR 修正の内容

本ブランチのソースへ適用済みの Linux 固有のランタイム修正です（参照用の記述。独立した `.patch` ファイルは同梱しない）。いずれも Linux/X11 で GUI と RTX Super-Res を動作させるために必要で、Windows ビルドには影響しません。

### B.1 モーダルダイアログの空表示 (`jasna/gui/app.py`, `jasna/gui/wizard.py`, `jasna/gui/components.py`)

**問題**: Linux/X11 で customtkinter の `CTkToplevel` ダイアログが完全に空（テキストもボタンも無い暗いウィンドウ）で開く。対象: **About**（`app.py` `_show_about`）、**System Check** / 初回ウィザード（`wizard.py` `FirstRunWizard`）、**プリセット作成**と**確認**ダイアログ（`components.py` `PresetDialog`, `ConfirmDialog`）。いずれも生成直後、子ウィジェット描画前に `grab_set()`（場合により `lift()` / `focus_force()`）を呼ぶ。一部のウィンドウマネージャでは、ウィンドウはマップされるが未描画のまま残り、ダイアログがリサイズ不可のため再描画が走らない。

**修正**: 先に全ての子ウィジェットを生成し、その後 `lift()` / `grab_set()` / `focus_force()` を `self.after(200, …)` / `after(250, …)` でイベントループの後続ティックに遅延させ、内容が描画された後にモーダルグラブを確立する。Windows の挙動は不変（同じ呼び出しが 1 ティック遅れて走るだけ）。

### B.2 RTX Super-Res の TensorRT 版数衝突 (`jasna/restorer/rtx_superres_secondary_restorer.py`)

**問題**: RTX Super-Res を有効にすると、jasna の TensorRT エンジン逆シリアライズ中に `IRuntime::deserializeCudaEngine ... Serialization assertion stdVersionRead == kSERIALIZATION_VERSION failed. Version tag does not match` で中断する。`nvidia-vfx`（nvvfx）パッケージは自前の TensorRT **10.9**（`nvvfx/libs/libnvinfer.so.10`）を同梱し、`nvvfx/_lib_loader.py` で `RTLD_GLOBAL` 読み込みする。一方 jasna のパイプラインエンジンは TensorRT **10.16**（`tensorrt_libs`）でビルドされる。両者は soname `libnvinfer.so.10` を共有し、ELF のシンボル解決は「先にグローバルスコープへ入った方」を使う。nvvfx が jasna の TensorRT ランタイムより先にロードされると、`torch-tensorrt` が nvvfx の古い 10.9 にバインドされ、jasna の新しい 10.16 製エンジンを読めない（両者はシリアライズのバージョンタグが異なる）。Windows は DLL ロード順（`tensorrt_libs` を先に）で回避しているが、Linux には同等の仕掛けが無かった。

**修正**: RTX Super-Res リストアモジュールの import 時（`nvvfx` の import より前）に実行される `_preload_tensorrt_runtime()` を追加（upstream `6545b78` が後に同等修正を独自実装したため、関数名は upstream に合わせた）。Linux では `tensorrt_libs` を特定し、その `libnvinfer.so.10` / `libnvinfer_plugin.so.10` を `ctypes.RTLD_GLOBAL` で先読みすることで、TensorRT 10.16 のシンボルを先にグローバルスコープへ入れ、後続の nvvfx のロードも 10.16 に解決させる。Windows では no-op（順序は既存の DLL パス処理が担当）。
