# Building Jasna for Linux

Linux で Jasna のセットアップを行い、**ソースから実行**する手順です。

> **ブランチとベース。** 本ガイドは `0.11.2+modi`（upstream `b623466f` = v0.11.2 タグ）の手順です。GPU スタックは `pyproject.toml` の upstream のピンに追従しています（torch 2.12.1+cu130 / torchvision 0.27.1+cu130 / torch-tensorrt 2.12.0 / tensorrt 10.16.1.11 / comfy-kitchen 0.2.38 / ultralytics 8.4.96 以上）。本ブランチが足しているのは、`nvidia-vfx==0.2.0.0` の固定（upstream は無固定）、FP8 復元用の通常依存 `nvidia-cudnn-frontend`、`torchcodec` extra、任意の `nvidia-rtx` extra（torch 2.12.0 / torchvision 0.27.0 / torch-tensorrt-rtx 2.12.1）です。ブランチが upstream に加えている変更とその理由は [changes_vs_upstream.md](changes_vs_upstream.md) にあります。
>
> **検証済み**: Ubuntu 26.04 LTS + RTX 5080、2026-10-16。実施内容と過去ベースは[付録 B](#付録-b-本セットアップの履歴)。他のディストリでも動作しますが、パッケージ名や ffmpeg の導入手順は異なります。
>
> **VALI は必須。** Kruk2 氏の VALI フォーク（`python_vali` 4.8.8）を自前でビルドして入れます（5.2 節）。`[nvidia]` extra は `python_vali==4.8.8` を固定していますが、4.8.8 は PyPI に公開されていません（PyPI の最新は 4.8.7。2026-10-16 時点）。また jasna の VALI デコーダはフォークにしかない API を使います。そのため CUDA Toolkit と FFmpeg の dev パッケージが前提になります。
>
> **ソースから実行します。** このフォークに Linux 向け frozen バイナリはありません（実験的な Nuitka スクリプトは Windows 専用。[frozen_build.md](frozen_build.md)）。LTX と支援者モデルは upstream の公式リリースでしか動きません。

---

## 1. 前提ソフトウェア

| カテゴリ | 要件 | 入手元 | 備考 |
|---|---|---|---|
| OS | Ubuntu 26.04 x64（または同等） | n/a | 26.04 で検証。`apt` に ffmpeg 8 がある |
| ビルドツール | `build-essential`, `pkg-config` | apt | **必須**。VALI フォーク wheel のビルド（5.2 節）用 |
| Python | **3.13 + 3.13-dev + 3.13-tk** | apt (`python3.13 python3.13-dev python3.13-tk`) | `requires-python` は `>=3.12` だが、本ガイドの検証は 3.13 のみ（Ubuntu 26.04 の既定 `python3` = 3.14 は cu130 GPU スタックとの組み合わせが未検証）。`python3.13-dev` は VALI wheel のビルド（5.2 節）に、`python3.13-tk` は GUI の実行に必要 |
| uv | 最新 | [astral.sh/uv](https://docs.astral.sh/uv/) | venv を管理 |
| NVIDIA ドライバ | **580+** | ディストリ / NVIDIA | 起動チェックが Linux では 580 以上を要求。GPU は compute capability 7.5+ |
| ffmpeg / ffprobe | **v8**（実行時 CLI） | apt | 起動チェックが `ffprobe` のメジャーバージョン 8 を要求。`ffmpeg` CLI は HLS ストリーミング等で使用 |
| FFmpeg 8 dev | `libavcodec-dev` ほか | apt | **必須**。VALI のビルドがリンクする（実行時も distro の FFmpeg ライブラリを使う） |
| CUDA Toolkit | 13.x（`/usr/local/cuda-13.3` で検証） | NVIDIA | **必須**。VALI ビルドの `nvcc` 用。実行時には使わない |

コンパイルするのは VALI フォーク wheel（5.2 節）だけです。PyAV を含むほかの GPU 系パッケージはすべて wheel で入るため、CUDA Toolkit はビルド時の `nvcc` のためだけに要ります。

### 1.1 システムパッケージのインストール

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential pkg-config \
  python3.13 python3.13-dev python3.13-tk \
  ffmpeg \
  libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev
```

`uv` が無ければ導入します:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> ⚠ **apt の ffmpeg は CLI としてだけ使います。** PyPI の av wheel は自前の FFmpeg を同梱しており、distro のライブラリには触れません（4節）。`libav*-dev` は VALI のビルド（5.2 節）が distro の FFmpeg にリンクするために入れます。

---

## 2. リポジトリのクローン

ここでクローンするのは `jasna` です。VALI フォークは 5.2 節でクローンします。本ガイドでは作業ルートを `$WORKSPACE` と呼びます。

```bash
export WORKSPACE="$HOME/jasna-dev"     # ← 任意のディレクトリ
mkdir -p "$WORKSPACE" && cd "$WORKSPACE"

git clone -b modi https://github.com/sh202603/jasna.git   # modi ブランチ（本フォークの既定）
```

---

## 3. Python 仮想環境の作成

`jasna` リポジトリ配下に `.venv` を、システムの `python3.13` から作成します。

```bash
cd "$WORKSPACE/jasna"
ls -l /usr/bin/python3.13                       # 無ければ python3.13 系を先に入れる（1.1節）
uv venv --python /usr/bin/python3.13 .venv
export VIRTUAL_ENV="$WORKSPACE/jasna/.venv"
python --version                                # -> Python 3.13.x（3.14 なら作り直し）
```

> ⚠ **`--python` はパス（`/usr/bin/python3.13`）で渡すこと。** バージョン要求形式（`--python 3.13`）や省略にすると、uv が managed CPython（最新 = 3.14）を取得したり、Ubuntu 26.04 の既定 `python3`（3.14）を拾うことがあります。`python --version` が 3.14 を返したら `rm -rf .venv` してパス指定で作り直してください。

---

## 4. PyAV

ビルドするものはありません。GPU パスは PyAV の current_ctx API と av 18.1 の CUDA stream 明示指定を使い、`pyproject.toml` は `av==18.1.0`（upstream の VALI wheel がリンクしているのと同じリリース）に固定しています。5 節のインストールで PyPI から解決されます。この wheel は十分新しい nv-codec-headers でビルドした FFmpeg 8 を同梱しており（`hevc_nvenc` は `lookahead_level` を受理）、distro の FFmpeg にリンクすることはありません。

2026 年 8 月より前の暫定の自前ビルド wheel（`18.0.0`）が venv に残っていても、5 節のインストールで置き換わります。[付録 B](#付録-b-本セットアップの履歴) を参照。

---

## 5. jasna 本体のインストール

次の順で入れます。`[nvidia]` extra は `python_vali==4.8.8` を固定していますが、4.8.8 は PyPI にありません。そこで先に Kruk2 氏のフォーク（[codeberg.org/Kruk2/vali](https://codeberg.org/Kruk2/vali)）から wheel をビルドし、jasna のインストール時に `--find-links` で渡します。

1. VALI のビルドに要るパッケージを venv に入れる（5.1 節）
2. VALI フォーク wheel をビルドする（5.2 節）
3. jasna 本体を入れる（5.3 節）
4. mmengine パッチを当てる（5.4 節）

### 5.1 ビルド用パッケージの導入

```bash
cd "$WORKSPACE/jasna"
uv pip install cmake ninja scikit-build "setuptools<81"
```

`uv venv` で作った venv には setuptools が入っていません。vali の `setup.py` は `pkg_resources` を import します。setuptools 80.x はその import に「`pkg_resources` は削除予定なので `Setuptools<81` に固定せよ」と警告するため、`<81` に固定します。cmake、ninja、scikit-build は 5.3 節の `[dev]` でも入りますが、ビルドの方が先なのでここで入れます。

### 5.2 VALI フォーク wheel のビルド

native デコードは、まず VALI NVDEC デコーダを試し、使えなければ PyAV へエスカレーションします。環境変数 `JASNA_DECODE_BACKEND`（`auto` / `vali` / `pyav-hw` / `pyav-sw`、既定 `auto`）でこの連鎖を明示指定でき、modi の `--video-backend` はその上位層です（`native` を選んだときにこの連鎖が働きます）。jasna の VALI デコーダはフォークにしかない API `DecodeSingleSurfaceAsyncDetailed` を使います。この API を持たない wheel では、リーダーは警告を出して毎回 PyAV にフォールバックします。

前提（1 節で導入済みのもの）:

- **CUDA Toolkit 13.x**（`/usr/local/cuda-13.3`）。distro の `/usr/bin/nvcc` 単体では CMake の `FindCUDAToolkit` を満たせないため、下のコマンドで toolkit パスを明示します。
- **FFmpeg 8 の dev パッケージ**（`libavcodec-dev` ほか）。av wheel（自前の FFmpeg を同梱）と違い、vali のビルドは distro FFmpeg にリンクし、wheel には何も同梱しません。VALI が FFmpeg を使うのは demux/parse だけなので、av 側の FFmpeg に課される nv-codec-headers の条件はここでは当たりません。

```bash
cd "$WORKSPACE"
git clone https://codeberg.org/Kruk2/vali.git       # 既存 checkout があれば省略
cd vali
git checkout 0e5c01e      # = 4.8.8 ピン: DecodeSingleSurfaceAsyncDetailed、ASF/VC-1 タイムスタンプ修正、decoder-open の GIL 解放
git submodule update --init --recursive

CUDACXX=/usr/local/cuda-13.3/bin/nvcc \
CUDAToolkit_ROOT=/usr/local/cuda-13.3 \
CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc" \
VIRTUAL_ENV="$WORKSPACE/jasna/.venv" \
  uv build --wheel --no-build-isolation
ls dist/                  # -> python_vali-4.8.8-cp313-cp313-linux_x86_64.whl
```

`--no-build-isolation` が必要なのは、vali の `setup.py` が `pkg_resources` を import しており、隔離ビルド環境にはそれが無いためです（5.1 節で venv に入れた setuptools を使わせます）。既存の checkout で別のコミットに切り替えて再ビルドする場合は、前回の CMake キャッシュ `_skbuild/` を消してから行います。ビルド中に CMake が pybind11 をダウンロードするため、ネットワーク接続が要ります。

> フォークコミット `0e5c01e`（= 4.8.8 ピン、2026-10-16、RTX 5080。前回は `3ad0d54` = 4.8.7、2026-07-30）で検証済み: パイプラインの 2 つのデコードパスが両方 VALI を通り、同梱テストクリップの出力は PyAV デコード時と md5 完全一致、デコーダのテストも合格、av wheel および torchcodec バックエンドとの共存も確認（この 2 つは auditwheel でマングル/同梱した FFmpeg を持ち、vali は distro の FFmpeg を ldconfig 経由で解決する）。

### 5.3 jasna 本体のインストール

GPU スタックは extras に分かれています（`nvidia` = NVIDIA スタック、`amd` = ROCm 用）。NVIDIA ビルドでは `nvidia` extra が `torch==2.12.1+cu130` / `torchvision==0.27.1+cu130` / `tensorrt==10.16.1.11` / `torch-tensorrt==2.12.0` / `nvidia-vfx==0.2.0.0` / `comfy-kitchen==0.2.38` / `python_vali==4.8.8` を入れますが、一部は既定の PyPI にありません。5.2 節の wheel の置き場所と PyTorch cu130 インデックスを指定し、フラグを 2 つ追加します:

```bash
cd "$WORKSPACE/jasna"
uv pip install -e .[dev,nvidia] \
    --find-links "$WORKSPACE/vali/dist" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

各フラグの理由:

- `--find-links "$WORKSPACE/vali/dist"`：5.2 節でビルドした `python_vali-4.8.8` の入手元。付け忘れると `python_vali==4.8.8` が解決できず失敗する。
- `--extra-index-url https://download.pytorch.org/whl/cu130`：`torch+cu130` / `torch-tensorrt+cu130` wheel の入手元。
- `--index-strategy unsafe-best-match`：`torch-tensorrt==2.12.0` をインデックスの local-version リリース `2.12.0+cu130` で満たせるようにする。
- `--prerelease=allow`：推移依存（`nvidia-cuda-runtime-cu13`）がプレリリース。

`[dev]` extra は `nuitka>=2.4`, `pytest`, `pytest-cov`, `scikit-build`, `cmake`, `ninja` を入れます。`[nvidia]` extra は GPU スタック（torch cu130 / TensorRT / torch-tensorrt / nvidia-vfx / VALI）を入れます — **指定を忘れると torch が入らず起動しません**。

入ったのがフォーク版の VALI であることを確認します:

```bash
python -c "import python_vali as v; print(hasattr(v.PyDecoder, 'DecodeSingleSurfaceAsyncDetailed'))"   # -> True
# jasna を --log-level info で実行すると "Using VALI NVDEC decoder for <file>" が出ます
```

`False` なら、フォークでない wheel が入っています。5.2 節の wheel で入れ直します（`--no-deps` は uv が venv の他のパッケージに触らないようにするため）:

```bash
uv pip install --force-reinstall --no-deps "$WORKSPACE"/vali/dist/python_vali-4.8.8-*.whl
```

**注: `nvidia-vfx` は 0.2.0.0（VFX SDK 1.3.0）に固定しています。** PyPI 上のエントリは 4 KB の殻で、上記インストール中に `pypi.nvidia.com` から実 wheel（443 MB）を取ってくるため、追加のインデックス指定は要りません。0.2.0.0 は `--frame-gen-backend rtx`（[frame_generation.md](frame_generation.md)）が使う `VideoFrameGeneration` effect を提供し、TensorRT を同梱しません。0.1.0.1 が入った古い venv は `uv pip install nvidia-vfx==0.2.0.0` でその場で更新できます（[付録 B](#付録-b-本セットアップの履歴)）。

**オプション: torchcodec バックエンド。** 実験的な torchcodec のデコード/エンコード経路（`--video-backend torchcodec`/`auto`）を使う場合は、`torchcodec` extra を追加し、同じフラグで `.[dev,nvidia,torchcodec]` を入れます（PyPI の torchcodec は GPU 非対応版のため、PyTorch インデックスの `+cu130` 版を取らせるには `--extra-index-url` が要ります）:

```bash
uv pip install -e .[dev,nvidia,torchcodec] \
    --find-links "$WORKSPACE/vali/dist" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

これで `torchcodec>=0.15.0` が入ります。通常のセットアップには不要で、既定の `native` バックエンド（PyAV）は torchcodec なしで動作します。詳細は [torchcodec_backend.md](torchcodec_backend.md)。

**補足: FP8 復元バックエンドに追加のインストール手順は不要です。** 依存 `nvidia-cudnn-frontend` は `pyproject.toml` の通常依存で、上記コマンドで一緒に入ります。cuDNN ランタイム（9.17 以上）は torch cu130 wheel に同梱済みです。機能自体は実行時 opt-in（`--fp8-recon`、FP8 対応 GPU sm89 以上が必要）で、使えない環境では TensorRT エンジンにフォールバックします。詳細は [fp8_recon.md](fp8_recon.md)。

**任意: TensorRT-RTX フレーバー。** `nvidia` の代わりに `nvidia-rtx` extra を入れると、TensorRT スタック全体が TensorRT-RTX（JIT コンパイル）に切り替わり、エンジンビルドが分単位から秒単位になります。
`torch-tensorrt` と `torch-tensorrt-rtx` は同じ `torch_tensorrt` パッケージを提供するため、両 extra は 1 つの venv に共存できません。専用 venv を使ってください。`nvidia-rtx` extra も `python_vali==4.8.8` を固定しているため、同じ `--find-links` が要ります。torch は 2.12.0 / torchvision 0.27.0 で、`nvidia` extra（2.12.1 / 0.27.1）と版が違います。

```bash
uv pip install -e .[dev,nvidia-rtx] \
    --find-links "$WORKSPACE/vali/dist" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

フレーバーはインストール済み wheel から自動判別されます。
エンジンは `.rtx` タグ付きの名前でキャッシュされるため、両フレーバーで 1 つの `model_weights` ディレクトリを共有できます。
mmengine パッチ（5.4 節）はこの venv にも必要です。
詳細と実測値は [tensorrt_rtx.md](tensorrt_rtx.md)。

### 5.4 mmengine パッチの適用

`patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff` を venv の mmengine 0.10.7 に当てます。diff は 4 ファイルを変更します:

- `mmengine/runner/checkpoint.py`: `torch.load` に `weights_only=False` を足す。torch 2.6 以降は既定が `weights_only=True` で、BasicVSR++ の重み（`lada_mosaic_restoration_model_generic_v1.2.pth`）に入っている mmengine の `HistoryBuffer` を読めずに `Weights only load failed` で止まるため
- `mmengine/model/__init__.py` と `mmengine/model/wrappers/__init__.py`: FSDP ラッパーの import を `torch.distributed.is_available()` のときだけにする（分散機能を持たない torch でも import できるようにする）
- `mmengine/dist/dist.py`: `from __future__ import annotations` を足す

> upstream は `7c64105` でこの diff の 1 行を `weights_only=True` に変えましたが、その版を当てるとソース実行では上記の `v1.2.pth` が読めません（2026-10-16 に確認）。このフォークは `False` のまま同梱しています（[changes_vs_upstream.md](changes_vs_upstream.md) 8 節）。

```bash
cd "$WORKSPACE/jasna"
patch -p1 -d .venv/lib/python3.13/site-packages \
    < patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff

# 確認（どちらも 1 行ヒットすれば適用済み）
grep -n "weights_only=False" .venv/lib/python3.13/site-packages/mmengine/runner/checkpoint.py
grep -n "is_available" .venv/lib/python3.13/site-packages/mmengine/model/__init__.py
```

> パッチは mmengine を入れ直すと消えます（`uv pip install` が mmengine を入れ直すのは、版が変わったときと `--reinstall` を付けたとき）。消えたら再適用してください。当て済みの venv に当て直すと `Hunk #1 FAILED` などで失敗します。旧版や upstream 版の diff が当たっている venv も同じで、その場合は mmengine を素の状態に戻してから当てます: `uv pip install --reinstall --no-deps mmengine==0.10.7`。

### 5.5 YOLO 検出モデルに追加パッケージは不要

YOLO 系検出モデル（`lada-yolo-*`、`zelefans-vr-yolo-v2`）は、TensorRT エンジンをビルドする前に ultralytics が ONNX へエクスポートします。`onnx` は通常依存として入っており、エクスポートは `simplify=False` で走るため `onnxslim` も `onnxruntime` も要りません。RF-DETR モデルは事前ビルド済みの `.onnx` を TensorRT が直接パースします。

---

## 6. モデルウェイトとアセットの配置

`$WORKSPACE/jasna/model_weights/` に以下のファイルを置く（upstream のリリースパッケージから取り出す）:

- `lada_mosaic_restoration_model_generic_v1.2.pth`
- `rfdetr-v6.onnx`（v0.9.0 以降の既定検出器。4K 向け `rfdetr-v6-large.onnx` と VR180 向け `rfdetr-vr-v1.onnx` は任意、旧 `rfdetr-v5.onnx` も引き続き使える）
- `lada_mosaic_detection_model_v4_fast.pt`
- `lada_vr_mosaic_detection_model_v2_accurate.pt`（VR180 向け YOLO 検出モデル `zelefans-vr-yolo-v2` の実体）

ソースから実行すると Jasna は `model_weights/` を自動解決する（付録 A.1）ため、ここに置くだけでよい。

検出モデルは `model_weights/` にあるファイルから自動発見されるため、4 つ目が無くても他機能は動作する。その場合は `zelefans-vr-yolo-v2` がモデル一覧に現れず、VR180 の検出に使えないだけである。

テストクリップ 2 本は通常リポジトリに同梱され `$WORKSPACE/jasna/assets/`（`test_clip1_1080p.mp4`, `test_clip1_2160p.mp4`）にある。無い場合は upstream リリースから抽出する。

### 6.1 オプション: RIFE フレーム補間モデル (`--frame-gen` 用)

フレームレート倍化 (`--frame-gen {2x,4x}`) を使う場合のみ必要。RIFE 重みは**同梱されない**（非商用条項のため）ので、`scripts/make_rife_torchscript.py` で TorchScript チェックポイントを自分で作成する。

1. Practical-RIFE をクローンし、4.x のモデルパッケージ（**v4.25 で動作確認**）をダウンロードして `<repo>/train_log/` に `RIFE_HDv3.py`, `IFNet_HDv3.py`, `flownet.pkl` が揃う状態にする:
   ```bash
   git clone https://github.com/hzwer/Practical-RIFE
   # README に従いモデルパッケージを Practical-RIFE/train_log/ にダウンロード・展開
   ```
2. プロジェクトの venv で TorchScript に変換する（`$WORKSPACE/jasna` で実行）:
   ```bash
   .venv/bin/python scripts/make_rife_torchscript.py \
       --rife-repo /path/to/Practical-RIFE \
       --output model_weights/rife.pth --validate
   ```

**ソースから実行**する場合は `model_weights/rife.pth` を自動で参照する（他のウェイトと同じリゾルバ経由。または `JASNA_MODEL_WEIGHTS_DIR` をそれを含むフォルダに向ける）。詳細手順: [frame_generation.md](frame_generation.md)。

### 6.2 オプション: FlashVSR 二次復元（実験的）

FlashVSR（`--secondary-restoration flashvsr` / `flashvsr-inline`）は別リポジトリのチェックアウトと専用 venv を必要とする。チェックアウトには fork [`sh202603/FlashVSR_plus`](https://github.com/sh202603/FlashVSR_plus) を使う（inline に要る tiny-long の修正と `--flashvsr-accel` の高速化を含み、`uv sync` で依存が揃う）。セットアップ手順は [flashvsr.md](flashvsr.md) を参照。

---

### 6.3 オプション: SeedVR2 一次復元（実験的）

SeedVR2（`--restoration-model-name seedvr2`）は別リポジトリ `ComfyUI-SeedVR2_VideoUpscaler` のチェックアウトと専用 venv、`model_weights/` への LoRA（約 90 MB）配置を必要とする。パッチは不要で、jasna の venv には何も入らない。セットアップ手順は [seedvr2.md](seedvr2.md) を参照。

---

### 6.4 オプション: SwiftVR 二次復元と SwiftVR distill（実験的）

SwiftVR（`--secondary-restoration swiftvr` / `swiftvr-inline`。`--ltx-crop-vsr` の `swiftvr` バックエンドも同じ）は fork [`sh202603/SwiftVR`](https://github.com/sh202603/SwiftVR)（`modi` ブランチ）のチェックアウトと `uv sync` の専用 venv、約 20 GB のチェックポイントを必要とする。小さな蒸留モデル（`--secondary-restoration swiftvr-distill`。`--ltx-crop-vsr` の `swiftvr-distill` バックエンドも同じ）は [okatti/swiftvr-distill](https://huggingface.co/okatti/swiftvr-distill) の `swiftvr-distill.pt` を `model_weights/` に置くだけでよい。セットアップ手順は [swiftvr.md](swiftvr.md) を参照。

---

## 7. ソースから実行 / 動作確認

1〜6節が完了したら、venv 内でソースチェックアウトから直接 Jasna を実行します:

```bash
cd "$WORKSPACE/jasna"
python -m jasna --version    # -> 0.11.2+modi
python -m jasna --help
python -m jasna --input assets/test_clip1_1080p.mp4 --output /tmp/out.mp4   # 短いクリップを処理
python -m jasna              # GUI を起動（引数なし）
```

> **GUI の起動は `python -m jasna`（引数なし）です。** console script の `jasna` は `jasna.main:main` に直結しており、引数なしでも GUI ディスパッチ（`jasna/__main__.py`）を通りません。frozen ビルドの引数なし起動だけが GUI に入ります。CLI も `python -m jasna` で起動するのを勧めます。console script の `jasna` は `__main__.py` が先に行う設定（`CUDA_MODULE_LOADING=LAZY`、マルチプロセスの子プロセスを止めるガード）を通りません。

`model_weights/` は自動解決され（付録 A.1）、`ffmpeg`/`ffprobe`（v8）はシステムの `PATH` から使われます。インストールパスは ASCII のみ（起動時に強制）。

> **最初の処理実行は遅い。** 初回使用時に GPU 向け TensorRT エンジンがコンパイルされます（15〜60 分）。`model_weights/` 内のウェイト隣（`<モデル名>_sub_engines/` 等）にキャッシュされ、以降は再利用されます。v0.9.1 ベース以降のエンジンキャッシュはそのまま有効です（v0.11.1 までエンジン周りのコードは変わりましたが、TensorRT は 10.16.1.11 のままでエンジンのファイル名も同じです）。`--max-clip-size` の変更で再コンパイルされることもなくなりました。

---

## 8. パッケージング / frozen ビルド

現時点で、このフォークから **Linux 向けのパッケージ化された frozen バイナリを生成する公開手段はありません。** upstream のパッケージングツールはプライベートな submodule（`jasna/protection`）にあり、この公開フォークには含まれません。フォーク独自の実験的 Nuitka スクリプト（`scripts/build_nuitka.py`、[frozen_build.md](frozen_build.md)）は Windows 専用です。

したがって公開でサポートされる経路は**ソースから実行**（7節）です。パッケージ済みバイナリが必要なら upstream Kruk2/jasna の公式リリースを使ってください。

---

## トラブルシューティング

### PyAV / FFmpeg

- **エンコード開始直後に `ValueError: hevc_nvenc did not accept encoder option(s): ['lookahead_level']`**
  av wheel が古い nv-codec-headers の FFmpeg にリンクされている。公式 PyPI wheel（18.1 以降は対応 FFmpeg を同梱）では起きず、暫定期の自前ビルド wheel の残存か、distro FFmpeg に対するソースビルドが原因。`uv pip install --reinstall "av>=18.1"` で解消する。

- **エンコーダ/デコーダ初期化時に `current_ctx` 関連の `TypeError` 等で失敗する**
  av が 18.0.0 のまま（PyPI の旧リリース、またはバージョン表記が同じ `18.0.0` の暫定自前ビルド wheel）。`uv pip install "av>=18.1"` で入れ直す（4節）。

- **Jasna が起動拒否: ffprobe のバージョン違い**
  起動チェックは `ffprobe` の**メジャーバージョン 8** を要求します。`apt` の ffmpeg 8 を導入する（1.1節）。

### 環境 / Python

- **GUI 起動時に `ModuleNotFoundError: No module named 'tkinter'`**
  GUI には Tcl/Tk が必要で、これは `python3.13-tk` パッケージです（注: `python3-tk` は別のインタプリタ向け）。`sudo apt-get install -y python3.13-tk`。CLI は無くても動作します。

- **venv の Python が 3.14**
  `uv venv --python /usr/bin/python3.13 .venv` とパス指定で作り直す（3節）。

### jasna インストール

- **`uv pip install -e .[dev,nvidia]` が `python_vali==4.8.8` を解決できずに失敗**。4.8.8 は PyPI に無い。5.2 節で VALI フォーク wheel をビルドし、`--find-links "$WORKSPACE/vali/dist"` を付けて再実行（5.3 節）。
- **VALI のビルドが `ModuleNotFoundError: No module named 'pkg_resources'` で失敗**。setuptools 81 以降が入っている。`uv pip install "setuptools<81"` で戻す（5.1 節）。
- **`uv pip install -e .[dev,nvidia]` が `no version of torch==2.12.1+cu130` で失敗**。`--extra-index-url https://download.pytorch.org/whl/cu130` を追加。
- **`torch-tensorrt==2.12.0+cu130 ... unsatisfiable`**。`--index-strategy unsafe-best-match --prerelease=allow` を追加（5節）。

### ランタイム

- **Jasna が `.pth`/`.onnx`/`.pt` で `FileNotFoundError`**
  `model_weights/` のファイル欠落、またはリゾルバが別の場所を見ている。6 節のウェイトを `$WORKSPACE/jasna/model_weights/` に置くか、`JASNA_MODEL_WEIGHTS_DIR` をそれらが入ったフォルダに設定する。

- **BasicVSR++ の読み込みが `Weights only load failed ... HistoryBuffer` で失敗する**
  mmengine パッチが当たっていないか、upstream 版（`weights_only=True`）が当たっている。5.4 節のとおり mmengine を素に戻して同梱の diff を当てる。

- **RTX Super-Res で `IRuntime::deserializeCudaEngine ... Version tag does not match`**
  `nvidia-vfx` 0.1.0.1 以前に限る症状。それらの wheel は TensorRT 10.9 を同梱し、jasna の TensorRT 10.16 と soname `libnvinfer.so.10` を共有するため、nvvfx 側が先にロードされるとエンジンを読めなくなる。固定版の 0.2.0.0 は TensorRT を同梱せず、衝突を避ける先読みガードも残っている（付録 A.3）。この症状が出たら `uv pip install nvidia-vfx==0.2.0.0`。

- **`--frame-gen-backend rtx` が「RTX Video Frame Generation is not available in nvidia-vfx 0.1.0.1」と言う**
  venv が 0.2.0.0 固定（5 節）より前のもの。`uv pip install nvidia-vfx==0.2.0.0`。effect には加えて RTX 40 シリーズ以降の GPU とドライバ 570.190 以降 / 580.82 以降 / 590.44 以降が必要。

- **`--fp8-recon` と RTX Super-Res の併用で `Unable to load any of {libcudnn_graph.so.9.7.1, ...}` で中断**
  nvvfx が `LD_LIBRARY_PATH` 先頭に追記する同梱ディレクトリに、不完全な cuDNN 9.7 ディスパッチャだけが置かれているため。修正は本ブランチで適用済みです（`jasna/restorer/fp8_upsample.py` が `import cudnn` の前に torch 同梱の完全な cuDNN を優先させる）。

- **致命的 CUDA エラーの後、再実行が CUDA 初期化エラー（MPS 環境では `Error 807`）で失敗する**
  クラッシュ後にメインプロセスが残留して CUDA コンテキストを握り続けることがあります。特に MPS 運用では残留クライアントが新規初期化を塞ぎます。残留した jasna プロセスを kill してから再実行してください。

---

## 付録 A: 本セットアップが依存するブランチ側の変更

以下はブランチのソースに適用済みの変更です。独立した `.patch` ファイルは同梱しません（同梱する 2 つのパッチ、mmengine 用と FlashVSR_plus 用は別パッケージに当てるものです。5.4 節と [flashvsr.md](flashvsr.md)）。ブランチの変更の全体は [changes_vs_upstream.md](changes_vs_upstream.md) にあります。

### A.1 `model_weights/` ディレクトリの自動解決

`jasna/model_weights_resolver.py` が `model_weights/` を次の順で探します: 環境変数 `JASNA_MODEL_WEIGHTS_DIR`、実行ファイルの隣（frozen ビルドのみ）、カレントディレクトリ、`jasna` パッケージの隣（ソースツリー）。CLI、検出モデルレジストリ、エンジンパス、GUI はハードコードの `Path("model_weights")` ではなくこれを経由するので、CLI は任意のディレクトリから実行でき、6 節はファイルを置くだけで済みます。

### A.2 Linux/X11 のモーダルダイアログ

About ダイアログと初回ウィザードは、upstream の `grab_modal()` を 250 ms 遅らせて呼びます（`jasna/gui/app.py`、`jasna/gui/wizard.py`）。ウィジェットの描画より先にモーダルグラブを掛けると、一部の X11 ウィンドウマネージャでリサイズ不可の `CTkToplevel` が空のまま残るためです。upstream の他のダイアログはグラブ前に `wait_visibility()` を呼んでおり、変更していません。

### A.3 RTX Super Res 復元器の TensorRT ランタイム先読み

`jasna/restorer/rtx_superres_secondary_restorer.py` の `_preload_tensorrt_runtime()` は、`nvvfx` の import より前に `tensorrt_libs` の `libnvinfer.so.10` を `RTLD_GLOBAL` でロードします。nvvfx 0.1.0.1 が同梱していた TensorRT 10.9 への対策でした（jasna の 10.16 と同じ soname のため先にロードされた方が勝ち、10.16 でビルドしたエンジンを逆シリアライズできなくなる）。固定版 0.2.0.0 は TensorRT を同梱しないため今は no-op で、upstream も同じ関数を持ちます（`6545b78`）。0.1.0.1 が残る venv のために残しています。

---

## 付録 B: 本セットアップの履歴

ベースごとのソース実行手順の変化と、本ガイドを検証した時点（新しい順）。不要になった手順は要約だけを残し、全文は本ファイルの git 履歴にあります。

- **v0.11.2 ベース（2026-10-18 検証、RTX 5080、Ubuntu 26.04）**: ピンの変更は `comfy-kitchen` 0.2.36 → 0.2.38 だけ（カーネルの変更は SM 8.9 未満のみ。既存 venv なら `uv pip install comfy-kitchen==0.2.38` で足りる）。upstream は LTX の less-overlap モードと HDD インストール警告を追加。VALI の再ビルド（`0e5c01e` のまま）もエンジンの再コンパイルも不要。検証（Linux、RTX 5080、2026-10-18）: モデル重みのある checkout での全テストを、同じ deselect と timeout で素の upstream `b623466f` と比較し、失敗は upstream と同じ箇所だけ（protection、gcc-15 での native LTX ビルド、表示環境、ultralytics）。10 秒のテストクリップで modi 機能の CLI 実走（BasicVSR++ の FP8 あり/なし、torchcodec、fMP4、RTX Super Res、SwiftVR distill、SwiftVR offline、SeedVR2、RIFE と RTX のフレーム生成、`--segments`、30 秒プローブ素材での `--ltx-crop-vsr swiftvr-distill`、両方の hop での `--ltx-trial`、LTX + frame-gen の拒否）。HDD 判定は本機（NVMe）で False。`comfy-kitchen` 0.2.38 を既存 venv に入れ、kitchen 関連テスト合格。既知の罠: `model_weights/` の**無い** checkout で全テストを流すと `tests/test_fp8_upsample.py::test_fp8_forward_parity_small` で pytest プロセスごと abort する（`cudnn.create_handle()` が abort、exit 134）。v0.11.1 ベースでも同じで新規ではなく、単体では再現しない。
- **v0.11.1 ベース（2026-10-16 検証、RTX 5080、Ubuntu 26.04）**: upstream が torch 2.12.1 / torchvision 0.27.1 を固定し、`comfy-kitchen==0.2.36` と `ultralytics>=8.4.96`（YOLO export に onnxruntime も onnxslim も要らなくなり、5.5 節は注記だけになった）を追加、`av==18.1.0` を厳密固定、デコードバックエンドの選択を `JASNA_DECODE_BACKEND` だけにした（in-code の `DECODE_BACKEND` トグルは廃止）。VALI フォーク wheel は Kruk2 氏の `0e5c01e`（4.8.8 + ASF/VC-1 タイムスタンプ修正 + decoder-open の GIL 解放）で再ビルド。検証: 同じ deselect と timeout で素の upstream と比較した full pytest（残る失敗はすべて upstream でも失敗）、modi 機能の e2e CLI 実走（native / torchcodec バックエンド、FP8、フレーム生成、SeedVR2、FlashVSR inline、SwiftVR inline / オフライン / distill、`--ltx-crop-vsr`、segments、streaming）、両デコードパスで VALI フォーク wheel が実効。同日、`python_vali==4.8.8` が PyPI に無い（素の venv では `[nvidia]` が解決できない）ことを受けて VALI を必須と位置づけ、5 節を「ビルド用パッケージ（setuptools<81 を含む）→ VALI wheel のビルド → `--find-links` 付きの jasna インストール → mmengine パッチ」の順に組み替えた。mmengine パッチは、upstream が `7c64105` で `weights_only=True` に変えた版だとソース実行で `v1.2.pth` が読めないことを確認し、フォーク側で `False` のまま同梱することにした（4 ファイル分の説明を追加）。新しい 5 節は、空の venv と新規 clone の vali から通しで確認した（`[nvidia]` が `--find-links` で解決、`DecodeSingleSurfaceAsyncDetailed` が `True`、mmengine パッチ後に `load_model()` が `v1.2.pth` を読める）。
- **nvidia-vfx 0.2.0.0 固定（2026-10-03）**: `--frame-gen-backend rtx` の `VideoFrameGeneration` effect のため、ブランチが `nvidia-vfx==0.2.0.0` を固定（upstream は無 pin）。0.2.0.0 は TensorRT を同梱しなくなり、付録 A.3 の `libnvinfer.so.10` 衝突は構造的に消滅、Linux wheel は約 1.1 GB から 443 MB に縮小。それ以前の venv には 0.1.0.1 が入っている: `uv pip install nvidia-vfx==0.2.0.0`。
- **PyAV を PyPI から（2026-08-24）**: av 18.1.0 が current_ctx API とエンコーダが使う CUDA stream 明示指定を含んで PyPI に公開され、PyAV main（コミット `f6f0a5e`）を BtbN FFmpeg 8.1 shared ビルドにリンクして auditwheel で同梱する暫定手順が終了。暫定 wheel は `18.0.0` を報告するため、5 節のインストールで置き換わる（すぐ替えるなら `uv pip install "av>=18.1"`）。公式 wheel で Linux 実機検証: メディア層テストと、NVDEC デコード・NVENC エンコードを含むフルパイプラインの e2e。
- **v0.10.0 ベース（2026-08-13）**: 環境変数 `JASNA_DECODE_BACKEND`（`auto` / `vali` / `pyav-hw` / `pyav-sw`）。`python_vali` のピンは 4.8.8（フォークコミット `f4a67f8`）。
- **v0.9.1 ベース（2026-07-30 検証）**: VALI NVDEC デコードバックエンドが任意の自前ビルド wheel として復活（現 5.2 節。当時はフォークコミット `3ad0d54` = 4.8.7）。エンコーダが `CudaContext(cuda_stream=...)` で CUDA stream を渡すようになり、av 18.1.0 までは上記の PyAV main wheel が必須だった。TensorRT エンジンは一度全再コンパイル。
- **d7a99bd ベース（2026-07-23 検証）**: upstream の per-language docs、ロケール分割、composition root。セットアップ手順の変化なし。
- **v0.8.1 ベース（2026-07-19 検証）**: GPU スタックが `nvidia` / `amd` extras へ（`.[dev,nvidia]` の由来）、`requires-python` は `>=3.12` に緩和、Linux ドライバ最小 580。
- **v0.8.0 ベース（2026-07-18 検証）**: upstream がメディア層を PyAV（NVDEC/NVENC）へ移し、`python_vali` / `PyNvVideoCodec` をランタイムから撤去。それらのネイティブビルドとともに、旧ガイドの前提だった CUDA Toolkit、cmake / ninja、`libav*-dev`、`FFMPEG_DIR` prefix、`setuptools<80` 固定、`mkvmerge` が不要になった（CUDA Toolkit、cmake / ninja、`libav*-dev` は、VALI フォークのビルドが前提になった現在は再び必要。5.2 節）。付録 A.2 の Linux GUI 修正はこの時期のもの（`wait_visibility()` の扱いは upstream、250 ms の遅延はブランチ側）。
