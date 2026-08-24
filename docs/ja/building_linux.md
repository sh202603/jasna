# Building Jasna for Linux

Linux で Jasna のセットアップを行い、**ソースから実行**する手順です。

> **本ガイドは `v0.9.1+modi` ブランチの手順です。** GPU スタック（**torch 2.12.0+cu130 / torchvision 0.27.0+cu130 / torch-tensorrt 2.12.0+cu130 / tensorrt 10.16.1.11**）は v0.7.2 期から変わらず、依存ピンは本ブランチの `pyproject.toml` に適用済みです。TensorRT が 10.16 系に留まるのは、`torch-tensorrt==2.12.0` が `tensorrt>=10.16.1,<10.17.0` を要求するためです（torch-tensorrt は TensorRT 11 に未対応）。

> **v0.8.0 でビルド手順は大幅に簡素化されました。** upstream v0.8.0 でメディア層が PyAV（NVDEC/NVENC）へ移行し、`python_vali` / `PyNvVideoCodec` のネイティブビルドが丸ごと不要になりました。これに伴い、旧ガイドの前提だった CUDA Toolkit、cmake / ninja、ffmpeg dev パッケージ（`libav*-dev`）、`FFMPEG_DIR` prefix、`setuptools<80` の固定、`mkvmerge` はすべて不要です。残る特殊手順は **PyAV wheel の自前ビルド**（4 節）だけです。v0.9.1 ベースからは**任意**で VALI デコードバックエンドを再びビルドできます（5.3 節）が、必須ではありません。
>
> **このブランチの主な追加機能:** RIFE による 2x/4x フレーム生成（[frame_generation.md](frame_generation.md)）、torchcodec バックエンド（[torchcodec_backend.md](torchcodec_backend.md)）、cuDNN FP8 復元（[fp8_recon.md](fp8_recon.md)）、SeedVR2 一次復元（[seedvr2.md](seedvr2.md)）、FlashVSR 二次復元（[flashvsr.md](flashvsr.md)）。v0.7.2+modi にあった AV1 / 8bit / BT.601 と BT.2020 出力は upstream v0.8.0 に吸収されました。全差分は [changes_vs_upstream.md](changes_vs_upstream.md) を参照してください。

> **パッケージングについて:** この公開フォークは Jasna を**ソースから実行**します。Linux 向けに frozen バイナリを生成する公開手段はありません（同梱の実験的 Nuitka スクリプト `scripts/build_nuitka.py` は Windows 向けで、しかも v0.8.0 のメディア層移行に未追従です）。パッケージ済みバイナリが必要なら upstream Kruk2/jasna の公式リリースを使ってください。

> 本ガイドは **Ubuntu 26.04 LTS** + **RTX 5080** で **upstream `v0.9.1` タグ（`a7cdaf8`）ベースの `0.9.1+modi` を検証済み**です（2026-07-30、full pytest（e2e 込み）の失敗集合が同一マシンで測った素の v0.9.1 ベースラインと完全一致、CLI スモークは native / torchcodec+fp8-recon / fmp4 / rfdetr-v6 / frame-gen / segments / streaming / flashvsr-inline、GUI 起動スモーク、5.3 節の VALI fork wheel のビルドと実効確認込み。過去ベース: d7a99bd 2026-07-23、v0.8.1 2026-07-19、v0.8.0 2026-07-18）。他のディストリでも動作しますが、パッケージ名や ffmpeg の導入手順は異なります。

---

## 1. 前提ソフトウェア

| カテゴリ | 要件 | 入手元 | 備考 |
|---|---|---|---|
| OS | Ubuntu 26.04 x64（または同等） | n/a | 26.04 で検証。`apt` に ffmpeg 8 がある |
| ビルドツール | `build-essential`, `pkg-config` | apt | 任意の VALI バックエンドのビルド（5.3節）にのみ使用 |
| Python | **3.13 + 3.13-dev + 3.13-tk** | apt (`python3.13 python3.13-dev python3.13-tk`) | v0.8.1 で `requires-python` は `>=3.12` に緩和されたが、本ガイドの検証は 3.13 のみ（Ubuntu 26.04 の既定 `python3` = 3.14 は cu130 GPU スタックとの組み合わせが未検証）。`python3.13-dev` は任意の VALI wheel のビルド（5.3節）に、`python3.13-tk` は GUI の実行に必要 |
| uv | 最新 | [astral.sh/uv](https://docs.astral.sh/uv/) | venv を管理 |
| NVIDIA ドライバ | **580+** | ディストリ / NVIDIA | 起動チェックが Linux では 580 以上を要求。GPU は compute capability 7.5+ |
| ffmpeg / ffprobe | **v8**（実行時 CLI） | apt | 起動チェックが `ffprobe` のメジャーバージョン 8 を要求。`ffmpeg` CLI は HLS ストリーミング等で使用 |

CUDA Toolkit は基本セットアップでは不要になりました。torch / tensorrt の pip wheel が CUDA ランタイムを同梱しており、自前でコンパイルするネイティブ拡張ももうありません（av 18.1.0 以降、PyAV wheel は PyPI から入ります。4節）。拡張をコンパイルするのは任意の VALI バックエンドのビルドだけで、その追加前提（CUDA Toolkit を含む）は 5.3 節に記載しています。

### 1.1 システムパッケージのインストール

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential pkg-config \
  python3.13 python3.13-dev python3.13-tk \
  ffmpeg
```

`uv` が無ければ導入します:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> ⚠ **apt の ffmpeg は CLI としてだけ使います。** PyPI の av wheel は自前の FFmpeg を同梱しており、distro のライブラリには触れません（4節）。旧ガイドで必要だった `libav*-dev` は基本セットアップでは不要で、任意の VALI バックエンドをビルドする場合にだけ必要になります（5.3 節）。

---

## 2. リポジトリのクローン

クローンするのは `jasna` だけです。`vali` / `PyNvVideoCodec` は v0.8.0 のランタイムからは使われないため、チェックアウト不要になりました。本ガイドでは作業ルートを `$WORKSPACE` と呼びます。

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

旧ガイドにあったビルドツールの先入れ（`cmake ninja scikit-build "setuptools<80" wheel numpy`）は、対象だった `vali` / `PyNvVideoCodec` のビルドごと廃止されたため不要です。

---

## 4. PyAV wheel（av 18.1.0 以降は PyPI で解決）

GPU パスは PyAV の **current_ctx API** に加え、v0.9.1 ベースからはエンコーダが `CudaContext(cuda_stream=...)` で渡す **CUDA stream 明示指定**を使います。**av 18.1.0（PyPI、2026-08 公開）は両 API を含む最初のリリース**であり、自前ビルドは不要になりました。`pyproject.toml` の `av>=18.1,<19` が 5 節のインストールで解決されます。

PyPI のバイナリ wheel は十分新しい nv-codec-headers でビルドされた FFmpeg 8 を同梱しており（執筆時点で 8.1.2、`hevc_nvenc` は `lookahead_level` を受理）、旧自前ビルド手順で問題だった distro FFmpeg へのリンクの罠は該当しません。公式 18.1.0 wheel で Linux 実機検証済み（2026-08-24）: NVDEC デコード・NVENC エンコードを含むメディア層テストとフルパイプラインの e2e テストが合格。

> **既存 venv の移行:** 暫定の自前ビルド wheel はバージョンが `18.0.0` で、`av>=18.1` を満たさなくなったため、次回の 5 節のインストールで公式 wheel に自動で置き換わります。すぐ切り替えるだけなら `uv pip install "av>=18.1"` を一度実行してください。他に作業はありません。

本節が以前記載していた暫定手順（PyAV upstream main のコミット `f6f0a5e` を BtbN FFmpeg 8.1 shared ビルドにリンクして wheel をビルドし、auditwheel でライブラリを同梱）は、再び必要になった場合に備え本ガイドの git 履歴に残っています。

---

## 5. jasna 本体のインストール

v0.8.1 で GPU スタックは extras に分割されました（`nvidia` = NVIDIA スタック、`amd` = ROCm 用）。NVIDIA ビルドでは `nvidia` extra が `torch==2.12.0+cu130` / `torchvision==0.27.0+cu130` / `torch-tensorrt==2.12.0` / `nvidia-vfx` を入れますが、これらは既定の PyPI にありません。uv を PyTorch cu130 インデックスに向け、フラグを 2 つ追加します:

```bash
cd "$WORKSPACE/jasna"
uv pip install -e .[dev,nvidia] \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

各フラグの理由:

- `--extra-index-url https://download.pytorch.org/whl/cu130`：`torch+cu130` / `torch-tensorrt+cu130` wheel の入手元。
- `--index-strategy unsafe-best-match`：`torch-tensorrt==2.12.0` をインデックスの local-version リリース `2.12.0+cu130` で満たせるようにする。
- `--prerelease=allow`：推移依存（`nvidia-cuda-runtime-cu13`）がプレリリース。

`[dev]` extra は `nuitka>=2.4`, `pytest`, `pytest-cov`, `scikit-build`, `cmake`, `ninja` を入れます。`[nvidia]` extra は GPU スタック（torch cu130 / TensorRT / torch-tensorrt / nvidia-vfx。v0.8.1 で必須依存から分割）を入れます — **指定を忘れると torch が入らず起動しません**。

**オプション: torchcodec バックエンド。** 実験的な torchcodec のデコード/エンコード経路（`--video-backend torchcodec`/`auto`）を使う場合は、`torchcodec` extra を追加し、同じフラグで `.[dev,nvidia,torchcodec]` を入れます:

```bash
uv pip install -e .[dev,nvidia,torchcodec] \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

これで `torchcodec>=0.15.0` が入ります。通常のセットアップには不要で、既定の `native` バックエンド（PyAV）は torchcodec なしで動作します。詳細は [torchcodec_backend.md](torchcodec_backend.md)。

**補足: FP8 復元バックエンドに追加のインストール手順は不要です。** 依存 `nvidia-cudnn-frontend` は `pyproject.toml` の通常依存で、上記コマンドで一緒に入ります。cuDNN ランタイム（9.17 以上）は torch cu130 wheel に同梱済みです。機能自体は実行時 opt-in（`--fp8-recon`、FP8 対応 GPU sm89 以上が必要）で、使えない環境では TensorRT エンジンにフォールバックします。詳細は [fp8_recon.md](fp8_recon.md)。

**任意: TensorRT-RTX フレーバー。** `nvidia` の代わりに `nvidia-rtx` extra を入れると、TensorRT スタック全体が TensorRT-RTX（JIT コンパイル）に切り替わり、エンジンビルドが分単位から秒単位になります。
`torch-tensorrt` と `torch-tensorrt-rtx` は同じ `torch_tensorrt` パッケージを提供するため、両 extra は 1 つの venv に共存できません。専用 venv を使ってください。

```bash
uv pip install -e .[dev,nvidia-rtx] \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

フレーバーはインストール済み wheel から自動判別されます。
エンジンは `.rtx` タグ付きの名前でキャッシュされるため、両フレーバーで 1 つの `model_weights` ディレクトリを共有できます。
mmengine パッチ（§5.1）はこの venv にも必要です。
詳細と実測値は [tensorrt_rtx.md](tensorrt_rtx.md)。

### 5.1 mmengine パッチの適用 (torch 2.6+ 対応)

`mmengine.runner.checkpoint` 内の `torch.load` 呼び出しに `weights_only=False` を追加します。torch 2.6 以降は既定が `weights_only=True` になり、既存の `.pth` チェックポイント読み込みが壊れるためです。diff は `patches/` に同梱。

```bash
cd "$WORKSPACE/jasna"
patch -p1 -d .venv/lib/python3.13/site-packages \
    < patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff

# 確認
grep -n "weights_only=False" .venv/lib/python3.13/site-packages/mmengine/runner/checkpoint.py
```

> `uv pip install -e .[dev,nvidia]` を再実行すると `mmengine` が再インストールされ、このパッチが消えます。その後は再適用してください。

### 5.2 ONNX パッケージ（YOLO 検出モデル用）

**YOLO 系検出モデル（lada-yolo-\* と zelefans-vr-yolo-v2）**を使う場合、ultralytics は TensorRT エンジンをビルドする前にモデルを ONNX へエクスポートします。これには自動では入らない 3 パッケージが必要です。（RF-DETR モデルは不要。事前ビルド済み `.onnx` を TensorRT が直接パースするため）

```bash
uv pip install onnx onnxslim onnxruntime
```

ソースから実行する場合、ultralytics が初回エクスポート時に未導入ならこれらを自動ダウンロードしますが、事前に入れておくと初回 YOLO 実行時の待ちを回避できます。

---

### 5.3 オプション: VALI NVDEC デコードバックエンドのビルド（`python_vali` フォーク）

v0.9.1 ベースの native デコードは、まず VALI NVDEC デコーダを試し、使えなければ PyAV へエスカレーションします（in-code トグル `DECODE_BACKEND`、既定 `auto`）。5 節で入る PyPI の素の `python_vali` wheel にはフォーク専用 API `DecodeSingleSurfaceAsyncDetailed` がないため、リーダーは警告を出して毎回 PyAV にフォールバックします。本節なしでもすべて動作します。実際に VALI デコード経路（と corrupt packet 耐性）を使いたい場合にだけ、フォーク wheel をビルドしてください。

1 節に対する追加の前提:

- **CUDA Toolkit 13.x**（`/usr/local/cuda-13.3`）。distro の `/usr/bin/nvcc` 単体では CMake の `FindCUDAToolkit` を満たせないため、下のコマンドで toolkit パスを明示します。
- **FFmpeg 8 の dev パッケージ**: `sudo apt-get install -y libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev`。av wheel（自前の FFmpeg を同梱）と違い、vali のビルドは distro FFmpeg にリンクし、wheel には何も同梱しません。VALI が FFmpeg を使うのは demux/parse だけなので、av 側の FFmpeg に課される nv-codec-headers の条件はここでは当たりません。
- cmake、ninja、scikit-build、`pkg_resources` 入りの setuptools: いずれも 5 節の venv に導入済みです。

```bash
cd "$WORKSPACE"
git clone https://codeberg.org/Kruk2/vali.git       # 既存 checkout があれば省略
cd vali
git submodule update --init --recursive

CUDACXX=/usr/local/cuda-13.3/bin/nvcc \
CUDAToolkit_ROOT=/usr/local/cuda-13.3 \
CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc" \
VIRTUAL_ENV="$WORKSPACE/jasna/.venv" \
  uv build --wheel --no-build-isolation
uv pip install --force-reinstall dist/python_vali-4.8.7-*.whl
```

`--no-build-isolation` が必要なのは、vali の `setup.py` が `pkg_resources` を import しており、隔離ビルド環境にはそれが無いためです。`--force-reinstall` が必要なのは、wheel のバージョンが置き換え先の PyPI パッケージと同じ（4.8.7）ためです。

確認:

```bash
python -c "import python_vali as v; print(hasattr(v.PyDecoder, 'DecodeSingleSurfaceAsyncDetailed'))"   # -> True
# jasna を --log-level info で実行すると "Using VALI NVDEC decoder for <file>" が出ます
```

> フォークコミット `3ad0d54`（= 4.8.7 ピン、2026-07-30、RTX 5080）で検証済み: パイプラインの 2 つのデコードパスが両方 VALI を通り、同梱テストクリップの出力は PyAV デコード時と md5 完全一致、av wheel および torchcodec バックエンドとの共存も確認（この 2 つは auditwheel でマングル/同梱した FFmpeg を持ち、vali は distro の FFmpeg を ldconfig 経由で解決する）。

---

## 6. モデルウェイトとアセットの配置

`$WORKSPACE/jasna/model_weights/` に以下 4 ファイルを置く:

- `lada_mosaic_restoration_model_generic_v1.2.pth`
- `rfdetr-v5.onnx`
- `lada_mosaic_detection_model_v4_fast.pt`
- `lada_vr_mosaic_detection_model_v2_accurate.pt`（v0.8.0 で追加。VR180 向け YOLO 検出モデル `zelefans-vr-yolo-v2` の実体）

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

FlashVSR（`--secondary-restoration flashvsr` / `flashvsr-inline`）は別リポジトリのチェックアウトと専用 venv を必要とし、inline 用にはチェックアウトへの同梱パッチ適用も要る。セットアップ手順は [flashvsr.md](flashvsr.md) を参照。

---

### 6.3 オプション: SeedVR2 一次復元（実験的）

SeedVR2（`--restoration-model-name seedvr2`）は別リポジトリ `ComfyUI-SeedVR2_VideoUpscaler` のチェックアウトと専用 venv、`model_weights/` への LoRA（約 90 MB）配置を必要とする。パッチは不要で、jasna の venv には何も入らない。セットアップ手順は [seedvr2.md](seedvr2.md) を参照。

---

## 7. ソースから実行 / 動作確認

1〜6節が完了したら、venv 内でソースチェックアウトから直接 Jasna を実行します:

```bash
cd "$WORKSPACE/jasna"
python -m jasna --version    # -> 0.9.1+modi
python -m jasna --help
jasna --input assets/test_clip1_1080p.mp4 --output /tmp/out.mp4   # 短いクリップを処理
python -m jasna              # GUI を起動（引数なし）
```

> **GUI の起動は `python -m jasna`（引数なし）です。** console script の `jasna` は `jasna.main:main` に直結しており、引数なしでも GUI ディスパッチ（`jasna/__main__.py`）を通りません。frozen ビルドの引数なし起動だけが GUI に入ります。

`model_weights/` は自動解決され（付録 A.1）、`ffmpeg`/`ffprobe`（v8）はシステムの `PATH` から使われます。`mkvmerge` は不要になりました。インストールパスは ASCII のみ（起動時に強制）。

> **最初の処理実行は遅い。** 初回使用時に GPU 向け TensorRT エンジンがコンパイルされます（15〜60 分）。`model_weights/` 内のウェイト隣（`<モデル名>_sub_engines/` 等）にキャッシュされ、以降は再利用されます。v0.7.2 から移行した場合は、既定 max-clip-size の変更（90）に伴い一部エンジンが再コンパイルされます（loop_body 系のキャッシュは名前が変わらないため再利用されます）。

---

## 8. パッケージング / frozen ビルド

現時点で、このフォークから **Linux 向けのパッケージ化された frozen バイナリを生成する公開手段はありません。** upstream のパッケージングツールはプライベートな submodule（`jasna/protection`）にあり、この公開フォークには含まれません。フォーク同梱の実験的 Nuitka スクリプト（`scripts/build_nuitka.py`、[frozen_build.md](frozen_build.md)）は Windows 向けで、v0.8.0 のメディア層移行に未追従です（旧 `python_vali` / `PyNvVideoCodec` の DLL 同梱を前提としたまま）。

したがって公開でサポートされる経路は**ソースから実行**（7節）です。パッケージ済みバイナリが必要なら upstream Kruk2/jasna の公式リリースを使ってください。

---

## トラブルシューティング

### PyAV / FFmpeg

- **エンコード開始直後に `ValueError: hevc_nvenc did not accept encoder option(s): ['lookahead_level']`**
  av wheel が古い nv-codec-headers の FFmpeg にリンクされている。公式 PyPI wheel（18.1 以降は対応 FFmpeg を同梱）では起きず、暫定期の自前ビルド wheel の残存か、distro FFmpeg に対するソースビルドが原因。`uv pip install --reinstall "av>=18.1"` で解消する。

- **エンコーダ/デコーダ初期化時に `current_ctx` 関連の `TypeError` 等で失敗する**
  av が 18.0.0 のまま（PyPI の旧リリース、またはバージョン表記が同じ `18.0.0` の暫定自前ビルド wheel）。`uv pip install "av>=18.1"` で入れ直す（4節）。

- **Jasna が起動拒否: ffprobe のバージョン違い**
  起動チェックは `ffprobe` の**メジャーバージョン 8** を要求します。`apt` の ffmpeg 8 を導入する（1.1節）。`mkvmerge` のチェックは v0.8.0 で廃止されました。

### 環境 / Python

- **GUI 起動時に `ModuleNotFoundError: No module named 'tkinter'`**
  GUI には Tcl/Tk が必要で、これは `python3.13-tk` パッケージです（注: `python3-tk` は別のインタプリタ向け）。`sudo apt-get install -y python3.13-tk`。CLI は無くても動作します。

- **venv の Python が 3.14**
  `uv venv --python /usr/bin/python3.13 .venv` とパス指定で作り直す（3節）。

### jasna インストール

- **`uv pip install -e .[dev,nvidia]` が `no version of torch==2.12.0+cu130` で失敗**。`--extra-index-url https://download.pytorch.org/whl/cu130` を追加。
- **`torch-tensorrt==2.12.0+cu130 ... unsatisfiable`**。`--index-strategy unsafe-best-match --prerelease=allow` を追加（5節）。

### ランタイム

- **Jasna が `.pth`/`.onnx`/`.pt` で `FileNotFoundError`**
  `model_weights/` のファイル欠落、またはリゾルバが別の場所を見ている。3 つのウェイトを `$WORKSPACE/jasna/model_weights/`（6節）に置くか、`JASNA_MODEL_WEIGHTS_DIR` をそれらが入ったフォルダに設定する。

- **RTX Super-Res で `IRuntime::deserializeCudaEngine ... Version tag does not match`**
  `nvidia-vfx`（nvvfx）同梱の TensorRT 10.9 と jasna の TensorRT 10.16 が soname `libnvinfer.so.10` を共有するため、nvvfx 側が先にロードされるとエンジンを読めなくなる。修正は本ブランチで適用済みです（付録 B.2。upstream にも同等修正が入り収束）。この症状が出る場合は venv の作り直しを検討。

- **`--fp8-recon` と RTX Super-Res の併用で `Unable to load any of {libcudnn_graph.so.9.7.1, ...}` で中断**
  nvvfx が `LD_LIBRARY_PATH` 先頭に追記する同梱ディレクトリに、不完全な cuDNN 9.7 ディスパッチャだけが置かれているため。修正は本ブランチで適用済みです（`jasna/restorer/fp8_upsample.py` が `import cudnn` の前に torch 同梱の完全な cuDNN を優先させる）。

- **致命的 CUDA エラーの後、再実行が CUDA 初期化エラー（MPS 環境では `Error 807`）で失敗する**
  クラッシュ後にメインプロセスが残留して CUDA コンテキストを握り続けることがあります。特に MPS 運用では残留クライアントが新規初期化を塞ぎます。残留した jasna プロセスを kill してから再実行してください。

---

## 付録 A: 本ブランチのビルド/ランタイム改修の内容

本ブランチのソースへ適用済みの変更です（参照用の記述。独立した `.patch` ファイルは同梱しない）。

### A.1 `model_weights/` ディレクトリの自動解決

`jasna/model_weights_resolver.py` を追加し、`model_weights/` を優先順位で探索: `JASNA_MODEL_WEIGHTS_DIR` 環境変数 → 実行ファイルの隣 → カレントディレクトリ → ソースツリーの隣。`main.py`、`mosaic/detection_registry.py`、`engine_paths.py`、GUI（`gui/processor.py`, `gui/engine_preflight.py`）がハードコードの `Path("model_weights")` ではなく resolver を経由します。これで CLI を任意のディレクトリから実行可能に。

### A.2 DLL ロード補助（torchcodec バックエンドのみ）

v0.8.0 で native 経路（PyAV）の DLL 補助は不要になりました。現在は torchcodec バックエンドのモジュール（`jasna/media/torchcodec_decoder.py` / `torchcodec_encoder.py`）が **Windows のみ** torchcodec パッケージのディレクトリと `CUDA_PATH\bin`（設定時）を DLL 検索パスに登録します。Linux では no-op。

### A.3 BasicVSR++ ベンチの TRT API 修正

`jasna/benchmark/basicvsrpp_restoration.py` を新しい `_preprocess_engine` API に更新し `--benchmark basicvsrpp` を動作させる。クロスプラットフォーム。

---

## 付録 B: 本ブランチの Linux GUI / RTX-VSR 修正の内容

本ブランチのソースへ適用済みの Linux 固有のランタイム修正です。いずれも Linux/X11 で GUI と RTX Super-Res を動作させるために必要で、Windows ビルドには影響しません。

### B.1 モーダルダイアログの空表示 (`jasna/gui/app.py`, `jasna/gui/wizard.py`, `jasna/gui/components.py`)

**問題**: Linux/X11 で customtkinter の `CTkToplevel` ダイアログが完全に空（テキストもボタンも無い暗いウィンドウ）で開く。対象: **About**（`app.py` `_show_about`）、**System Check** / 初回ウィザード（`wizard.py` `FirstRunWizard`）、**プリセット作成**と**確認**ダイアログ（`components.py` `PresetDialog`, `ConfirmDialog`）。いずれも生成直後、子ウィジェット描画前に `grab_set()`（場合により `lift()` / `focus_force()`）を呼ぶ。一部のウィンドウマネージャでは、ウィンドウはマップされるが未描画のまま残り、ダイアログがリサイズ不可のため再描画が走らない。

**修正**: 先に全ての子ウィジェットを生成し、その後 `lift()` / `grab_set()` / `focus_force()` を `self.after(200, …)` / `after(250, …)` でイベントループの後続ティックに遅延させ、内容が描画された後にモーダルグラブを確立する。Windows の挙動は不変（同じ呼び出しが 1 ティック遅れて走るだけ）。

### B.2 RTX Super-Res の TensorRT 版数衝突 (`jasna/restorer/rtx_superres_secondary_restorer.py`)

**問題**: RTX Super-Res を有効にすると、jasna の TensorRT エンジン逆シリアライズ中に `IRuntime::deserializeCudaEngine ... Serialization assertion stdVersionRead == kSERIALIZATION_VERSION failed. Version tag does not match` で中断する。`nvidia-vfx`（nvvfx）パッケージは自前の TensorRT **10.9**（`nvvfx/libs/libnvinfer.so.10`）を同梱し、`nvvfx/_lib_loader.py` で `RTLD_GLOBAL` 読み込みする。一方 jasna のパイプラインエンジンは TensorRT **10.16**（`tensorrt_libs`）でビルドされる。両者は soname `libnvinfer.so.10` を共有し、ELF のシンボル解決は「先にグローバルスコープへ入った方」を使う。nvvfx が jasna の TensorRT ランタイムより先にロードされると、`torch-tensorrt` が nvvfx の古い 10.9 にバインドされ、jasna の新しい 10.16 製エンジンを読めない。

**修正**: RTX Super-Res リストアモジュールの import 時（`nvvfx` の import より前）に実行される `_preload_tensorrt_runtime()` を追加。Linux では `tensorrt_libs` を特定し、その `libnvinfer.so.10` / `libnvinfer_plugin.so.10` を `ctypes.RTLD_GLOBAL` で先読みすることで、TensorRT 10.16 のシンボルを先にグローバルスコープへ入れ、後続の nvvfx のロードも 10.16 に解決させる。Windows では no-op。upstream も後に同等修正を独自実装したため（`6545b78`）、関数名は upstream に合わせて収束済み。
