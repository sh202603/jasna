# Building Jasna for Linux

Linux で Jasna のセットアップを行い、**ソースから実行**する手順です。

> **本ガイドは `v0.8.0+modi` ブランチの手順です。** GPU スタック（**torch 2.12.0+cu130 / torchvision 0.27.0+cu130 / torch-tensorrt 2.12.0+cu130 / tensorrt 10.16.1.11**）は v0.7.2 期から変わらず、依存ピンは本ブランチの `pyproject.toml` に適用済みです。TensorRT が 10.16 系に留まるのは、`torch-tensorrt==2.12.0` が `tensorrt>=10.16.1,<10.17.0` を要求するためです（torch-tensorrt は TensorRT 11 に未対応）。

> **v0.8.0 でビルド手順は大幅に簡素化されました。** upstream v0.8.0 でメディア層が PyAV（NVDEC/NVENC）へ移行し、`python_vali` / `PyNvVideoCodec` のネイティブビルドが丸ごと不要になりました。これに伴い、旧ガイドの前提だった CUDA Toolkit、cmake / ninja、ffmpeg dev パッケージ（`libav*-dev`）、`FFMPEG_DIR` prefix、`setuptools<80` の固定、`mkvmerge` はすべて不要です。残る特殊手順は **PyAV wheel の自前ビルド**（4節。PyAV 18.1.0 が PyPI に公開されるまでの暫定）だけです。
>
> **このブランチの主な追加機能:** RIFE による 2x/4x フレーム生成（[FRAME_GENERATION_ja.md](FRAME_GENERATION_ja.md)）、torchcodec バックエンド（[TORCHCODEC_BACKEND_ja.md](TORCHCODEC_BACKEND_ja.md)）、cuDNN FP8 復元（[FP8_RECON_ja.md](FP8_RECON_ja.md)）、FlashVSR 二次復元（[FLASHVSR_ja.md](FLASHVSR_ja.md)）。v0.7.2+modi にあった AV1 / 8bit / BT.601 と BT.2020 出力は upstream v0.8.0 に吸収されました。全差分は [CHANGES_vs_upstream_ja.md](CHANGES_vs_upstream_ja.md) を参照してください。

> **パッケージングについて:** この公開フォークは Jasna を**ソースから実行**します。Linux 向けに frozen バイナリを生成する公開手段はありません（同梱の実験的 Nuitka スクリプト `scripts/build_nuitka.py` は Windows 向けで、しかも v0.8.0 のメディア層移行に未追従です）。パッケージ済みバイナリが必要なら upstream Kruk2/jasna の公式リリースを使ってください。

> 本ガイドは **Ubuntu 26.04 LTS** + **RTX 5080** + **ドライバ 595.71.05** で検証済みです（2026-07-18、pytest とフルパイプライン実走）。他のディストリでも動作しますが、パッケージ名や ffmpeg の導入手順は異なります。

---

## 1. 前提ソフトウェア

| カテゴリ | 要件 | 入手元 | 備考 |
|---|---|---|---|
| OS | Ubuntu 26.04 x64（または同等） | n/a | 26.04 で検証。`apt` に ffmpeg 8 がある |
| ビルドツール | `build-essential`, `pkg-config` | apt | PyAV wheel のビルド（4節）にのみ使用 |
| Python | **3.13 + 3.13-dev + 3.13-tk** | apt (`python3.13 python3.13-dev python3.13-tk`) | Ubuntu 26.04 の既定 `python3` は 3.14 で、`pyproject.toml` の `requires-python = ">=3.13,<3.14"` が拒否する。`python3.13-dev` は PyAV wheel のビルドに、`python3.13-tk` は GUI の実行に必要 |
| uv | 最新 | [astral.sh/uv](https://docs.astral.sh/uv/) | venv を管理 |
| NVIDIA ドライバ | **580+** | ディストリ / NVIDIA | 起動チェックが Linux では 580 以上を要求。GPU は compute capability 7.5+ |
| ffmpeg / ffprobe | **v8**（実行時 CLI） | apt | 起動チェックが `ffprobe` のメジャーバージョン 8 を要求。`ffmpeg` CLI は HLS ストリーミング等で使用 |

CUDA Toolkit は不要になりました。torch / tensorrt の pip wheel が CUDA ランタイムを同梱しており、自前でコンパイルするネイティブ拡張は PyAV だけで、PyAV は CUDA を使いません（NVDEC/NVENC はリンク先 FFmpeg 側の機能です）。

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

> ⚠ **apt の ffmpeg は CLI としてだけ使います。** PyAV wheel を distro の FFmpeg ライブラリにリンクすると全エンコードが起動直後に失敗します（理由は 4 節）。旧ガイドで必要だった `libav*-dev` はもう入れる必要がありません（入っていても PKG_CONFIG_PATH を正しく設定する限り害はありません）。

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

## 4. PyAV wheel のビルドと導入（暫定手順）

v0.8.0 の GPU パスは PyAV 18.1.0 で入る **current_ctx API**（torch が初期化済みの CUDA コンテキストを NVDEC/NVENC と共有する仕組み）を使いますが、18.1.0 は未公開で、PyPI の av 18.0.0 にはこの API がありません。そのため PyAV upstream main の `61e4aa8`（current_ctx マージ済み）から wheel を自前ビルドします。

**av 18.1.0 が PyPI に公開されたら本節は丸ごと不要になります。** `pyproject.toml` の `av>=18,<19` がそのまま解決します（PyPI のバイナリ wheel は対応済みの FFmpeg を同梱しています）。

リンク先には **nv-codec-headers 12.2 以降でビルドされた FFmpeg 8** が必要です。distro の FFmpeg は古い nv-codec-headers でビルドされており、`hevc_nvenc` に `lookahead_level` オプションがありません。これにリンクした PyAV では、jasna の全エンコードが開始直後に `ValueError: hevc_nvenc did not accept encoder option(s): ['lookahead_level']` で失敗します。BtbN の共有ビルド（下記）はこの条件を満たすことを確認済みです。

### 4.1 BtbN FFmpeg 8.1 shared ビルドの取得

[BtbN/FFmpeg-Builds の Releases](https://github.com/BtbN/FFmpeg-Builds/releases) から `ffmpeg-n8.1-latest-linux64-gpl-shared-*.tar.xz` を取得して展開します。

```bash
cd "$WORKSPACE"
tar xf ffmpeg-n8.1-latest-linux64-gpl-shared-*.tar.xz
mv ffmpeg-n8.1-latest-linux64-gpl-shared "$WORKSPACE/ffmpeg-n8.1-shared"
ls "$WORKSPACE/ffmpeg-n8.1-shared/lib/pkgconfig"   # libavcodec.pc などが見えること
```

### 4.2 wheel のビルド

```bash
cd "$WORKSPACE"
git clone https://github.com/PyAV-Org/PyAV.git
cd PyAV
git checkout 61e4aa8

export PKG_CONFIG_PATH="$WORKSPACE/ffmpeg-n8.1-shared/lib/pkgconfig"
export VIRTUAL_ENV="$WORKSPACE/jasna/.venv"        # 3節で設定済みなら不要
uv build --wheel                                   # -> dist/av-18.0.0-cp311-abi3-linux_x86_64.whl
```

### 4.3 auditwheel での FFmpeg vendoring（必須）

ビルドした wheel には FFmpeg の `.so` を auditwheel で同梱（vendoring）します。この工程は省略できません。av と torchcodec バックエンドは同じ soname（`libavcodec.so.62` 等）のライブラリを別ビルドで持ち込むため、システム側の解決に任せると先にロードされた方へ他方が縛られ、ロード順次第で壊れます。auditwheel は同梱コピーの soname を av 専用に書き換える（mangling）ため、この衝突自体が起きなくなります。

```bash
uv pip install auditwheel patchelf
LD_LIBRARY_PATH="$WORKSPACE/ffmpeg-n8.1-shared/lib" \
  "$WORKSPACE/jasna/.venv/bin/auditwheel" repair dist/av-*.whl \
  --plat manylinux_2_41_x86_64 -w dist/
uv pip install dist/av-18.0.0-*manylinux*.whl
```

生成物は `av-18.0.0-cp311-abi3-manylinux_2_28_x86_64.manylinux_2_41_x86_64.whl` のような名前になります（abi3 wheel なので Python 3.13 でそのまま使えます）。

> **注意: この wheel はバージョン番号が PyPI の 18.0.0 と同じです。** `uv pip show av` では区別できず、取り違えると実行時に current_ctx 関連のエラーになります（トラブルシューティング参照）。5節の `uv pip install -e .[dev]` は同版の av を置き換えないため、本節を先に済ませておけばそのまま維持されます。

---

## 5. jasna 本体のインストール

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

これで `torchcodec>=0.15.0` が入ります。通常のセットアップには不要で、既定の `native` バックエンド（PyAV）は torchcodec なしで動作します。詳細は [TORCHCODEC_BACKEND_ja.md](TORCHCODEC_BACKEND_ja.md)。

**補足: FP8 復元バックエンドに追加のインストール手順は不要です。** 依存 `nvidia-cudnn-frontend` は `pyproject.toml` の通常依存で、上記コマンドで一緒に入ります。cuDNN ランタイム（9.17 以上）は torch cu130 wheel に同梱済みです。機能自体は実行時 opt-in（`--fp8-recon`、FP8 対応 GPU sm89 以上が必要）で、使えない環境では TensorRT エンジンにフォールバックします。詳細は [FP8_RECON_ja.md](FP8_RECON_ja.md)。

### 5.1 mmengine パッチの適用 (torch 2.6+ 対応)

`mmengine.runner.checkpoint` 内の `torch.load` 呼び出しに `weights_only=False` を追加します。torch 2.6 以降は既定が `weights_only=True` になり、既存の `.pth` チェックポイント読み込みが壊れるためです。diff は `patches/` に同梱。

```bash
cd "$WORKSPACE/jasna"
patch -p1 -d .venv/lib/python3.13/site-packages \
    < patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff

# 確認
grep -n "weights_only=False" .venv/lib/python3.13/site-packages/mmengine/runner/checkpoint.py
```

> `uv pip install -e .[dev]` を再実行すると `mmengine` が再インストールされ、このパッチが消えます。その後は再適用してください。

### 5.2 ONNX パッケージ（YOLO 検出モデル用）

**YOLO 系検出モデル（lada-yolo-\* と zelefans-vr-yolo-v2）**を使う場合、ultralytics は TensorRT エンジンをビルドする前にモデルを ONNX へエクスポートします。これには自動では入らない 3 パッケージが必要です。（RF-DETR モデルは不要。事前ビルド済み `.onnx` を TensorRT が直接パースするため）

```bash
uv pip install onnx onnxslim onnxruntime
```

ソースから実行する場合、ultralytics が初回エクスポート時に未導入ならこれらを自動ダウンロードしますが、事前に入れておくと初回 YOLO 実行時の待ちを回避できます。

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

**ソースから実行**する場合は `model_weights/rife.pth` を自動で参照する（他のウェイトと同じリゾルバ経由。または `JASNA_MODEL_WEIGHTS_DIR` をそれを含むフォルダに向ける）。詳細手順: [FRAME_GENERATION_ja.md](FRAME_GENERATION_ja.md)。

### 6.2 オプション: FlashVSR 二次復元（実験的）

FlashVSR（`--secondary-restoration flashvsr` / `flashvsr-inline`）は別リポジトリのチェックアウトと専用 venv を必要とし、inline 用にはチェックアウトへの同梱パッチ適用も要る。セットアップ手順は [FLASHVSR_ja.md](FLASHVSR_ja.md) を参照。

---

## 7. ソースから実行 / 動作確認

1〜6節が完了したら、venv 内でソースチェックアウトから直接 Jasna を実行します:

```bash
cd "$WORKSPACE/jasna"
python -m jasna --version    # -> 0.8.0+modi
python -m jasna --help
jasna --input assets/test_clip1_1080p.mp4 --output /tmp/out.mp4   # 短いクリップを処理
python -m jasna              # GUI を起動（引数なし）
```

> **GUI の起動は `python -m jasna`（引数なし）です。** console script の `jasna` は `jasna.main:main` に直結しており、引数なしでも GUI ディスパッチ（`jasna/__main__.py`）を通りません。frozen ビルドの引数なし起動だけが GUI に入ります。

`model_weights/` は自動解決され（付録 A.1）、`ffmpeg`/`ffprobe`（v8）はシステムの `PATH` から使われます。`mkvmerge` は不要になりました。インストールパスは ASCII のみ（起動時に強制）。

> **最初の処理実行は遅い。** 初回使用時に GPU 向け TensorRT エンジンがコンパイルされます（15〜60 分）。`model_weights/` 内のウェイト隣（`<モデル名>_sub_engines/` 等）にキャッシュされ、以降は再利用されます。v0.7.2 から移行した場合は、既定 max-clip-size の変更（90）に伴い一部エンジンが再コンパイルされます（loop_body 系のキャッシュは名前が変わらないため再利用されます）。

---

## 8. パッケージング / frozen ビルド

現時点で、このフォークから **Linux 向けのパッケージ化された frozen バイナリを生成する公開手段はありません。** upstream のパッケージングツールはプライベートな submodule（`jasna/protection`）にあり、この公開フォークには含まれません。フォーク同梱の実験的 Nuitka スクリプト（`scripts/build_nuitka.py`、[FROZEN_BUILD_ja.md](FROZEN_BUILD_ja.md)）は Windows 向けで、v0.8.0 のメディア層移行に未追従です（旧 `python_vali` / `PyNvVideoCodec` の DLL 同梱を前提としたまま）。

したがって公開でサポートされる経路は**ソースから実行**（7節）です。パッケージ済みバイナリが必要なら upstream Kruk2/jasna の公式リリースを使ってください。

---

## トラブルシューティング

### PyAV / FFmpeg

- **エンコード開始直後に `ValueError: hevc_nvenc did not accept encoder option(s): ['lookahead_level']`**
  PyAV wheel が古い nv-codec-headers の FFmpeg（distro ビルド等）にリンクされている。4節の BtbN ビルドに `PKG_CONFIG_PATH` を向けて wheel を作り直し、`uv pip install --reinstall` で入れ直す。

- **エンコーダ/デコーダ初期化時に `current_ctx` 関連の `TypeError` 等で失敗する**
  av が PyPI の 18.0.0 のまま（カスタム wheel 未導入、または依存再インストールで置き換わった）。4節の wheel を `uv pip install --reinstall dist/av-*manylinux*.whl` で入れ直す。

- **`uv build` が `libavcodec` を見つけられない**
  `pkg-config` 未導入か、`PKG_CONFIG_PATH` が BtbN 展開先の `lib/pkgconfig` を指していない（4.1〜4.2節）。

- **Jasna が起動拒否: ffprobe のバージョン違い**
  起動チェックは `ffprobe` の**メジャーバージョン 8** を要求します。`apt` の ffmpeg 8 を導入する（1.1節）。`mkvmerge` のチェックは v0.8.0 で廃止されました。

### 環境 / Python

- **GUI 起動時に `ModuleNotFoundError: No module named 'tkinter'`**
  GUI には Tcl/Tk が必要で、これは `python3.13-tk` パッケージです（注: `python3-tk` は別のインタプリタ向け）。`sudo apt-get install -y python3.13-tk`。CLI は無くても動作します。

- **venv の Python が 3.14**
  `uv venv --python /usr/bin/python3.13 .venv` とパス指定で作り直す（3節）。

### jasna インストール

- **`uv pip install -e .[dev]` が `no version of torch==2.12.0+cu130` で失敗**。`--extra-index-url https://download.pytorch.org/whl/cu130` を追加。
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
