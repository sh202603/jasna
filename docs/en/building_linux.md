# Building Jasna for Linux

How to set up Jasna on Linux and run it **from source**.

> **This guide covers the `v0.9.1+modi` branch.** The GPU stack (**torch 2.12.0+cu130 / torchvision 0.27.0+cu130 / torch-tensorrt 2.12.0+cu130 / tensorrt 10.16.1.11**) is unchanged from the v0.7.2 era, and the pins are already applied in `pyproject.toml` on this branch. TensorRT stays on the **10.16** line because `torch-tensorrt==2.12.0` requires `tensorrt>=10.16.1,<10.17.0` (torch-tensorrt does not support TensorRT 11 yet).

> **v0.8.0 simplified the setup considerably.** Upstream v0.8.0 moved the media layer to PyAV (NVDEC/NVENC), which removed the native builds of `python_vali` / `PyNvVideoCodec` entirely. With them went every prerequisite the old guide needed for those builds: the CUDA Toolkit, cmake / ninja, the ffmpeg dev packages (`libav*-dev`), the `FFMPEG_DIR` prefix, the `setuptools<80` pin, and `mkvmerge`. The one special step that remains is **building a PyAV wheel yourself** (Section 4). Since the v0.9.1 base, an **optional** VALI decode backend can be built again (Section 5.3), but nothing requires it.
>
> **Main additions on this branch:** 2x/4x frame generation via RIFE ([frame_generation.md](frame_generation.md)), the torchcodec backend ([torchcodec_backend.md](torchcodec_backend.md)), the cuDNN FP8 restoration backend ([fp8_recon.md](fp8_recon.md)), and FlashVSR secondary restoration ([flashvsr.md](flashvsr.md)). The AV1 / 8-bit / BT.601 and BT.2020 output features of v0.7.2+modi were absorbed into upstream v0.8.0. See [changes_vs_upstream.md](changes_vs_upstream.md) for the full delta.

> **Packaging note:** This public fork runs Jasna **from source**. There is no public way to produce a frozen Linux binary from it (the bundled experimental Nuitka script `scripts/build_nuitka.py` targets Windows, and has not been updated for the v0.8.0 media-layer migration either). If you want a pre-packaged binary, use upstream Kruk2/jasna's official releases.

> This guide is verified at **`0.9.1+modi` on the upstream `v0.9.1` tag (`a7cdaf8`)** on **Ubuntu 26.04 LTS** with an **RTX 5080** (2026-07-30: full pytest including e2e with a failure set identical to the vanilla v0.9.1 baseline measured on the same machine, plus CLI smokes for native / torchcodec+fp8-recon / fmp4 / rfdetr-v6 / frame-gen / segments / streaming / flashvsr-inline, a GUI launch smoke, and the optional VALI fork wheel of Section 5.3 built and active; earlier bases: d7a99bd 2026-07-23, v0.8.1 2026-07-19, v0.8.0 2026-07-18). Other distributions work too, but package names and the ffmpeg install steps will differ.

---

## 1. Prerequisites

| Category | Requirement | Source | Notes |
|---|---|---|---|
| OS | Ubuntu 26.04 x64 (or similar) | n/a | Verified on 26.04; ships ffmpeg 8 in `apt` |
| Build tools | `build-essential`, `pkg-config` | apt | Used only to build the PyAV wheel (Section 4) |
| Python | **3.13 + 3.13-dev + 3.13-tk** | apt (`python3.13 python3.13-dev python3.13-tk`) | v0.8.1 relaxed `requires-python` to `>=3.12`, but this guide is verified on 3.13 only (Ubuntu 26.04's default `python3` = 3.14 is unverified against the cu130 GPU stack). `python3.13-dev` is needed to build the PyAV wheel; `python3.13-tk` is needed to run the GUI |
| uv | latest | [astral.sh/uv](https://docs.astral.sh/uv/) | Manages the venv |
| NVIDIA driver | **580+** | your distro / NVIDIA | The startup check requires 580+ on Linux. GPU must be compute capability 7.5+ |
| ffmpeg / ffprobe | **v8** (runtime CLI) | apt | The startup check requires `ffprobe` major version 8; the `ffmpeg` CLI is used by HLS streaming and similar paths |

The CUDA Toolkit is no longer needed. The torch / tensorrt pip wheels bundle the CUDA runtime, and the only native extension you compile yourself is PyAV, which does not use CUDA (NVDEC/NVENC come from the linked FFmpeg).

### 1.1 Install system packages

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential pkg-config \
  python3.13 python3.13-dev python3.13-tk \
  ffmpeg
```

Install `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> ⚠ **The apt ffmpeg is used as a CLI only.** Linking the PyAV wheel against the distro's FFmpeg libraries makes every encode fail right at startup (see Section 4 for why). The `libav*-dev` packages the old guide required are no longer needed for the core setup (having them installed does no harm as long as `PKG_CONFIG_PATH` is set correctly); only the optional VALI backend build needs them again (Section 5.3).

---

## 2. Clone the repository

Only `jasna` needs to be cloned. `vali` / `PyNvVideoCodec` are no longer used by the v0.8.0 runtime, so checking them out is unnecessary. This guide calls the working root `$WORKSPACE`.

```bash
export WORKSPACE="$HOME/jasna-dev"     # ← any directory of your choice
mkdir -p "$WORKSPACE" && cd "$WORKSPACE"

git clone -b modi https://github.com/sh202603/jasna.git   # modi branch (this fork's default)
```

---

## 3. Create the Python virtual environment

Create `.venv` under the `jasna` repository from the system `python3.13`.

```bash
cd "$WORKSPACE/jasna"
ls -l /usr/bin/python3.13                       # install the python3.13 packages first if missing (Section 1.1)
uv venv --python /usr/bin/python3.13 .venv
export VIRTUAL_ENV="$WORKSPACE/jasna/.venv"
python --version                                # -> Python 3.13.x (recreate if it says 3.14)
```

> ⚠ **Pass `--python` as a path (`/usr/bin/python3.13`).** With the version-request form (`--python 3.13`) or no flag, uv may fetch a managed CPython (latest = 3.14) or pick up Ubuntu 26.04's default `python3` (3.14). If `python --version` reports 3.14, `rm -rf .venv` and recreate with the path form.

The old guide's tool pre-install (`cmake ninja scikit-build "setuptools<80" wheel numpy`) is obsolete; it existed only for the `vali` / `PyNvVideoCodec` builds that v0.8.0 removed.

---

## 4. Build and install the PyAV wheel (interim)

The GPU path needs PyAV's **current_ctx API** plus (since the v0.9.1 base) the **explicit CUDA stream support** the encoder passes via `CudaContext(cuda_stream=...)`. PyPI's av 18.0.0 has neither, so you build a wheel from PyAV upstream **main** (upstream docs say the same; `f6f0a5e` is the commit verified with this guide).

**Once av 18.1.0 reaches PyPI, this whole section becomes unnecessary.** The `av>=18,<19` requirement in `pyproject.toml` will resolve on its own (the PyPI binary wheels bundle a compatible FFmpeg).

The link target must be an **FFmpeg 8 built with nv-codec-headers 12.2+**. Distro FFmpeg builds use older nv-codec-headers whose `hevc_nvenc` lacks the `lookahead_level` option; with PyAV linked against one, every jasna encode fails right at startup with `ValueError: hevc_nvenc did not accept encoder option(s): ['lookahead_level']`. The BtbN shared builds (below) are confirmed to satisfy the requirement.

### 4.1 Get the BtbN FFmpeg 8.1 shared build

Download `ffmpeg-n8.1-latest-linux64-gpl-shared-*.tar.xz` from the [BtbN/FFmpeg-Builds releases](https://github.com/BtbN/FFmpeg-Builds/releases) and extract it.

```bash
cd "$WORKSPACE"
tar xf ffmpeg-n8.1-latest-linux64-gpl-shared-*.tar.xz
mv ffmpeg-n8.1-latest-linux64-gpl-shared "$WORKSPACE/ffmpeg-n8.1-shared"
ls "$WORKSPACE/ffmpeg-n8.1-shared/lib/pkgconfig"   # should list libavcodec.pc etc.
```

### 4.2 Build the wheel

```bash
cd "$WORKSPACE"
git clone https://github.com/PyAV-Org/PyAV.git
cd PyAV
# build from main; f6f0a5e is the commit verified with this guide
git checkout f6f0a5e

export PKG_CONFIG_PATH="$WORKSPACE/ffmpeg-n8.1-shared/lib/pkgconfig"
export VIRTUAL_ENV="$WORKSPACE/jasna/.venv"        # already set if you did Section 3 in this shell
uv build --wheel                                   # -> dist/av-18.0.0-cp311-abi3-linux_x86_64.whl
```

### 4.3 Vendor FFmpeg into the wheel with auditwheel (mandatory)

Vendor the FFmpeg `.so` files into the wheel with auditwheel. This step cannot be skipped: av and the torchcodec backend each bring libraries with the same sonames (`libavcodec.so.62` etc.) from different builds, and if resolution is left to the system, whichever loads first captures the other, so things break depending on load order. auditwheel rewrites (mangles) the sonames of the vendored copies to be av-private, which removes the collision entirely.

```bash
uv pip install auditwheel patchelf
LD_LIBRARY_PATH="$WORKSPACE/ffmpeg-n8.1-shared/lib" \
  "$WORKSPACE/jasna/.venv/bin/auditwheel" repair dist/av-*.whl \
  --plat manylinux_2_41_x86_64 -w dist/
uv pip install dist/av-18.0.0-*manylinux*.whl
```

The result is named like `av-18.0.0-cp311-abi3-manylinux_2_28_x86_64.manylinux_2_41_x86_64.whl` (an abi3 wheel, so it works on Python 3.13 as-is).

> **Note: this wheel carries the same version number as PyPI's 18.0.0.** `uv pip show av` cannot tell them apart; if they get mixed up you see current_ctx-related errors at runtime (see Troubleshooting). The `uv pip install -e .[dev,nvidia]` in Section 5 does not replace an installed av of the same version, so doing this section first keeps your wheel in place.

---

## 5. Install jasna itself

Since v0.8.1 the GPU stack is split into extras (`nvidia` = the NVIDIA stack, `amd` = the ROCm one). For an NVIDIA build the `nvidia` extra pulls in `torch==2.12.0+cu130` / `torchvision==0.27.0+cu130` / `torch-tensorrt==2.12.0` / `nvidia-vfx`, which are not on the default PyPI index. Point uv at the PyTorch cu130 index and add two flags:

```bash
cd "$WORKSPACE/jasna"
uv pip install -e .[dev,nvidia] \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

Why each flag:

- `--extra-index-url https://download.pytorch.org/whl/cu130`: source of the `torch+cu130` / `torch-tensorrt+cu130` wheels.
- `--index-strategy unsafe-best-match`: lets `torch-tensorrt==2.12.0` be satisfied by the index's local-version release `2.12.0+cu130`.
- `--prerelease=allow`: a transitive dependency (`nvidia-cuda-runtime-cu13`) is a prerelease.

The `[dev]` extra installs `nuitka>=2.4`, `pytest`, `pytest-cov`, `scikit-build`, `cmake`, `ninja`. The `[nvidia]` extra installs the GPU stack (torch cu130 / TensorRT / torch-tensorrt / nvidia-vfx; split out of the required dependencies in v0.8.1) — **omitting it leaves you without torch and the app will not start**.

**Optional: the torchcodec backend.** To use the experimental torchcodec decode/encode path (`--video-backend torchcodec`/`auto`), add the `torchcodec` extra and install `.[dev,nvidia,torchcodec]` with the same flags:

```bash
uv pip install -e .[dev,nvidia,torchcodec] \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

This installs `torchcodec>=0.15.0`. It is not needed for a normal setup; the default `native` backend (PyAV) works without torchcodec. Details: [torchcodec_backend.md](torchcodec_backend.md).

**Note: the FP8 restoration backend needs no extra install steps.** Its dependency `nvidia-cudnn-frontend` is a regular dependency in `pyproject.toml` and comes in with the command above. The cuDNN runtime (9.17+) ships inside the torch cu130 wheels. The feature itself is a runtime opt-in (`--fp8-recon`, needs an FP8-capable GPU, sm89+) and falls back to the TensorRT engines where unavailable. Details: [fp8_recon.md](fp8_recon.md).

**Optional: the TensorRT-RTX flavor.** Installing the `nvidia-rtx` extra *instead of* `nvidia` switches the whole TensorRT stack to TensorRT-RTX (JIT compilation, engine builds finish in seconds instead of minutes). The two extras are mutually exclusive in one venv — `torch-tensorrt` and `torch-tensorrt-rtx` both ship the `torch_tensorrt` package — so use a dedicated venv:

```bash
uv pip install -e .[dev,nvidia-rtx] \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

The flavor is detected automatically from the installed wheel; engines are cached under `.rtx`-tagged names, so both flavors can share one `model_weights` directory. The mmengine patch (§5.1) applies to this venv too. Details and measured numbers: [tensorrt_rtx.md](tensorrt_rtx.md).

### 5.1 Apply the mmengine patch (torch 2.6+ compatibility)

Add `weights_only=False` to the `torch.load` calls inside `mmengine.runner.checkpoint`. From torch 2.6 the default flipped to `weights_only=True`, which breaks loading the existing `.pth` checkpoints. The diff ships in `patches/`.

```bash
cd "$WORKSPACE/jasna"
patch -p1 -d .venv/lib/python3.13/site-packages \
    < patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff

# verify
grep -n "weights_only=False" .venv/lib/python3.13/site-packages/mmengine/runner/checkpoint.py
```

> Re-running `uv pip install -e .[dev,nvidia]` reinstalls `mmengine` and wipes this patch. Re-apply it afterwards.

### 5.2 ONNX packages (for YOLO detection models)

When using a **YOLO-backend detection model (lada-yolo-\* and zelefans-vr-yolo-v2)**, ultralytics exports the model to ONNX before building the TensorRT engine, which needs three packages that are not installed automatically. (RF-DETR models do not need them: their pre-built `.onnx` is parsed by TensorRT directly.)

```bash
uv pip install onnx onnxslim onnxruntime
```

When running from source, ultralytics auto-downloads them on the first export if missing, but pre-installing avoids the wait on the first YOLO run.

---

### 5.3 Optional: build the VALI NVDEC decode backend (`python_vali` fork)

Since the v0.9.1 base, the native decode path tries a VALI NVDEC decoder first and escalates to PyAV when it is unavailable (in-code `DECODE_BACKEND` toggle, default `auto`). Section 5 installs the stock PyPI `python_vali` wheel, which lacks the fork-only `DecodeSingleSurfaceAsyncDetailed` API, so the reader logs a warning and falls back to PyAV for every file. Everything works without this section; build the fork wheel only if you want the actual VALI decode path and its corrupt-packet tolerance.

Additional prerequisites on top of Section 1:

- **CUDA Toolkit 13.x** under `/usr/local/cuda-13.3`. The bare distro `nvcc` at `/usr/bin/nvcc` does not satisfy CMake's `FindCUDAToolkit`, so the toolkit paths are passed explicitly below.
- **FFmpeg 8 dev packages**: `sudo apt-get install -y libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev`. Unlike the av wheel (Section 4), the vali build links the distro FFmpeg and vendors nothing into the wheel. VALI only demuxes/parses with FFmpeg, so the nv-codec-headers problem that rules out distro FFmpeg for the av wheel does not apply here.
- cmake, ninja, scikit-build and a setuptools that still ships `pkg_resources`: all already present in the venv after Section 5.

```bash
cd "$WORKSPACE"
git clone https://codeberg.org/Kruk2/vali.git       # skip if you already have the checkout
cd vali
git submodule update --init --recursive

CUDACXX=/usr/local/cuda-13.3/bin/nvcc \
CUDAToolkit_ROOT=/usr/local/cuda-13.3 \
CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc" \
VIRTUAL_ENV="$WORKSPACE/jasna/.venv" \
  uv build --wheel --no-build-isolation
uv pip install --force-reinstall dist/python_vali-4.8.7-*.whl
```

`--no-build-isolation` is required because vali's `setup.py` imports `pkg_resources`, which an isolated build environment does not provide. `--force-reinstall` is required because the wheel carries the same version (4.8.7) as the PyPI package it replaces.

Verify:

```bash
python -c "import python_vali as v; print(hasattr(v.PyDecoder, 'DecodeSingleSurfaceAsyncDetailed'))"   # -> True
# a jasna run with --log-level info now prints: "Using VALI NVDEC decoder for <file>"
```

> Verified with fork commit `3ad0d54` (= the 4.8.7 pin, 2026-07-30, RTX 5080): both pipeline decode passes go through VALI, the output is md5-identical to the PyAV-decode output on the bundled test clip, and the wheel coexists with the av wheel and the torchcodec backend (those two carry auditwheel-mangled / vendored FFmpeg copies; vali resolves the distro's via ldconfig).

---

## 6. Model weights and assets

Place these 4 files in `$WORKSPACE/jasna/model_weights/`:

- `lada_mosaic_restoration_model_generic_v1.2.pth`
- `rfdetr-v5.onnx`
- `lada_mosaic_detection_model_v4_fast.pt`
- `lada_vr_mosaic_detection_model_v2_accurate.pt` (added in v0.8.0; the weights behind the VR180 YOLO detection model `zelefans-vr-yolo-v2`)

When running from source, Jasna resolves `model_weights/` automatically (Appendix A.1), so placing them there is all that's needed.

Detection models are auto-discovered from the files present in `model_weights/`, so everything else works without the fourth file; `zelefans-vr-yolo-v2` then simply does not appear in the model list and cannot be used for VR180 detection.

The two test clips normally ship with the repository in `$WORKSPACE/jasna/assets/` (`test_clip1_1080p.mp4`, `test_clip1_2160p.mp4`). If absent, extract them from an upstream release.

### 6.1 Optional: RIFE frame-interpolation model (for `--frame-gen`)

Needed only for frame-rate doubling (`--frame-gen {2x,4x}`). The RIFE weights are **not bundled** (non-commercial license terms), so create the TorchScript checkpoint yourself with `scripts/make_rife_torchscript.py`.

1. Clone Practical-RIFE and download a 4.x model package (**verified with v4.25**) so that `<repo>/train_log/` contains `RIFE_HDv3.py`, `IFNet_HDv3.py`, `flownet.pkl`:
   ```bash
   git clone https://github.com/hzwer/Practical-RIFE
   # follow its README to download/extract the model package into Practical-RIFE/train_log/
   ```
2. Convert to TorchScript with the project venv (run in `$WORKSPACE/jasna`):
   ```bash
   .venv/bin/python scripts/make_rife_torchscript.py \
       --rife-repo /path/to/Practical-RIFE \
       --output model_weights/rife.pth --validate
   ```

When **running from source**, `model_weights/rife.pth` is picked up automatically (same resolver as the other weights; or point `JASNA_MODEL_WEIGHTS_DIR` at a folder containing it). Full walkthrough: [frame_generation.md](frame_generation.md).

### 6.2 Optional: FlashVSR secondary restoration (experimental)

FlashVSR (`--secondary-restoration flashvsr` / `flashvsr-inline`) needs a separate repository checkout with its own venv, and the inline mode additionally needs a bundled patch applied to that checkout. Setup instructions: [flashvsr.md](flashvsr.md).

---

## 7. Run from source / smoke test

With Sections 1–6 done, run Jasna directly from the source checkout inside the venv:

```bash
cd "$WORKSPACE/jasna"
python -m jasna --version    # -> 0.9.1+modi
python -m jasna --help
jasna --input assets/test_clip1_1080p.mp4 --output /tmp/out.mp4   # process a short clip
python -m jasna              # launch the GUI (no arguments)
```

> **Launch the GUI with `python -m jasna` (no arguments).** The console script `jasna` is wired straight to `jasna.main:main` and never goes through the GUI dispatch (`jasna/__main__.py`), even with no arguments. Only the frozen build's no-argument launch enters the GUI.

`model_weights/` resolves automatically (Appendix A.1) and `ffmpeg`/`ffprobe` (v8) are taken from `PATH`. `mkvmerge` is no longer required. The install path must be ASCII-only (enforced at startup).

> **The first processing run is slow.** On first use the GPU-specific TensorRT engines are compiled (15–60 min). They are cached next to the weights inside `model_weights/` (e.g. `<model>_sub_engines/`) and reused afterwards. When migrating from v0.7.2, some engines recompile because the default max-clip-size changed (90); the loop_body engine caches keep their names and are reused.

---

## 8. Packaging / frozen builds

There is currently **no public way to produce a packaged frozen Linux binary from this fork.** Upstream's packaging tooling lives in a **private submodule (`jasna/protection`)** that this public fork does not include. The fork's experimental Nuitka script (`scripts/build_nuitka.py`, [frozen_build.md](frozen_build.md)) targets Windows, and has not been updated for the v0.8.0 media-layer migration (it still assumes bundling the old `python_vali` / `PyNvVideoCodec` DLLs).

The publicly supported path is therefore **running from source** (Section 7). If you need a packaged binary, use upstream Kruk2/jasna's official releases.

---

## Troubleshooting

### PyAV / FFmpeg

- **Every encode fails right at startup with `ValueError: hevc_nvenc did not accept encoder option(s): ['lookahead_level']`**
  The PyAV wheel is linked against an FFmpeg built with old nv-codec-headers (e.g. the distro build). Rebuild the wheel with `PKG_CONFIG_PATH` pointing at the BtbN build (Section 4) and reinstall with `uv pip install --reinstall`.

- **Encoder/decoder initialization fails with a current_ctx-related `TypeError` or similar**
  av is still PyPI's 18.0.0 (the custom wheel was never installed, or a dependency reinstall replaced it). Reinstall the Section 4 wheel with `uv pip install --reinstall dist/av-*manylinux*.whl`.

- **`uv build` cannot find `libavcodec`**
  `pkg-config` is missing, or `PKG_CONFIG_PATH` does not point at the extracted BtbN `lib/pkgconfig` (Sections 4.1–4.2).

- **Jasna refuses to start: wrong ffprobe version**
  The startup check requires `ffprobe` **major version 8**. Install apt's ffmpeg 8 (Section 1.1). The `mkvmerge` check was removed in v0.8.0.

### Environment / Python

- **GUI launch fails with `ModuleNotFoundError: No module named 'tkinter'`**
  The GUI needs Tcl/Tk, which is the `python3.13-tk` package (note: `python3-tk` targets a different interpreter). `sudo apt-get install -y python3.13-tk`. The CLI works without it.

- **The venv's Python is 3.14**
  Recreate it with the path form: `uv venv --python /usr/bin/python3.13 .venv` (Section 3).

### jasna install

- **`uv pip install -e .[dev,nvidia]` fails with `no version of torch==2.12.0+cu130`.** Add `--extra-index-url https://download.pytorch.org/whl/cu130`.
- **`torch-tensorrt==2.12.0+cu130 ... unsatisfiable`.** Add `--index-strategy unsafe-best-match --prerelease=allow` (Section 5).

### Runtime

- **Jasna raises `FileNotFoundError` for a `.pth`/`.onnx`/`.pt`**
  Files missing from `model_weights/`, or the resolver is looking elsewhere. Put the 3 weights in `$WORKSPACE/jasna/model_weights/` (Section 6), or set `JASNA_MODEL_WEIGHTS_DIR` to a folder containing them.

- **RTX Super-Res fails with `IRuntime::deserializeCudaEngine ... Version tag does not match`**
  `nvidia-vfx` (nvvfx) bundles TensorRT 10.9 which shares the soname `libnvinfer.so.10` with jasna's TensorRT 10.16; if nvvfx loads first, jasna's engines cannot be deserialized. The fix is applied on this branch (Appendix B.2; upstream later converged on the same fix). If you see this, consider recreating the venv.

- **Combining `--fp8-recon` with RTX Super-Res aborts with `Unable to load any of {libcudnn_graph.so.9.7.1, ...}`**
  nvvfx prepends its bundled libs directory to `LD_LIBRARY_PATH`, which contains only an incomplete cuDNN 9.7 dispatcher. The fix is applied on this branch (`jasna/restorer/fp8_upsample.py` puts torch's complete cuDNN first before `import cudnn`).

- **After a fatal CUDA error, re-running fails CUDA initialization (`Error 807` under MPS)**
  After a crash the main process can linger and keep holding the CUDA context; under MPS the lingering client blocks new initialization. Kill the leftover jasna process, then re-run.

---

## Appendix A: build/runtime modifications on this branch

Changes already applied to this branch's source (reference description; no standalone `.patch` files are shipped).

### A.1 Automatic resolution of the `model_weights/` directory

Adds `jasna/model_weights_resolver.py`, which searches for `model_weights/` in priority order: the `JASNA_MODEL_WEIGHTS_DIR` environment variable → next to the executable → the current directory → next to the source tree. `main.py`, `mosaic/detection_registry.py`, `engine_paths.py`, and the GUI (`gui/processor.py`, `gui/engine_preflight.py`) go through the resolver instead of a hardcoded `Path("model_weights")`. This lets the CLI run from any directory.

### A.2 DLL load assistance (torchcodec backend only)

With v0.8.0 the native path (PyAV) no longer needs DLL assistance. Today the torchcodec backend modules (`jasna/media/torchcodec_decoder.py` / `torchcodec_encoder.py`) register the torchcodec package directory and `CUDA_PATH\bin` (when set) on the DLL search path, **on Windows only**. On Linux this is a no-op.

### A.3 BasicVSR++ benchmark TRT API fix

Updates `jasna/benchmark/basicvsrpp_restoration.py` to the new `_preprocess_engine` API so `--benchmark basicvsrpp` works. Cross-platform.

---

## Appendix B: Linux GUI / RTX-VSR fixes on this branch

Linux-specific runtime fixes already applied to this branch's source. All are needed to make the GUI and RTX Super-Res work on Linux/X11; they do not affect Windows builds.

### B.1 Blank modal dialogs (`jasna/gui/app.py`, `jasna/gui/wizard.py`, `jasna/gui/components.py`)

**Problem**: On Linux/X11, customtkinter `CTkToplevel` dialogs open completely blank (a dark window with no text or buttons). Affected: **About** (`app.py` `_show_about`), **System Check** / first-run wizard (`wizard.py` `FirstRunWizard`), the **preset creation** and **confirmation** dialogs (`components.py` `PresetDialog`, `ConfirmDialog`). Each calls `grab_set()` (sometimes `lift()` / `focus_force()`) right after creation, before the child widgets have rendered. Under some window managers the window gets mapped but stays unrendered, and since the dialogs are non-resizable, no repaint ever fires.

**Fix**: Create all child widgets first, then defer `lift()` / `grab_set()` / `focus_force()` to a later event-loop tick via `self.after(200, …)` / `after(250, …)`, establishing the modal grab after the content has rendered. Windows behavior is unchanged (the same calls just run one tick later).

### B.2 RTX Super-Res TensorRT version clash (`jasna/restorer/rtx_superres_secondary_restorer.py`)

**Problem**: With RTX Super-Res enabled, jasna aborts while deserializing its TensorRT engines with `IRuntime::deserializeCudaEngine ... Serialization assertion stdVersionRead == kSERIALIZATION_VERSION failed. Version tag does not match`. The `nvidia-vfx` (nvvfx) package bundles its own TensorRT **10.9** (`nvvfx/libs/libnvinfer.so.10`) and loads it with `RTLD_GLOBAL` (`nvvfx/_lib_loader.py`), while jasna's pipeline engines are built with TensorRT **10.16** (`tensorrt_libs`). Both share the soname `libnvinfer.so.10`, and ELF symbol resolution uses whichever entered the global scope first. If nvvfx loads before jasna's TensorRT runtime, `torch-tensorrt` binds to nvvfx's old 10.9 and cannot read jasna's newer 10.16 engines.

**Fix**: Adds `_preload_tensorrt_runtime()`, executed at import time of the RTX Super-Res restorer module (before `nvvfx` is imported). On Linux it locates `tensorrt_libs` and preloads its `libnvinfer.so.10` / `libnvinfer_plugin.so.10` with `ctypes.RTLD_GLOBAL`, so TensorRT 10.16's symbols enter the global scope first and nvvfx's later load also resolves to 10.16. On Windows it is a no-op. Upstream later implemented the same fix independently (`6545b78`), so the function name matches upstream and the implementations have converged.
