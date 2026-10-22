# Building Jasna for Linux

How to set up Jasna on Linux and run it **from source**.

> **Branch and base.** This guide covers `0.11.3+modi` (upstream `cf8a968c` = the v0.11.3 tag). The GPU stack follows upstream's pins in `pyproject.toml` (torch 2.12.1+cu130 / torchvision 0.27.1+cu130 / torch-tensorrt 2.12.0 / tensorrt 10.16.1.11 / comfy-kitchen 0.2.38 / ultralytics 8.4.96+); this branch adds the `nvidia-vfx==0.2.0.0` pin (upstream leaves it unpinned), `nvidia-cudnn-frontend` as a regular dependency for FP8 restoration, the `torchcodec` extra, and the optional `nvidia-rtx` extra (torch 2.12.0 / torchvision 0.27.0 / torch-tensorrt-rtx 2.12.1). What the branch adds on top of upstream, and why, is in [changes_vs_upstream.md](changes_vs_upstream.md).
>
> **Verified** on Ubuntu 26.04 LTS with an RTX 5080 on 2026-10-16; what was run, and the earlier bases, are in [Appendix B](#appendix-b-history-of-this-setup). Other distributions work too, but package names and the ffmpeg install steps differ.
>
> **VALI is required.** You build Kruk2's VALI fork (`python_vali` 4.8.8) yourself and install it (Section 5.2). The `[nvidia]` extra pins `python_vali==4.8.8`, which is not published on PyPI (the newest release there is 4.8.7, as of 2026-10-16), and jasna's VALI decoder uses an API only the fork has. This makes the CUDA Toolkit and the FFmpeg dev packages prerequisites.
>
> **Run from source.** There is no frozen Linux binary of this fork (the experimental Nuitka script is Windows-only, [frozen_build.md](frozen_build.md)). LTX and the supporter models work only in upstream's official release.

---

## 1. Prerequisites

| Category | Requirement | Source | Notes |
|---|---|---|---|
| OS | Ubuntu 26.04 x64 (or similar) | n/a | Verified on 26.04; ships ffmpeg 8 in `apt` |
| Build tools | `build-essential`, `pkg-config` | apt | **Required**, for the VALI fork wheel build (Section 5.2) |
| Python | **3.13 + 3.13-dev + 3.13-tk** | apt (`python3.13 python3.13-dev python3.13-tk`) | `requires-python` is `>=3.12`, but this guide is verified on 3.13 only (Ubuntu 26.04's default `python3` = 3.14 is unverified against the cu130 GPU stack). `python3.13-dev` is needed for the VALI wheel build (Section 5.2); `python3.13-tk` is needed to run the GUI |
| uv | latest | [astral.sh/uv](https://docs.astral.sh/uv/) | Manages the venv |
| NVIDIA driver | **580+** | your distro / NVIDIA | The startup check requires 580+ on Linux. GPU must be compute capability 7.5+ |
| ffmpeg / ffprobe | **v8** (runtime CLI) | apt | The startup check requires `ffprobe` major version 8; the `ffmpeg` CLI is used by HLS streaming and similar paths |
| FFmpeg 8 dev | `libavcodec-dev` and the rest | apt | **Required**: the VALI build links against it (and VALI uses the distro FFmpeg libraries at runtime) |
| CUDA Toolkit | 13.x (verified with `/usr/local/cuda-13.3`) | NVIDIA | **Required**, for `nvcc` in the VALI build; not used at runtime |

The only thing compiled is the VALI fork wheel (Section 5.2). Every other GPU package, PyAV included, comes as a wheel, so the CUDA Toolkit is needed only for `nvcc` at build time.

### 1.1 Install system packages

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential pkg-config \
  python3.13 python3.13-dev python3.13-tk \
  ffmpeg \
  libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev
```

Install `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> ⚠ **The apt ffmpeg is used as a CLI only.** The av wheel from PyPI bundles its own FFmpeg and never touches the distro libraries (Section 4). The `libav*-dev` packages are there so that the VALI build (Section 5.2) can link against the distro FFmpeg.

---

## 2. Clone the repository

Clone `jasna` here; the VALI fork is cloned in Section 5.2. This guide calls the working root `$WORKSPACE`.

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

---

## 4. PyAV

Nothing to build. The GPU path needs PyAV's current_ctx API and the explicit CUDA stream support of av 18.1; `pyproject.toml` pins `av==18.1.0` (the release upstream's VALI wheel is built against), and the Section 5 install resolves it from PyPI. The wheel bundles its own FFmpeg 8 built with new-enough nv-codec-headers (`hevc_nvenc` accepts `lookahead_level`), so the distro FFmpeg is never linked.

A venv that still carries the interim self-built `18.0.0` wheel from before August 2026 is fixed by the Section 5 install; see [Appendix B](#appendix-b-history-of-this-setup).

---

## 5. Install jasna itself

Install in this order. The `[nvidia]` extra pins `python_vali==4.8.8`, but 4.8.8 is not on PyPI, so first build the wheel from Kruk2's fork ([codeberg.org/Kruk2/vali](https://codeberg.org/Kruk2/vali)), then hand it to the jasna install with `--find-links`.

1. Install the packages the VALI build needs into the venv (Section 5.1)
2. Build the VALI fork wheel (Section 5.2)
3. Install jasna itself (Section 5.3)
4. Apply the mmengine patch (Section 5.4)

### 5.1 Install the build packages

```bash
cd "$WORKSPACE/jasna"
uv pip install cmake ninja scikit-build "setuptools<81"
```

A venv created with `uv venv` has no setuptools. vali's `setup.py` imports `pkg_resources`, and setuptools 80.x answers that import with a warning that `pkg_resources` is slated for removal and to pin `Setuptools<81`, so pin `<81`. cmake, ninja and scikit-build also come with the `[dev]` extra in Section 5.3, but the build runs first, so install them here.

### 5.2 Build the VALI fork wheel

The native decode path tries a VALI NVDEC decoder first and escalates to PyAV when it is unavailable; the `JASNA_DECODE_BACKEND` environment variable (`auto` / `vali` / `pyav-hw` / `pyav-sw`, default `auto`) selects the chain explicitly, and modi's `--video-backend` sits one layer above it (the chain applies when `native` is selected). jasna's VALI decoder uses `DecodeSingleSurfaceAsyncDetailed`, an API that only the fork has; with a wheel that lacks it, the reader logs a warning and falls back to PyAV for every file.

Prerequisites (installed in Section 1):

- **CUDA Toolkit 13.x** under `/usr/local/cuda-13.3`. The bare distro `nvcc` at `/usr/bin/nvcc` does not satisfy CMake's `FindCUDAToolkit`, so the toolkit paths are passed explicitly below.
- **FFmpeg 8 dev packages** (`libavcodec-dev` and the rest). Unlike the av wheel (which vendors its own FFmpeg), the vali build links the distro FFmpeg and vendors nothing into the wheel. VALI only demuxes/parses with FFmpeg, so the nv-codec-headers constraint on the av side's FFmpeg does not apply here.

```bash
cd "$WORKSPACE"
git clone https://codeberg.org/Kruk2/vali.git       # skip if you already have the checkout
cd vali
git checkout 0e5c01e      # = the 4.8.8 pin: DecodeSingleSurfaceAsyncDetailed, the ASF/VC-1 timestamp fixes, the decoder-open GIL release
git submodule update --init --recursive

CUDACXX=/usr/local/cuda-13.3/bin/nvcc \
CUDAToolkit_ROOT=/usr/local/cuda-13.3 \
CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc" \
VIRTUAL_ENV="$WORKSPACE/jasna/.venv" \
  uv build --wheel --no-build-isolation
ls dist/                  # -> python_vali-4.8.8-cp313-cp313-linux_x86_64.whl
```

`--no-build-isolation` is required because vali's `setup.py` imports `pkg_resources`, which an isolated build environment does not provide (this makes the build use the setuptools installed in Section 5.1). When an existing checkout moves to another commit, delete the previous CMake cache `_skbuild/` before rebuilding. CMake downloads pybind11 during the build, so it needs network access.

> Verified with fork commit `0e5c01e` (= the 4.8.8 pin, 2026-10-16, RTX 5080; before that `3ad0d54` = 4.8.7 on 2026-07-30): both pipeline decode passes go through VALI, the output is md5-identical to the PyAV-decode output on the bundled test clip, the decoder tests pass, and the wheel coexists with the av wheel and the torchcodec backend (those two carry auditwheel-mangled / vendored FFmpeg copies; vali resolves the distro's via ldconfig).

### 5.3 Install jasna itself

The GPU stack lives in extras (`nvidia` = the NVIDIA stack, `amd` = the ROCm one). For an NVIDIA build the `nvidia` extra pulls in `torch==2.12.1+cu130` / `torchvision==0.27.1+cu130` / `tensorrt==10.16.1.11` / `torch-tensorrt==2.12.0` / `nvidia-vfx==0.2.0.0` / `comfy-kitchen==0.2.38` / `python_vali==4.8.8`, some of which are not on the default PyPI index. Point uv at the Section 5.2 wheel and the PyTorch cu130 index, and add two flags:

```bash
cd "$WORKSPACE/jasna"
uv pip install -e .[dev,nvidia] \
    --find-links "$WORKSPACE/vali/dist" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

Why each flag:

- `--find-links "$WORKSPACE/vali/dist"`: source of the `python_vali-4.8.8` wheel built in Section 5.2. Without it, `python_vali==4.8.8` cannot be resolved and the install fails.
- `--extra-index-url https://download.pytorch.org/whl/cu130`: source of the `torch+cu130` / `torch-tensorrt+cu130` wheels.
- `--index-strategy unsafe-best-match`: lets `torch-tensorrt==2.12.0` be satisfied by the index's local-version release `2.12.0+cu130`.
- `--prerelease=allow`: a transitive dependency (`nvidia-cuda-runtime-cu13`) is a prerelease.

The `[dev]` extra installs `nuitka>=2.4`, `pytest`, `pytest-cov`, `scikit-build`, `cmake`, `ninja`. The `[nvidia]` extra installs the GPU stack (torch cu130 / TensorRT / torch-tensorrt / nvidia-vfx / VALI) — **omitting it leaves you without torch and the app will not start**.

Check that the fork build of VALI is the one installed:

```bash
python -c "import python_vali as v; print(hasattr(v.PyDecoder, 'DecodeSingleSurfaceAsyncDetailed'))"   # -> True
# a jasna run with --log-level info now prints: "Using VALI NVDEC decoder for <file>"
```

If it prints `False`, a non-fork wheel got installed. Reinstall the Section 5.2 wheel (`--no-deps` keeps uv from touching the rest of the venv):

```bash
uv pip install --force-reinstall --no-deps "$WORKSPACE"/vali/dist/python_vali-4.8.8-*.whl
```

**Note: `nvidia-vfx` is pinned to 0.2.0.0 (VFX SDK 1.3.0).** The PyPI entry is a 4 KB stub that downloads the real 443 MB wheel from `pypi.nvidia.com` during the install above, so no extra index is needed. 0.2.0.0 provides the `VideoFrameGeneration` effect used by `--frame-gen-backend rtx` ([frame_generation.md](frame_generation.md)) and bundles no TensorRT. An older venv with 0.1.0.1 is upgraded in place with `uv pip install nvidia-vfx==0.2.0.0` ([Appendix B](#appendix-b-history-of-this-setup)).

**Optional: the torchcodec backend.** To use the experimental torchcodec decode/encode path (`--video-backend torchcodec`/`auto`), add the `torchcodec` extra and install `.[dev,nvidia,torchcodec]` with the same flags (the torchcodec on PyPI is the CPU-only build; `--extra-index-url` is what lets uv pick the `+cu130` build from the PyTorch index):

```bash
uv pip install -e .[dev,nvidia,torchcodec] \
    --find-links "$WORKSPACE/vali/dist" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

This installs `torchcodec>=0.15.0`. It is not needed for a normal setup; the default `native` backend (PyAV) works without torchcodec. Details: [torchcodec_backend.md](torchcodec_backend.md).

**Note: the FP8 restoration backend needs no extra install steps.** Its dependency `nvidia-cudnn-frontend` is a regular dependency in `pyproject.toml` and comes in with the command above. The cuDNN runtime (9.17+) ships inside the torch cu130 wheels. The feature itself is a runtime opt-in (`--fp8-recon`, needs an FP8-capable GPU, sm89+) and falls back to the TensorRT engines where unavailable. Details: [fp8_recon.md](fp8_recon.md).

**Optional: the TensorRT-RTX flavor.** Installing the `nvidia-rtx` extra *instead of* `nvidia` switches the whole TensorRT stack to TensorRT-RTX (JIT compilation, engine builds finish in seconds instead of minutes). The two extras are mutually exclusive in one venv — `torch-tensorrt` and `torch-tensorrt-rtx` both ship the `torch_tensorrt` package — so use a dedicated venv. The `nvidia-rtx` extra pins `python_vali==4.8.8` too, so it needs the same `--find-links`; it pins torch 2.12.0 / torchvision 0.27.0, unlike the `nvidia` extra's 2.12.1 / 0.27.1:

```bash
uv pip install -e .[dev,nvidia-rtx] \
    --find-links "$WORKSPACE/vali/dist" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

The flavor is detected automatically from the installed wheel; engines are cached under `.rtx`-tagged names, so both flavors can share one `model_weights` directory. The mmengine patch (Section 5.4) applies to this venv too. Details and measured numbers: [tensorrt_rtx.md](tensorrt_rtx.md).

### 5.4 Apply the mmengine patch

Apply `patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff` to the venv's mmengine 0.10.7. The diff changes four files:

- `mmengine/runner/checkpoint.py`: adds `weights_only=False` to `torch.load`. From torch 2.6 the default is `weights_only=True`, which cannot read the mmengine `HistoryBuffer` stored in BasicVSR++ weights saved as a training checkpoint (such as the 287 MB full `lada_mosaic_restoration_model_generic_v1.2.pth`) and stops with `Weights only load failed`. The standard 78 MB file placed in Section 6 loads without this change
- `mmengine/model/__init__.py` and `mmengine/model/wrappers/__init__.py`: import the FSDP wrapper only when `torch.distributed.is_available()` (so mmengine imports on a torch without distributed support)
- `mmengine/dist/dist.py`: adds `from __future__ import annotations`

> Upstream changed that one line to `weights_only=True` in `7c64105`. That version still reads the standard `v1.2.pth` from Section 6, but not the full file of the same name (a 287 MB training checkpoint). This fork ships the diff with `False` so that training checkpoints such as the full file still load ([changes_vs_upstream.md](changes_vs_upstream.md), Section 8). To convert the full file so that it loads with `weights_only=True`, see [Appendix A of models.md](models.md#appendix-a-converting-a-restoration-model-that-weights_onlytrue-cannot-load).

```bash
cd "$WORKSPACE/jasna"
patch -p1 -d .venv/lib/python3.13/site-packages \
    < patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff

# verify (one matching line each means it's applied)
grep -n "weights_only=False" .venv/lib/python3.13/site-packages/mmengine/runner/checkpoint.py
grep -n "is_available" .venv/lib/python3.13/site-packages/mmengine/model/__init__.py
```

> Reinstalling mmengine wipes the patch (`uv pip install` reinstalls it only when its version changes or with `--reinstall`); re-apply it afterwards. Applying it to an already-patched venv fails (`Hunk #1 FAILED` and the like), and so does applying it over an older or the upstream version of the diff; in that case restore pristine mmengine first: `uv pip install --reinstall --no-deps mmengine==0.10.7`.

### 5.5 YOLO detection models need no extra packages

A YOLO-backend detection model (`lada-yolo-*`, `zelefans-vr-yolo-v2`) is exported to ONNX by ultralytics before its TensorRT engine is built. `onnx` is a regular dependency and the export runs with `simplify=False`, so neither `onnxslim` nor `onnxruntime` is needed. RF-DETR models ship a pre-built `.onnx` that TensorRT parses directly.

---

## 6. Model weights and assets

Place these files in `$WORKSPACE/jasna/model_weights/` (taken from an upstream release package):

- `lada_mosaic_restoration_model_generic_v1.2.pth` (the standard 78 MB file; a 287 MB full file circulates under the same name, see [Appendix A of models.md](models.md#appendix-a-converting-a-restoration-model-that-weights_onlytrue-cannot-load))
- `rfdetr-v6.onnx` (the default detector since v0.9.0; `rfdetr-v6-large.onnx` for 4K and `rfdetr-vr-v1.onnx` for VR180 are optional, and the old `rfdetr-v5.onnx` still works)
- `lada_mosaic_detection_model_v4_fast.pt`
- `lada_vr_mosaic_detection_model_v2_accurate.pt` (the weights behind the VR180 YOLO detection model `zelefans-vr-yolo-v2`)

When running from source, Jasna resolves `model_weights/` automatically (Appendix A.1), so placing them there is all that's needed.

Detection models are auto-discovered from the files present in `model_weights/`, so everything else works without the fourth file; `zelefans-vr-yolo-v2` then simply does not appear in the model list and cannot be used for VR180 detection.

The two test clips normally ship with the repository in `$WORKSPACE/jasna/assets/` (`test_clip1_1080p.mp4`, `test_clip1_2160p.mp4`). If absent, extract them from an upstream release.

### 6.1 Optional: RIFE frame-interpolation model (for `--frame-gen`)

Needed only for frame-rate doubling (`--frame-gen {2x,4x}`) with the default `rife` backend. On an RTX 40 series or newer GPU, `--frame-gen-backend rtx` (NVIDIA RTX Video Frame Generation from the `nvidia-vfx` wheel installed in Section 5) needs no weights at all; see [frame_generation.md](frame_generation.md). The RIFE weights are **not bundled** (non-commercial license terms), so create the TorchScript checkpoint yourself with `scripts/make_rife_torchscript.py`.

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

FlashVSR (`--secondary-restoration flashvsr` / `flashvsr-inline`) needs a separate repository checkout with its own venv. Use the fork [`sh202603/FlashVSR_plus`](https://github.com/sh202603/FlashVSR_plus) (it includes the tiny-long fix inline needs and the `--flashvsr-accel` speed-up, and `uv sync` installs its dependencies). Setup instructions: [flashvsr.md](flashvsr.md).

---

### 6.3 Optional: SeedVR2 primary restoration (experimental)

SeedVR2 (`--restoration-model-name seedvr2`) needs a separate `ComfyUI-SeedVR2_VideoUpscaler` checkout with its own venv plus the ~90 MB LoRA in `model_weights/`; no patch is required and nothing is installed into jasna's venv. Setup instructions: [seedvr2.md](seedvr2.md).

---

### 6.4 Optional: SwiftVR secondary restoration and SwiftVR distill (experimental)

SwiftVR (`--secondary-restoration swiftvr` / `swiftvr-inline`, also the `swiftvr` backend of `--ltx-crop-vsr`) needs the fork [`sh202603/SwiftVR`](https://github.com/sh202603/SwiftVR) (branch `modi`) checked out with its own `uv sync` venv and the ~20 GB checkpoint. The small distilled model (`--secondary-restoration swiftvr-distill`, also the `swiftvr-distill` backend of `--ltx-crop-vsr`) needs nothing but `swiftvr-distill.pt` from [okatti/swiftvr-distill](https://huggingface.co/okatti/swiftvr-distill) in `model_weights/`. Setup instructions: [swiftvr.md](swiftvr.md).

---

## 7. Run from source / smoke test

With Sections 1–6 done, run Jasna directly from the source checkout inside the venv:

```bash
cd "$WORKSPACE/jasna"
python -m jasna --version    # -> 0.11.3+modi
python -m jasna --help
python -m jasna --input assets/test_clip1_1080p.mp4 --output /tmp/out.mp4   # process a short clip
python -m jasna              # launch the GUI (no arguments)
```

> **Launch the GUI with `python -m jasna` (no arguments).** The console script `jasna` is wired straight to `jasna.main:main` and never goes through the GUI dispatch (`jasna/__main__.py`), even with no arguments. Only the frozen build's no-argument launch enters the GUI. Prefer `python -m jasna` for the CLI too: the console script `jasna` skips what `__main__.py` sets up first (`CUDA_MODULE_LOADING=LAZY` and the guard that stops multiprocessing children).

`model_weights/` resolves automatically (Appendix A.1) and `ffmpeg`/`ffprobe` (v8) are taken from `PATH`. The install path must be ASCII-only (enforced at startup).

> **The first processing run is slow.** On first use the GPU-specific TensorRT engines are compiled (15–60 min). They are cached next to the weights inside `model_weights/` (e.g. `<model>_sub_engines/`) and reused afterwards. Engine caches from the v0.9.1 base onwards stay valid (the engine code changed through v0.11.1, but TensorRT is still 10.16.1.11 and the engine file names are the same); `--max-clip-size` no longer triggers a recompile.

---

## 8. Packaging / frozen builds

There is currently **no public way to produce a packaged frozen Linux binary from this fork.** Upstream's packaging tooling lives in a **private submodule (`jasna/protection`)** that this public fork does not include. The fork's own experimental Nuitka script (`scripts/build_nuitka.py`, [frozen_build.md](frozen_build.md)) is Windows-only.

The publicly supported path is therefore **running from source** (Section 7). If you need a packaged binary, use upstream Kruk2/jasna's official releases.

---

## Troubleshooting

### PyAV / FFmpeg

- **Every encode fails right at startup with `ValueError: hevc_nvenc did not accept encoder option(s): ['lookahead_level']`**
  The av wheel is linked against an FFmpeg built with old nv-codec-headers. This cannot happen with the official PyPI wheel (18.1+ bundles a compatible FFmpeg); it indicates a leftover wheel from the interim self-build era or a source build against the distro FFmpeg. Fix: `uv pip install --reinstall "av>=18.1"`.

- **Encoder/decoder initialization fails with a current_ctx-related `TypeError` or similar**
  av is still 18.0.0: either PyPI's old release or the interim self-built wheel, which also reports `18.0.0`. Fix: `uv pip install "av>=18.1"` (Section 4).

- **Jasna refuses to start: wrong ffprobe version**
  The startup check requires `ffprobe` **major version 8**. Install apt's ffmpeg 8 (Section 1.1).

### Environment / Python

- **GUI launch fails with `ModuleNotFoundError: No module named 'tkinter'`**
  The GUI needs Tcl/Tk, which is the `python3.13-tk` package (note: `python3-tk` targets a different interpreter). `sudo apt-get install -y python3.13-tk`. The CLI works without it.

- **The venv's Python is 3.14**
  Recreate it with the path form: `uv venv --python /usr/bin/python3.13 .venv` (Section 3).

### jasna install

- **`uv pip install -e .[dev,nvidia]` cannot resolve `python_vali==4.8.8`.** 4.8.8 is not on PyPI. Build the VALI fork wheel (Section 5.2) and re-run with `--find-links "$WORKSPACE/vali/dist"` (Section 5.3).
- **The VALI build fails with `ModuleNotFoundError: No module named 'pkg_resources'`.** setuptools 81 or newer is installed. Go back with `uv pip install "setuptools<81"` (Section 5.1).
- **`uv pip install -e .[dev,nvidia]` fails with `no version of torch==2.12.1+cu130`.** Add `--extra-index-url https://download.pytorch.org/whl/cu130`.
- **`torch-tensorrt==2.12.0+cu130 ... unsatisfiable`.** Add `--index-strategy unsafe-best-match --prerelease=allow` (Section 5).

### Runtime

- **Jasna raises `FileNotFoundError` for a `.pth`/`.onnx`/`.pt`**
  Files missing from `model_weights/`, or the resolver is looking elsewhere. Put the Section 6 weights in `$WORKSPACE/jasna/model_weights/`, or set `JASNA_MODEL_WEIGHTS_DIR` to a folder containing them.

- **Loading BasicVSR++ fails with `Weights only load failed ... HistoryBuffer`**
  The `v1.2.pth` in place is the 287 MB full file (a training checkpoint), and the mmengine patch is missing or the upstream version (`weights_only=True`) is applied. Restore pristine mmengine and apply the bundled diff as in Section 5.4, replace it with the standard file from Section 6, or convert the full file as in [Appendix A of models.md](models.md#appendix-a-converting-a-restoration-model-that-weights_onlytrue-cannot-load).

- **RTX Super-Res fails with `IRuntime::deserializeCudaEngine ... Version tag does not match`**
  Only with `nvidia-vfx` 0.1.0.1 or older: those wheels bundle TensorRT 10.9, which shares the soname `libnvinfer.so.10` with jasna's TensorRT 10.16; if nvvfx loads first, jasna's engines cannot be deserialized. The pinned 0.2.0.0 ships no TensorRT, and the preload guard against the clash is still in place (Appendix A.3). If you see this, run `uv pip install nvidia-vfx==0.2.0.0`.

- **`--frame-gen-backend rtx` says RTX Video Frame Generation is not available in nvidia-vfx 0.1.0.1**
  The venv predates the 0.2.0.0 pin (Section 5). `uv pip install nvidia-vfx==0.2.0.0`. The effect also needs an RTX 40 series or newer GPU and driver 570.190+ / 580.82+ / 590.44+.

- **Combining `--fp8-recon` with RTX Super-Res aborts with `Unable to load any of {libcudnn_graph.so.9.7.1, ...}`**
  nvvfx prepends its bundled libs directory to `LD_LIBRARY_PATH`, which contains only an incomplete cuDNN 9.7 dispatcher. The fix is applied on this branch (`jasna/restorer/fp8_upsample.py` puts torch's complete cuDNN first before `import cudnn`).

- **After a fatal CUDA error, re-running fails CUDA initialization (`Error 807` under MPS)**
  After a crash the main process can linger and keep holding the CUDA context; under MPS the lingering client blocks new initialization. Kill the leftover jasna process, then re-run.

---

## Appendix A: branch changes this setup relies on

Everything below is applied in the branch's source; no standalone patch files are shipped (the two bundled patches, for mmengine and FlashVSR_plus, are applied to other packages: Section 5.4 and [flashvsr.md](flashvsr.md)). The full list of what the branch changes is [changes_vs_upstream.md](changes_vs_upstream.md).

### A.1 Automatic resolution of the `model_weights/` directory

`jasna/model_weights_resolver.py` looks for `model_weights/` in this order: the `JASNA_MODEL_WEIGHTS_DIR` environment variable, next to the executable (frozen builds only), the current directory, then next to the `jasna` package (the source tree). The CLI, the detection registry, the engine paths and the GUI go through it instead of a hardcoded `Path("model_weights")`, so the CLI runs from any directory and Section 6 only has to place the files.

### A.2 Modal dialogs on Linux/X11

The About dialog and the first-run wizard defer upstream's `grab_modal()` by 250 ms (`jasna/gui/app.py`, `jasna/gui/wizard.py`) so the widgets are painted before the modal grab; under some X11 window managers an early `grab_set()` left the non-resizable `CTkToplevel` blank. Upstream's other dialogs call `wait_visibility()` before the grab and are unchanged.

### A.3 TensorRT runtime preload in the RTX Super Res restorer

`_preload_tensorrt_runtime()` in `jasna/restorer/rtx_superres_secondary_restorer.py` loads `tensorrt_libs`' `libnvinfer.so.10` with `RTLD_GLOBAL` before `nvvfx` is imported. It guarded against nvvfx 0.1.0.1's bundled TensorRT 10.9: same soname as jasna's 10.16, so whichever loaded first won and engines built with 10.16 failed to deserialize. The pinned 0.2.0.0 bundles no TensorRT, so the function is a no-op today; upstream carries the same function (`6545b78`), and the branch keeps it for venvs that still have 0.1.0.1.

---

## Appendix B: history of this setup

What changed in the from-source setup at each base, newest first, and when this guide was verified. Procedures that are no longer needed are only summarized; their full text is in this file's git history.

- **v0.11.3 base (verified 2026-10-22, RTX 5080, Ubuntu 26.04)**: no pin change (`pyproject.toml` differs from v0.11.2 only in the version), no native code change, no engine recompile and no VALI rebuild: a `0.11.2+modi` venv runs it as is. Upstream's delta is the LTX 2-step / 4-step models with faster VAE passes, the scene-cut detector's repeated-frame handling, PyAV's FFmpeg log hook left off and tolerant metadata decoding (details in [changes_vs_upstream.md](changes_vs_upstream.md), Appendix B). Verified 2026-10-22: the full test suite fails in the same 125 tests as pristine upstream (none modi-only), the CLI e2e matrix of every modi feature completes, and the `--ltx-crop-vsr` GPU test passes.
- **v0.11.2 base (verified 2026-10-18, RTX 5080, Ubuntu 26.04)**: the only pin change is `comfy-kitchen` 0.2.36 to 0.2.38 (kernel changes below SM 8.9 only; `uv pip install comfy-kitchen==0.2.38` in an existing venv is enough); upstream adds the LTX less-overlap mode and a hard-disk install warning. No VALI rebuild (still `0e5c01e`) and no engine recompile. Verification (Linux, RTX 5080, 2026-10-18): the full test suite from a checkout with the model weights, compared with pristine upstream `b623466f` under the same deselect and timeout, fails only where upstream fails (protection, the native LTX build with gcc-15, the display environment, ultralytics); a CLI run of every modi feature on the 10 s test clip (BasicVSR++ with and without FP8, torchcodec, fMP4, RTX Super Res, SwiftVR distill, SwiftVR offline, SeedVR2, RIFE and RTX frame generation, `--segments`, `--ltx-crop-vsr swiftvr-distill` on the 30 s probe clip, `--ltx-trial` with both hops, and the LTX + frame-gen rejection); the hard-disk check returns False on this NVMe install; and `comfy-kitchen` 0.2.38 installed into the existing venv with the kitchen-related tests passing. Known trap: the full suite from a checkout **without** `model_weights/` aborts the pytest process at `tests/test_fp8_upsample.py::test_fp8_forward_parity_small` (`cudnn.create_handle()` aborts; exit 134); it is not new (the v0.11.1 base does the same) and does not reproduce in isolation.
- **v0.11.1 base (verified 2026-10-16, RTX 5080, Ubuntu 26.04)**: upstream pins torch 2.12.1 / torchvision 0.27.1, adds `comfy-kitchen==0.2.36` and `ultralytics>=8.4.96` (whose YOLO export needs neither onnxruntime nor onnxslim, so Section 5.5 shrank to a note), pins `av==18.1.0` exactly, and selects the decode backend only through `JASNA_DECODE_BACKEND` (the in-code `DECODE_BACKEND` toggle is gone). The VALI fork wheel was rebuilt at Kruk2's `0e5c01e` (4.8.8 with the ASF/VC-1 timestamp fixes and the decoder-open GIL release). Verification: the full pytest suite compared with pristine upstream under the same deselect and timeout (every remaining failure also fails upstream), end-to-end CLI runs of the modi features (native and torchcodec backends, FP8, frame generation, SeedVR2, FlashVSR inline, SwiftVR inline / offline / distill, `--ltx-crop-vsr`, segments, streaming), and the VALI fork wheel active on both decode passes. The same day, since `python_vali==4.8.8` is not on PyPI (a fresh venv cannot resolve `[nvidia]`), VALI became a required step and Section 5 was reordered: build packages (including setuptools<81), the VALI wheel build, the jasna install with `--find-links`, then the mmengine patch. For the mmengine patch, the version upstream changed to `weights_only=True` in `7c64105` was found to break loading the full `v1.2.pth` in a source run, so the fork keeps shipping it with `False` (the section now describes all four files). The new Section 5 was then run end to end from an empty venv and a fresh vali clone: `[nvidia]` resolved through `--find-links`, `DecodeSingleSurfaceAsyncDetailed` was `True`, and after the mmengine patch `load_model()` read `v1.2.pth`.
- **nvidia-vfx 0.2.0.0 pin (2026-10-03)**: the branch pins `nvidia-vfx==0.2.0.0` (upstream leaves it unpinned) for the `VideoFrameGeneration` effect behind `--frame-gen-backend rtx`. 0.2.0.0 no longer bundles TensorRT, which ends the `libnvinfer.so.10` clash of Appendix A.3, and its Linux wheel shrinks from about 1.1 GB to 443 MB. Venvs created earlier hold 0.1.0.1: `uv pip install nvidia-vfx==0.2.0.0`.
- **PyAV from PyPI (2026-08-24)**: av 18.1.0 reached PyPI with the current_ctx API and the explicit CUDA stream support the encoder uses, ending the interim procedure of building a wheel from PyAV main (commit `f6f0a5e`, linked against a BtbN FFmpeg 8.1 shared build and vendored with auditwheel). That interim wheel reports `18.0.0`; the Section 5 install replaces it, or `uv pip install "av>=18.1"` does so at once. Verified with the official wheel on Linux: the media-layer suite and the full-pipeline e2e including NVDEC decode and NVENC encode.
- **v0.10.0 base (2026-08-13)**: the `JASNA_DECODE_BACKEND` environment variable (`auto` / `vali` / `pyav-hw` / `pyav-sw`); `python_vali` pinned at 4.8.8 (fork commit `f4a67f8`).
- **v0.9.1 base (verified 2026-07-30)**: the VALI NVDEC decode backend returned as an optional, self-built fork wheel (now Section 5.2; fork commit `3ad0d54` = 4.8.7 at the time). The encoder started passing a CUDA stream via `CudaContext(cuda_stream=...)`, which until av 18.1.0 required the PyAV-main wheel above. All TensorRT engines recompiled once.
- **d7a99bd base (verified 2026-07-23)**: upstream's per-language docs, locale split and composition root; no change to the setup.
- **v0.8.1 base (verified 2026-07-19)**: the GPU stack moved into the `nvidia` / `amd` extras (hence `.[dev,nvidia]`), `requires-python` relaxed to `>=3.12`, Linux driver minimum 580.
- **v0.8.0 base (verified 2026-07-18)**: upstream moved the media layer to PyAV (NVDEC/NVENC) and removed `python_vali` / `PyNvVideoCodec` from the runtime. With their native builds went every prerequisite the earlier guide needed: the CUDA Toolkit, cmake / ninja, the `libav*-dev` packages, the `FFMPEG_DIR` prefix, the `setuptools<80` pin and `mkvmerge` (the CUDA Toolkit, cmake / ninja and `libav*-dev` are needed again now that the VALI fork build is a prerequisite, Section 5.2). The Linux GUI fix of Appendix A.2 dates from this era (the `wait_visibility()` handling is upstream's; the 250 ms deferral is the branch's).
