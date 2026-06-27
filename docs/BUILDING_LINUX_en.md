# Building Jasna for Linux

How to set up the Jasna build dependencies on Linux and run jasna **from source**.

> **This guide covers the `v0.7.2+modi` branch.** It builds jasna against the GPU stack (**torch 2.12.0+cu130 / torchvision 0.27.0+cu130 / torch-tensorrt 2.12.0+cu130 / tensorrt 10.16.1.11**), already pinned in `pyproject.toml` on this branch. TensorRT stays on the **10.16** line because `torch-tensorrt==2.12.0` requires `tensorrt>=10.16.1,<10.17.0`; TensorRT 11 is not yet supported by torch-tensorrt.
>
> **New on this branch:** AV1 output, 8-bit (NV12) output, and BT.601/BT.2020 colorspace preservation (see [CODECS_AND_COLORSPACE_en.md](CODECS_AND_COLORSPACE_en.md)); and 2x/4x frame generation via RIFE (see [FRAME_GENERATION_en.md](FRAME_GENERATION_en.md)).

> **Packaging note:** This public fork builds the native GPU dependencies and runs jasna **from source**. It does **not** ship Nuitka packaging tooling to produce a frozen/packaged binary: that tooling is private (the same arrangement as upstream). If you want a pre-packaged binary instead of running from source, use upstream Kruk2/jasna's official releases. See [Packaging / frozen builds](#10-packaging--frozen-builds).

> This guide was verified on **Ubuntu 26.04 LTS** (ships ffmpeg 8, gcc 15, glibc 2.43) with **CUDA 13.3**, an RTX 50-series GPU, and **driver 595**. Other distributions work too, but the package names and ffmpeg/CUDA install steps will differ.

---

## 1. Prerequisites

> ⚠ **Two Linux-specific notes up front:**
> 1. **ffmpeg is taken from system dev packages, not downloaded.** This fork of `vali` does *not* auto-download ffmpeg (the upstream VALI README claims it does, but that does not apply here). You install ffmpeg-8 **dev** libraries via `apt`, and the build links against them. PyNvVideoCodec additionally needs them exposed in a specific directory layout via `FFMPEG_DIR` (Section 5).
> 2. **You must build against the system `python3.13` with its `-dev` headers**, not a self-contained/managed Python. PyNvVideoCodec's pybind11 resolves the Python include dir to `/usr/include/python3.13`, which only exists when `python3.13-dev` is installed. **Ubuntu 26.04's default `python3` is 3.14**: on 3.14 the vali / PyNvVideoCodec include-path resolution and the torch cu130 wheels do not match, so the build/resolve fails (`pyproject.toml` pins `requires-python = ">=3.13,<3.14"` to reject 3.14 outright). If `python3.13` is not in the default repos, add deadsnakes or similar, and **confirm `/usr/bin/python3.13` exists** before creating the venv (`ls -l /usr/bin/python3.13`).

| Category | Requirement | Source | Notes |
|---|---|---|---|
| OS | Ubuntu 26.04 x64 (or similar) | n/a | Verified on 26.04; ships ffmpeg 8 in `apt` |
| Build tools | `build-essential`, `pkg-config` | apt | gcc/g++ 15 is fine: CUDA 13.3's `nvcc` accepts it as host compiler |
| cmake / ninja | any recent | apt **or** the venv (Section 4) | The venv copies are used during `pip install`; system ones are optional |
| Python | **3.13 + 3.13-dev + 3.13-tk** | apt (`python3.13 python3.13-dev python3.13-tk`) | Dev headers are mandatory for vali (`Development.Module`) and PyNvVideoCodec (pybind11). `python3.13-tk` (Tcl/Tk) is required to run the GUI; without it `tkinter`/`customtkinter` fail to import |
| uv | latest | [astral.sh/uv](https://docs.astral.sh/uv/) | Manages the venv |
| CUDA Toolkit | **13.x** (verified 13.3) | NVIDIA apt repo | Toolkit only: does **not** replace your driver. `nvcc` must resolve to 13.x (Section 3) |
| NVIDIA Driver | 590+ (59x series) | your distro / NVIDIA | GPU must be compute capability 7.5+ |
| ffmpeg / ffprobe | **v8** (runtime) + **dev libs** (build) | apt | Runtime CLI must be major version 8; build needs `libav*-dev` (Section 1.1) |
| MKVToolNix | `mkvmerge` (runtime) | apt (`mkvtoolnix`) | Needed only when actually processing video, not for the build itself |

### 1.1 Install system packages

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential pkg-config cmake ninja-build \
  python3.13 python3.13-dev python3.13-tk \
  libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev \
  libavfilter-dev libavdevice-dev \
  ffmpeg mkvtoolnix
```

Install `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.2 Install the CUDA 13 toolkit

The CUDA *toolkit* (`nvcc`, headers, `cudart`) is needed to compile `vali` and `PyNvVideoCodec`. Install the **toolkit only**; it does not touch your existing driver.

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-3
```

This installs to `/usr/local/cuda-13.3`. (For a different Ubuntu version, swap `ubuntu2604` for `ubuntu2404` etc.)

> The `cuda-toolkit-13-3` package deliberately does **not** pull in a driver. Do not install the `cuda` meta-package unless you also want to replace your driver.

---

## 2. Clone the repositories

Place the three repositories **under the same parent directory**. This guide refers to it as `$WORKSPACE`.

```bash
export WORKSPACE="$HOME/jasna-dev"     # ← any directory of your choice
mkdir -p "$WORKSPACE" && cd "$WORKSPACE"

git clone https://codeberg.org/Kruk2/vali.git
git clone https://codeberg.org/Kruk2/PyNvVideoCodec.git
git clone -b modi https://github.com/sh202603/jasna.git   # the modi branch, this fork's default (3 patches pre-applied)

# vali has submodules (extern/dlpack, cmake-modules), initialize them
cd vali && git submodule update --init --recursive && cd ..
```

Layout:

```
$WORKSPACE/
  vali/
  PyNvVideoCodec/
  jasna/          <- the working root from here on
```

---

## 3. Build-environment variables (CUDA)

Set these **in the same shell** you'll build from. The `vali` / `PyNvVideoCodec` builds read them.

```bash
export CUDA_PATH=/usr/local/cuda-13.3
export CUDAToolkit_ROOT=/usr/local/cuda-13.3
export PATH="$CUDA_PATH/bin:$PATH"
```

Verify `nvcc` resolves to 13.x (your distro may also have an older `nvcc` on `PATH`; the prepend above wins):

```bash
nvcc --version            # must show "release 13.x"
which nvcc                 # /usr/local/cuda-13.3/bin/nvcc
```

### 3.0 Build env-var quick reference

| Env var | Value | Consumer | Section |
|---|---|---|---|
| `CUDA_PATH` / `CUDAToolkit_ROOT` | `/usr/local/cuda-13.3` | vali CMake, nvcc | 3 |
| `PATH` (prepend) | `$CUDA_PATH/bin` | shell / nvcc | 3 |
| `VIRTUAL_ENV` | `$WORKSPACE/jasna/.venv` | `uv pip install` target | 4 |
| `FFMPEG_DIR` | `$WORKSPACE/ffmpeg-prefix` | PyNvVideoCodec CMake | 5 / 7 |
| `CUDACXX` | `/usr/local/cuda-13.3/bin/nvcc` | PyNvVideoCodec CMake (CUDA 13 `sm_52` workaround) | 7 |

---

## 4. Create the Python virtual environment

Create a `.venv` under the `jasna` repo, **from the system `python3.13`** (not a managed/standalone Python; see the note in Section 1).

```bash
cd "$WORKSPACE/jasna"
ls -l /usr/bin/python3.13                       # if missing, install python3.13 / python3.13-dev first (Section 1.1)
uv venv --python /usr/bin/python3.13 .venv
export VIRTUAL_ENV="$WORKSPACE/jasna/.venv"
python --version                                # -> Python 3.13.x (recreate if it says 3.14)
```

> ⚠ **Always pass `--python` as a path (`/usr/bin/python3.13`).** With a version request (`--python 3.13`) or no `--python`, on a box without `/usr/bin/python3.13` uv may fetch a managed CPython (latest = **3.14**) or pick up Ubuntu 26.04's default `python3` (3.14). If `python --version` reports 3.14, `rm -rf .venv` and recreate it with the explicit path.

Install the build-side tools into the venv (so `--no-build-isolation` picks them up):

```bash
uv pip install cmake ninja scikit-build "setuptools<80" wheel numpy
```

> **Why `setuptools<80`**: `setuptools` ≥ 80 no longer ships `pkg_resources`, and `vali` / `PyNvVideoCodec` import `pkg_resources` in their `setup.py`. Pin it below 80 (resolves to ~79.0.1).

---

## 5. Expose ffmpeg-8 to the build

`vali` finds the system ffmpeg-8 dev libraries automatically (CMake's default search picks up `/usr/include/x86_64-linux-gnu` + `/usr/lib/x86_64-linux-gnu`), so it needs no extra configuration; the `apt` packages from Section 1.1 are enough.

**PyNvVideoCodec is different.** Its CMake expects an ffmpeg prefix with `include/` and `lib/x86_64/` subdirectories, which does **not** match Ubuntu's multiarch layout. Build a small symlink prefix that points at the system ffmpeg-8 files (keeping headers and libraries the same version):

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

## 6. Build and install vali

> **Prerequisites**: Section 3 (CUDA env) + Section 4 (venv + `setuptools<80`) + Section 1.1 (ffmpeg dev libs).

`vali`'s own `setup.py` already passes `-DCMAKE_CUDA_ARCHITECTURES=native` and pins `nvcc` from `CUDA_PATH`, so it configures and builds cleanly:

```bash
cd "$WORKSPACE/vali"
uv pip install . --no-build-isolation
```

The build bundles the ffmpeg `.so` files into the installed package. Verify the import (when torch isn't loaded yet, point the loader at the CUDA runtime libs):

```bash
LD_LIBRARY_PATH=/usr/local/cuda-13.3/targets/x86_64-linux/lib \
  python -c "import python_vali; print('vali OK', python_vali.__version__)"
```

> At Jasna runtime you don't need `LD_LIBRARY_PATH`: torch (cu130) loads its bundled CUDA 13 runtime first, which satisfies `vali`/`PyNvVideoCodec`. The `LD_LIBRARY_PATH` above is only for importing the native libs standalone.

---

## 7. Build and install PyNvVideoCodec

> **Prerequisites**: same as Section 6, plus the `FFMPEG_DIR` prefix (Section 5).

Two extra environment variables are required on a CUDA-13 system:

- **`FFMPEG_DIR`**: the symlink prefix from Section 5.
- **`CUDACXX`**: pre-sets `CMAKE_CUDA_COMPILER` so CMake skips its compiler-identification probe. That probe hardcodes `-arch=sm_52`, which **CUDA 13 removed** (`ptxas fatal: Value 'sm_52' is not defined`). Pre-setting the compiler avoids the probe; the project's default architecture list (`75;80;86;89;120`) is valid for CUDA 13.

```bash
cd "$WORKSPACE/PyNvVideoCodec"
export FFMPEG_DIR="$WORKSPACE/ffmpeg-prefix"
export CUDACXX=/usr/local/cuda-13.3/bin/nvcc
uv pip install . --no-build-isolation
```

Verify:

```bash
LD_LIBRARY_PATH=/usr/local/cuda-13.3/targets/x86_64-linux/lib \
  python -c "import PyNvVideoCodec as p; print('PyNvVideoCodec OK', p.__version__)"
```

> If you change settings and rebuild, wipe the stale cache first: `rm -rf _skbuild`.

---

## 8. Install jasna itself

> **Prerequisites**: `python_vali` (Section 6) and `PyNvVideoCodec` (Section 7) installed in the venv.

`pyproject.toml` depends on `torch==2.12.0+cu130` / `torchvision==0.27.0+cu130` / `torch-tensorrt==2.12.0`, which are not on the default PyPI. Point uv at the PyTorch cu130 index and add two flags:

```bash
cd "$WORKSPACE/jasna"
uv pip install -e .[dev] \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

Why each flag:

- `--extra-index-url https://download.pytorch.org/whl/cu130`: source of the `torch+cu130` / `torch-tensorrt+cu130` wheels.
- `--index-strategy unsafe-best-match`: lets uv satisfy `torch-tensorrt==2.12.0` with the index's local-version release `2.12.0+cu130`.
- `--prerelease=allow`: a transitive dep (`nvidia-cuda-runtime-cu13`) is a pre-release.

The `[dev]` extra installs `nuitka>=2.4`, `pytest`, `pytest-cov`, `scikit-build`, `cmake`, `ninja`.

**Optional: the torchcodec backend.** For the experimental torchcodec decode/encode path (`--video-backend torchcodec`/`auto`), add the `torchcodec` extra, i.e. install `.[dev,torchcodec]` with the same flags:

```bash
uv pip install -e .[dev,torchcodec] \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

This adds `torchcodec>=0.14.0`. It is **not** needed for a normal build; the default `native` backend works without it. See [TORCHCODEC_BACKEND_en.md](TORCHCODEC_BACKEND_en.md).

### 8.2 ONNX packages (for YOLO detection models)

If you use a **YOLO (lada-yolo-\*) detection model**, ultralytics exports it to ONNX before building the TensorRT engine, which needs three packages that are **not** pulled in automatically. (RF-DETR models do not need these: they parse a prebuilt `.onnx` via TensorRT directly.)

```bash
uv pip install onnx onnxslim onnxruntime
```

When running from source, ultralytics auto-downloads these on first export if they are missing, but installing them ahead of time avoids a stall on the first YOLO run.

Smoke test:

```bash
python -m jasna --version     # -> 0.7.2+modi
python -m jasna --help
```

### 8.1 Apply the mmengine patch (torch 2.6+ compatibility)

Adds `weights_only=False` to the `torch.load` call inside `mmengine.runner.checkpoint`. Since torch 2.6 the default flipped to `weights_only=True`, which breaks loading the existing `.pth` checkpoint. The diff ships in `patches/`.

```bash
cd "$WORKSPACE/jasna"
patch -p1 -d .venv/lib/python3.13/site-packages \
    < patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff

# verify
grep -n "weights_only=False" .venv/lib/python3.13/site-packages/mmengine/runner/checkpoint.py
```

> Re-running `uv pip install -e .[dev]` reinstalls `mmengine` and clobbers this patch. Re-apply afterwards.

---

## 9. Place model weights and assets

Drop the following 3 files into `$WORKSPACE/jasna/model_weights/`:

- `lada_mosaic_restoration_model_generic_v1.2.pth`
- `rfdetr-v5.onnx`
- `lada_mosaic_detection_model_v4_fast.pt`

When running from source, jasna resolves `model_weights/` automatically (Appendix A.1), so the files just need to live here.

The 2 test clips usually ship with the repository under `$WORKSPACE/jasna/assets/` (`test_clip1_1080p.mp4`, `test_clip1_2160p.mp4`); if they are absent, extract them from an upstream release.

### 9.1 Optional: RIFE frame-interpolation model (for `--frame-gen`)

Only needed for frame-rate up-conversion (`--frame-gen {2x,4x}`). The RIFE weights are **not bundled** (they carry non-commercial terms), so you create a TorchScript checkpoint yourself with `make_rife_torchscript.py`.

1. Clone Practical-RIFE and download a 4.x model package (verified with **v4.25**) so that `<repo>/train_log/` contains `RIFE_HDv3.py`, `IFNet_HDv3.py`, `flownet.pkl`:
   ```bash
   git clone https://github.com/hzwer/Practical-RIFE
   # download + unzip the model package into Practical-RIFE/train_log/ per its README
   ```
2. Convert to TorchScript using the project venv (run from `$WORKSPACE/jasna`):
   ```bash
   .venv/bin/python make_rife_torchscript.py \
       --rife-repo /path/to/Practical-RIFE \
       --output model_weights/rife.pth --validate
   ```

Running **from source** picks up `model_weights/rife.pth` automatically (same resolver as the other weights; or point `JASNA_MODEL_WEIGHTS_DIR` at a folder containing it). Full procedure: [docs/FRAME_GENERATION_en.md](FRAME_GENERATION_en.md).

---

## 10. Packaging / frozen builds

There is currently **no public way to produce a packaged/frozen binary from this fork.** Upstream switched its build from PyInstaller to **Nuitka**, but it does not publish any Nuitka build script or instructions; the actual packaging tooling lives in a **private submodule (`jasna/protection`)** that is not part of this public fork. The fork's old PyInstaller build scripts (`build_exe.py`, `jasna.spec`) have been **removed**.

The supported public path is therefore to **run jasna from source** (Section 11). If you need a pre-packaged binary, use upstream Kruk2/jasna's official releases.

---

## 11. Run from source / verify

With Sections 1–9 complete, run jasna directly from the source checkout in the venv:

```bash
cd "$WORKSPACE/jasna"
jasna --version              # -> 0.7.2+modi
jasna --help
jasna --input assets/test_clip1_1080p.mp4 --output /tmp/out.mp4   # process a short clip
jasna                        # launches the GUI (no args)
```

`model_weights/` is resolved automatically (Appendix A.1), and `ffmpeg`/`ffprobe` (v8) plus `mkvmerge` are taken from the system `PATH` (Section 1.1).

> **First processing run is slow.** TensorRT engines are compiled for your GPU on first use (15–60 min). They are cached next to the weights in `model_weights/` and reused on later runs.

---

## Troubleshooting

### Environment / Python

- **vali configure fails: `Could NOT find Python3 (missing: ... Development.Module)`**
  The Python development headers are missing. Install `python3.13-dev` and create the venv from `/usr/bin/python3.13` (Sections 1.1, 4). Do **not** use a managed/standalone Python here.

- **PyNvVideoCodec generate fails: `pybind11::module includes non-existent path /usr/include/python3.13`**
  Same cause: `python3.13-dev` not installed, or the venv was built from a non-system Python. Install `python3.13-dev` and rebuild the venv from `/usr/bin/python3.13`.

- **`vali` / `PyNvVideoCodec` build fails: `ModuleNotFoundError: No module named 'pkg_resources'`**
  `setuptools` ≥ 80 dropped `pkg_resources`. `uv pip install "setuptools<80"` in the venv (Section 4), then rebuild.

- **Launching the GUI fails with `ModuleNotFoundError: No module named 'tkinter'`**
  The GUI needs Tcl/Tk, which is the `python3.13-tk` package (note: `python3-tk` targets a different interpreter). Install it for the same `python3.13` the venv was built from: `sudo apt-get install -y python3.13-tk`. The CLI works without it; only the GUI requires it.

### CUDA

- **`nvcc --version` shows the wrong version**
  Another CUDA is earlier on `PATH`. Re-export `PATH="$CUDA_PATH/bin:$PATH"` (Section 3) and check `which nvcc`.

- **PyNvVideoCodec configure fails: `ptxas fatal: Value 'sm_52' is not defined for option 'gpu-name'`**
  CUDA 13 removed the Maxwell `sm_52` target, but CMake's compiler-ID probe hardcodes it. Pre-set the CUDA compiler so the probe is skipped: `export CUDACXX=/usr/local/cuda-13.3/bin/nvcc` (Section 7).

- **gcc is "too new"**
  CUDA 13.3's `nvcc` accepts gcc 15 (verified). If a future toolkit rejects your host gcc, install `gcc-14 g++-14` and add `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14` (or `export NVCC_PREPEND_FLAGS='-ccbin g++-14'`).

### ffmpeg

- **vali configure fails: `AVCODEC_LIB ... NOTFOUND`**
  The ffmpeg `-dev` packages aren't installed. Install `libav{codec,format,util}-dev libsw{resample,scale}-dev` (Section 1.1).

- **PyNvVideoCodec fails to find ffmpeg / wrong headers**
  `FFMPEG_DIR` isn't set, or its layout is wrong. Rebuild the symlink prefix with `include/` and `lib/x86_64/` (Section 5) and `export FFMPEG_DIR=...` before building.

### jasna install

- **`uv pip install -e .[dev]` fails: `no version of torch==2.12.0+cu130`**. Add `--extra-index-url https://download.pytorch.org/whl/cu130`.
- **`torch-tensorrt==2.12.0+cu130 ... unsatisfiable`**. Add `--index-strategy unsafe-best-match --prerelease=allow` (Section 8).

### Runtime

- **`import python_vali` fails standalone with a CUDA `.so` load error**
  Either run after `import torch` (which loads the bundled CUDA 13 runtime), or set `LD_LIBRARY_PATH=/usr/local/cuda-13.3/targets/x86_64-linux/lib`. Inside Jasna this is automatic.

- **Jasna fails with `FileNotFoundError` for a `.pth`/`.onnx`/`.pt`**
  A file in `model_weights/` is missing, or the resolver looked in the wrong place. Make sure the 3 weight files are in `$WORKSPACE/jasna/model_weights/` (Section 9), or set `JASNA_MODEL_WEIGHTS_DIR` to the folder that holds them.

- **Jasna refuses to start: ffmpeg/ffprobe wrong version, or mkvmerge missing**
  Jasna requires `ffmpeg`/`ffprobe` **major version 8** and `mkvmerge` on `PATH`. Install via `apt` (Section 1.1).

- **RTX Super-Res fails with `IRuntime::deserializeCudaEngine ... Serialization assertion stdVersionRead == kSERIALIZATION_VERSION failed` / `Version tag does not match`**
  The `nvidia-vfx` (nvvfx) package bundles its own TensorRT (10.9) and loads it with `RTLD_GLOBAL`, while jasna's engines are built with TensorRT 10.16 (`tensorrt_libs`). Both share the soname `libnvinfer.so.10`; if nvvfx's loads first, `torch-tensorrt` binds to 10.9 and can't read the 10.16 engines. The fix is already applied on this branch (see Appendix B.2): it pre-loads `tensorrt_libs`' `libnvinfer.so.10` with `RTLD_GLOBAL` before nvvfx is imported, so the 10.16 symbols win.

---

## Appendix A: build/runtime changes on this branch

A set of changes already applied to this branch's source (documented here for reference; no standalone `.patch` file is shipped). They are cross-platform; the Windows-specific parts are guarded by `sys.platform == "win32"` and are inert on Linux.

### A.1 Auto-resolution of the `model_weights/` directory

Adds `jasna/model_weights_resolver.py`, which locates `model_weights/` in priority order: `JASNA_MODEL_WEIGHTS_DIR` env var → next to the executable → current working directory → beside the source tree. `main.py`, `mosaic/detection_registry.py`, `engine_paths.py`, and the GUI (`gui/processor.py`, `gui/engine_preflight.py`) go through the resolver instead of a hardcoded `Path("model_weights")`. This lets the CLI run from any directory.

### A.2 DLL load helper (`jasna/media/video_decoder.py`)

Registers CUDA / `python_vali` directories on the DLL search path **on Windows only** (`if sys.platform == "win32"`). No-op on Linux.

### A.3 BasicVSR++ benchmark TRT-API fix

Updates `jasna/benchmark/basicvsrpp_restoration.py` to the newer `_preprocess_engine` API so `--benchmark basicvsrpp` works. Cross-platform.

---

## Appendix B: Linux GUI / RTX-VSR fixes on this branch

The Linux-specific runtime fixes, already applied to this branch's source (documented here for reference; no standalone `.patch` file is shipped). Both are required for the GUI and RTX Super-Res to work on Linux/X11; neither affects the Windows build.

### B.1 Blank modal dialogs (`jasna/gui/app.py`, `jasna/gui/wizard.py`, `jasna/gui/components.py`)

**Problem**: On Linux/X11, customtkinter `CTkToplevel` dialogs opened completely blank (an empty dark window with no text or buttons). Affected dialogs: **About** (`app.py` `_show_about`), **System Check** / first-run wizard (`wizard.py` `FirstRunWizard`), and the **preset-create** and **confirmation** dialogs (`components.py` `PresetDialog`, `ConfirmDialog`). Each calls `grab_set()` (and sometimes `lift()` / `focus_force()`) immediately on creation, before its child widgets are drawn. On some window managers this leaves the window mapped but unpainted, and because the dialogs are non-resizable the content never receives a redraw.

**Fix**: Create all child widgets first, then defer `lift()` / `grab_set()` / `focus_force()` to a later event-loop tick via `self.after(200, …)` / `after(250, …)`, so the modal grab is established only after the content has been painted. Windows behavior is unchanged (the same calls run, just a tick later).

### B.2 RTX Super-Res TensorRT version conflict (`jasna/restorer/rtx_superres_secondary_restorer.py`)

**Problem**: Enabling RTX Super-Res aborted the run while deserializing jasna's TensorRT engines with `IRuntime::deserializeCudaEngine ... Serialization assertion stdVersionRead == kSERIALIZATION_VERSION failed. Version tag does not match`. The `nvidia-vfx` (nvvfx) package bundles its own TensorRT **10.9** (`nvvfx/libs/libnvinfer.so.10`) and loads it with `RTLD_GLOBAL` in `nvvfx/_lib_loader.py`, while jasna's pipeline engines are built with TensorRT **10.16** (`tensorrt_libs`). Both libraries share the soname `libnvinfer.so.10`, and ELF symbol resolution uses whichever object entered the global scope first. When nvvfx loaded before jasna's TensorRT runtime, `torch-tensorrt` bound to nvvfx's older 10.9 and could not read jasna's newer 10.16-built engines (the two emit different serialization version tags). The Windows build avoids this via DLL load ordering (`tensorrt_libs` first); Linux had no equivalent safeguard.

**Fix**: Add `_preload_tensorrt_runtime()` (named after upstream `6545b78`, which later implemented the same fix independently), executed at module import of the RTX Super-Res restorer (before any `nvvfx` import). On Linux it locates `tensorrt_libs` and loads its `libnvinfer.so.10` / `libnvinfer_plugin.so.10` with `ctypes.RTLD_GLOBAL`, so TensorRT 10.16's symbols occupy the global scope first and nvvfx's later load resolves to 10.16. No-op on Windows (ordering is already handled by the bundled DLL-path logic).
