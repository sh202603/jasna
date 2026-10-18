# Building Jasna for Windows

How to set up Jasna on Windows and run it **from source**.

> **Branch and base.** This guide covers `0.11.2+modi` (upstream `b623466f` = the v0.11.2 tag). The GPU stack follows upstream's pins in `pyproject.toml` (torch 2.12.1+cu130 / torchvision 0.27.1+cu130 / torch-tensorrt 2.12.0 / tensorrt 10.16.1.11 / comfy-kitchen 0.2.38 / triton-windows 3.7.1 / ultralytics 8.4.96+); this branch adds the `nvidia-vfx==0.2.0.0` pin (upstream leaves it unpinned), `nvidia-cudnn-frontend` and, on Windows, `triton-windows>=3.7,<3.8` as regular dependencies for FP8 restoration, the `torchcodec` extra, and the optional `nvidia-rtx` extra (torch 2.12.0 / torchvision 0.27.0 / torch-tensorrt-rtx 2.12.1). What the branch adds on top of upstream, and why, is in [changes_vs_upstream.md](changes_vs_upstream.md).
>
> **Verification.** The `0.11.2+modi` base has **not been re-verified on Windows yet**; the v0.11.2 delta changes no pin except `comfy-kitchen` (0.2.36 to 0.2.38) and adds a hard-disk install warning whose Windows side (an `IOCTL_STORAGE_QUERY_PROPERTY` seek-penalty query) only a Windows run can exercise. The previous `0.11.1+modi` base was verified on real hardware on 2026-10-16 (Windows 11, RTX 5060 Ti 16 GB, driver 617.42, Python 3.13.9, CUDA 13.2): a v0.10.0-era venv updated with this guide, the full test suite compared against pristine upstream, CLI runs of the modi features, and the GUI. What changed at each base, and each verification, is in [Appendix B](#appendix-b-history-of-this-setup); for Linux see [building_linux.md](building_linux.md).
>
> **VALI is required.** On Windows you build Kruk2's VALI fork (`python_vali` 4.8.8) yourself and install it (Section 5.2). The `[nvidia]` extra pins `python_vali==4.8.8`, which is not published on PyPI (the newest release there is 4.8.7, with Linux wheels only, as of 2026-10-16), and jasna's VALI decoder uses an API only the fork has. This makes VS Build Tools and the CUDA Toolkit prerequisites.
>
> **Packaging.** This fork is run from source. Its experimental Nuitka script (`scripts\build_nuitka.py`) builds a standalone `jasna.exe` + `jasna-framegen.exe` distribution from the venv of this guide ([frozen_build.md](frozen_build.md), Section 8). LTX and the supporter models work only in upstream's official release.

---

## 1. Prerequisites

| Category | Requirement | winget ID / Source | Notes |
|---|---|---|---|
| OS | Windows 10/11 x64 | n/a | PowerShell 7+ recommended |
| Git | Git for Windows | `Git.Git` | For cloning the repository |
| uv | Latest | `astral-sh.uv` | Manages Python automatically |
| Python | 3.13 | n/a (managed by uv) | Auto-fetched by `uv venv --python 3.13` if missing. `requires-python` is `>=3.12`, but this guide is verified on 3.13 only (3.14 is unverified against the cu130 GPU stack) |
| NVIDIA Driver | **610+** | NVIDIA official or the NVIDIA App | The startup check requires 610+ on Windows. GPU must be compute capability 7.5+ |
| ffmpeg / ffprobe | **v8** shared build | [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases), `ffmpeg-n8.1-latest-win64-gpl-shared-*.zip` recommended (see below) | The runtime CLI (the startup check requires `ffprobe` major version 8) and the `include\` / `lib\` / `bin\` of the VALI build |
| VS Build Tools | 2022 (Desktop development with C++) | `Microsoft.VisualStudio.2022.BuildTools` | **Required**, for the VALI fork wheel build (Section 5.2); also used by the Nuitka bundle ([frozen_build.md](frozen_build.md)) |
| CUDA Toolkit | 13.x (verified with 13.2) | NVIDIA's installer | **Required**, for `nvcc` in the VALI build; not used at runtime |

The only thing compiled is the VALI fork wheel (Section 5.2). The torch / tensorrt wheels bundle the CUDA runtime and PyAV comes as a wheel too, so the CUDA Toolkit is needed only for `nvcc` at build time.

### 1.1 Choosing ffmpeg and adding it to PATH

For the CLI alone (`ffmpeg.exe` / `ffprobe.exe`), gyan.dev's [release-full build](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z) suffices (it is also what the startup check's error message points to), but the VALI build needs the `include\` / `lib\` / `bin\` of a shared build. The BtbN **shared build** is recommended because the same extraction covers all three:

- the runtime CLI (`ffmpeg.exe` / `ffprobe.exe`)
- the `include\` and `lib\` the VALI fork build links against (Section 5.2), and the `bin\` DLLs it bundles into the wheel
- the FFmpeg DLLs (`avcodec-62.dll` etc. in `bin\`) that the optional torchcodec backend needs at runtime

Extract it and add `bin\` to PATH:

```powershell
$Workspace = "C:\jasna-dev"          # ← any directory of your choice
# Download ffmpeg-n8.1-latest-win64-gpl-shared-*.zip from
# https://github.com/BtbN/FFmpeg-Builds/releases and extract to $Workspace\ffmpeg-n8.1-shared

$env:PATH = "$Workspace\ffmpeg-n8.1-shared\bin;$env:PATH"
ffmpeg -version                      # should report ffmpeg version n8.1 (major 8)
ffprobe -version
```

To make it permanent, add the `bin\` directory to the user `Path` environment variable (effective in new sessions).

### 1.2 One-shot install via winget

```powershell
winget install --id Git.Git       -e --source winget
winget install --id astral-sh.uv  -e --source winget

# for the VALI fork wheel build (Section 5.2):
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --silent `
  --override "--wait --quiet --add ProductLang En-us --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Install the NVIDIA driver (610+) from NVIDIA directly or via the NVIDIA App. Install CUDA Toolkit 13.x with NVIDIA's installer (default location `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`).

---

## 2. Clone the repository

Clone `jasna` here; the VALI fork is cloned in Section 5.2.

```powershell
$Workspace = "C:\jasna-dev"          # ← any directory of your choice
mkdir $Workspace -Force | Out-Null
cd $Workspace

git clone -b modi https://github.com/sh202603/jasna.git   # modi branch (this fork's default)
```

> In a new PowerShell session, set `$Workspace = "..."` again.

---

## 3. Create the Python virtual environment

Create `.venv` under the `jasna` repository. Run all subsequent `uv pip` commands with this venv activated.

```powershell
cd $Workspace\jasna
uv venv --python 3.13
.\.venv\Scripts\Activate.ps1
python --version                     # -> Python 3.13.x
```

uv's managed Python is fine here.

---

## 4. PyAV

Nothing to build. The GPU path needs PyAV's current_ctx API (NVDEC/NVENC share the CUDA context torch initialized) and the explicit CUDA stream support of av 18.1; `pyproject.toml` pins `av==18.1.0` (the release upstream's VALI wheel is built against), and the Section 5 install resolves it from PyPI. The wheel bundles its own FFmpeg 8 built with new-enough nv-codec-headers (`hevc_nvenc` accepts `lookahead_level`), so neither the BtbN build nor a delvewheel repair is involved; the BtbN `bin\` on PATH serves only the CLI tools and the optional torchcodec backend.

The official wheel is verified on Linux and on Windows (2026-10-16). The v0.10.0-era Windows setup used the interim self-built wheel, whose commit is contained in 18.1.0; that era, and how to tell whether a venv still has the interim wheel, is in [Appendix B](#appendix-b-history-of-this-setup).

---

## 5. Install jasna itself

On Windows, install in this order. The `[nvidia]` extra pins `python_vali==4.8.8`, but 4.8.8 is not on PyPI, so first build the wheel from Kruk2's fork ([codeberg.org/Kruk2/vali](https://codeberg.org/Kruk2/vali)), then hand it to the jasna install with `--find-links`.

1. Install the packages the VALI build needs into the venv (Section 5.1)
2. Build the VALI fork wheel (Section 5.2)
3. Install jasna itself (Section 5.3)
4. Apply the mmengine patch (Section 5.4)

### 5.1 Install the build packages

```powershell
cd $Workspace\jasna
uv pip install cmake ninja scikit-build "setuptools<81"
```

A venv created with `uv venv` has no setuptools. vali's `setup.py` imports `pkg_resources`, and setuptools 80.x answers that import with a warning that `pkg_resources` is slated for removal and to pin `Setuptools<81`, so pin `<81` (checked with 80.10.2). cmake, ninja and scikit-build also come with the `[dev]` extra in Section 5.3, but the build runs first, so install them here.

### 5.2 Build the VALI fork wheel

Native decode tries the VALI NVDEC decoder first and escalates to PyAV when it is unavailable (the `JASNA_DECODE_BACKEND` environment variable — `auto` / `vali` / `pyav-hw` / `pyav-sw` — selects this chain explicitly; modi's `--video-backend` sits one layer above it, and the chain applies when `native` is selected). jasna's VALI decoder uses `DecodeSingleSurfaceAsyncDetailed`, an API that only the fork has; with a wheel that lacks it, the reader logs a warning and falls back to PyAV every time.

Prerequisites (installed in Section 1):

- **VS Build Tools 2022** (build inside a cmd session that went through vcvars64)
- **CUDA Toolkit 13.x** (`nvcc`; vali's CMake resolves the toolkit from `CUDA_PATH`)
- **An FFmpeg 8 shared build**: vali's CMake **hardcodes `FFMPEG_ROOT` to `C:\Program Files\ffmpeg8`** (`src\CMakeLists.txt`). Place a shared build with `include\` / `lib\` / `bin\` there, or create a junction to the Section 1.1 extraction (in an elevated PowerShell: `New-Item -ItemType Junction -Path "C:\Program Files\ffmpeg8" -Target $Workspace\ffmpeg-n8.1-shared`).
- Keep the workspace path short. A build in a deep folder (a vali checkout path of about 150 characters) failed in CMake's compiler check with MSBuild's `error FTK1011` (it could not create its tracking log); the same checkout mapped to a short drive with `subst` built fine.

```powershell
cd $Workspace
git clone https://codeberg.org/Kruk2/vali.git       # skip if you have a checkout
cd vali
git checkout 0e5c01e      # = the 4.8.8 pin: DecodeSingleSurfaceAsyncDetailed, the ASF/VC-1 timestamp fixes, the decoder-open GIL release
git submodule update --init --recursive
```

When an existing checkout moves to another commit, delete the previous CMake cache `_skbuild\` before rebuilding. CMake downloads pybind11 during the build, so it needs network access.

Build by running `setup.py bdist_wheel` inside a **cmd session** with the MSVC + CUDA environment loaded (a .bat file keeps vcvars64 out of your PowerShell session):

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CL=/utf-8"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d <workspace>\vali
<workspace>\jasna\.venv\Scripts\python.exe setup.py bdist_wheel
REM -> dist\python_vali-4.8.8-cp313-cp313-win_amd64.whl
```

Unlike on Linux, the Windows wheel bundles the FFmpeg DLLs inside `python_vali\` (CMake's `PYPI_BUILD=0` default), so no delvewheel step is needed. Section 5.3 installs the wheel through `--find-links`.

The first part of the log shows `Trying 'Ninja (Visual Studio 18 2026 x64 v145)' generator - failure`, `error MSB8020` and `Configuring incomplete, errors occurred!` a few times. That is scikit-build probing the available generators; as long as a wheel ends up in `dist\`, the build succeeded (checked on 2026-10-16 with the minimal venv of Section 5.1: the build and the presence of `DecodeSingleSurfaceAsyncDetailed`).

> Fork commit `0e5c01e` (= the 4.8.8 pin) is verified on Linux and Windows (2026-10-16; on Windows, RTX 5060 Ti + CUDA 13.2, the batch file above builds it as is, and the video streams from `JASNA_DECODE_BACKEND=vali` and `pyav-hw` are md5-identical). The earlier `f4a67f8` (also the 4.8.8 pin) was verified on the same machine on 2026-08-13. The previous verification was at fork commit `3ad0d54` (= the 4.8.7 pin, 2026-07-30, RTX 5080 + CUDA 13.2 + gyan.dev 8.1 shared): both decode paths of the pipeline go through VALI, the bundled test clip's output is **bitstream-md5-identical** to the PyAV-decoded run, and coexistence with the av wheel and the torchcodec backend is confirmed (av vendors FFmpeg via delvewheel, vali bundles its own copy, torchcodec resolves via PATH — three separate in-process DLL copies, which is fine on Windows because each DLL has its own symbol namespace).

### 5.3 Install jasna itself

The GPU stack lives in extras (`nvidia` = the NVIDIA stack, `amd` = the ROCm one). For an NVIDIA build the `nvidia` extra pulls in `torch==2.12.1+cu130` / `torchvision==0.27.1+cu130` / `tensorrt==10.16.1.11` / `torch-tensorrt==2.12.0` / `nvidia-vfx==0.2.0.0` / `comfy-kitchen==0.2.38` / `triton-windows==3.7.1.post27` / `python_vali==4.8.8`, some of which are not on the default PyPI index. **Point uv at the PyTorch wheel index and at the Section 5.2 wheel, and add two more flags**:

```powershell
cd $Workspace\jasna
uv pip install -e .[dev,nvidia] `
    --find-links $Workspace\vali\dist `
    --extra-index-url https://download.pytorch.org/whl/cu130 `
    --index-strategy unsafe-best-match `
    --prerelease=allow
```

Why each flag:

- `--find-links $Workspace\vali\dist`: source of the `python_vali-4.8.8` wheel built in Section 5.2. Without it, `python_vali==4.8.8` cannot be resolved and the install fails
- `--extra-index-url https://download.pytorch.org/whl/cu130`: source of the `torch+cu130` / `torchvision+cu130` / `torch-tensorrt+cu130` wheels
- `--index-strategy unsafe-best-match`: needed so `torch-tensorrt==2.12.0` from `pyproject.toml` can be satisfied by `torch-tensorrt==2.12.0+cu130` (local version) on the PyTorch index; the default first-index strategy refuses
- `--prerelease=allow`: the transitive dependency `nvidia-cuda-runtime-cu13==0.0.0a0` is a prerelease

The `[dev]` extra installs `nuitka>=2.4`, `pytest`, `pytest-cov`, `scikit-build`, `cmake`, `ninja`. The `[nvidia]` extra installs the GPU stack (torch cu130 / TensorRT / torch-tensorrt / nvidia-vfx / VALI) — **omitting it leaves you without torch and the app will not start**.

Check that the fork build of VALI is the one installed:

```powershell
python -c "import python_vali as v; print(hasattr(v.PyDecoder, 'DecodeSingleSurfaceAsyncDetailed'))"   # -> True
# running jasna with --log-level info prints "Using VALI NVDEC decoder for <file>"
```

If it prints `False`, a non-fork wheel got installed. Reinstall the Section 5.2 wheel:

```powershell
uv pip install --force-reinstall --no-deps (Get-Item $Workspace\vali\dist\python_vali-4.8.8-*.whl)
```

**Note: `nvidia-vfx` is pinned to 0.2.0.0 (VFX SDK 1.3.0).** The PyPI entry is a stub that downloads the real wheel (436 MB for Windows x64) from `pypi.nvidia.com` during the install above. 0.2.0.0 provides the `VideoFrameGeneration` effect behind `--frame-gen-backend rtx` ([frame_generation.md](frame_generation.md)), bundles no TensorRT, and ships its own `cudart64_12.dll` next to torch's CUDA 13 runtime; the wheel fetch, the DLL loading, the cudart 12/13 coexistence and the cuDNN combination with `--fp8-recon` are verified on Windows (RTX 5080). An older venv with 0.1.0.1 is upgraded in place with `uv pip install nvidia-vfx==0.2.0.0` ([Appendix B](#appendix-b-history-of-this-setup)).

**Optional: the torchcodec backend.** To use the experimental torchcodec decode/encode path (`--video-backend torchcodec`/`auto`), add the `torchcodec` extra and install `.[dev,nvidia,torchcodec]` with the same flags (the torchcodec on PyPI is the CPU-only build; `--extra-index-url` is what lets uv pick the `+cu130` build from the PyTorch index):

```powershell
uv pip install -e .[dev,nvidia,torchcodec] `
    --find-links $Workspace\vali\dist `
    --extra-index-url https://download.pytorch.org/whl/cu130 `
    --index-strategy unsafe-best-match `
    --prerelease=allow
```

This installs `torchcodec>=0.15.0`. torchcodec needs the FFmpeg DLLs at runtime, so keep the BtbN shared build's `bin\` on PATH (Section 1.1). It is not needed for a normal setup; the default `native` backend (PyAV) works without torchcodec. Details: [torchcodec_backend.md](torchcodec_backend.md).

**Note: the FP8 restoration backend needs no extra install steps.** Its dependencies `nvidia-cudnn-frontend` and (on Windows) `triton-windows` are regular dependencies in `pyproject.toml` and come in with the command above. The cuDNN runtime (9.17+) ships inside the torch cu130 wheels. The feature is a runtime opt-in (`--fp8-recon`, needs an FP8-capable GPU, sm89+) and falls back to the TensorRT engines where unavailable. Verified on Windows 11 + RTX 5080; run with `--log-level info` to see `CudnnFP8Upsample: enabled` (the default `--log-level error` hides that line). Details: [fp8_recon.md](fp8_recon.md).

**Optional: the TensorRT-RTX flavor.** Installing the `nvidia-rtx` extra *instead of* `nvidia` switches the TensorRT stack to TensorRT-RTX (JIT compilation; engine builds finish in seconds instead of minutes). The two extras are mutually exclusive in one venv, so use a dedicated venv. Engines are cached under `.rtx`-tagged names (`.rtx.win`), so both flavors can share one `model_weights` directory. The `nvidia-rtx` extra pins torch 2.12.0 / torchvision 0.27.0, unlike the `nvidia` extra's 2.12.1 / 0.27.1. This venv also needs the Section 5.1 build packages (if you build the VALI wheel per venv), the same `--find-links` as Section 5.3 (it pins `python_vali==4.8.8` too), and the Section 5.4 mmengine patch. Details and measured numbers (validated on Windows 11 / RTX 5060 Ti): [tensorrt_rtx.md](tensorrt_rtx.md).


### 5.4 Apply the mmengine patch

Apply `patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff` to the venv's mmengine 0.10.7. The diff changes four files:

- `mmengine/runner/checkpoint.py`: adds `weights_only=False` to `torch.load`. From torch 2.6 the default is `weights_only=True`, which cannot read the mmengine `HistoryBuffer` stored in the BasicVSR++ weights (`lada_mosaic_restoration_model_generic_v1.2.pth`) and stops with `Weights only load failed`
- `mmengine/model/__init__.py` and `mmengine/model/wrappers/__init__.py`: import the FSDP wrapper only when `torch.distributed.is_available()` (so mmengine imports on a torch without distributed support)
- `mmengine/dist/dist.py`: adds `from __future__ import annotations`

> Upstream changed that one line to `weights_only=True` in `7c64105`, but with that version a source run cannot read the `v1.2.pth` above (checked 2026-10-16). This fork ships the diff with `False` ([changes_vs_upstream.md](changes_vs_upstream.md), Section 8).

Apply it with the GNU patch that ships with Git for Windows (with the venv activated, in `$Workspace\jasna`). Do not use the Python `patch` package (1.16): it silently skips the `model/__init__.py` hunk, which is one line off.

```powershell
cd $Workspace\jasna

& "C:\Program Files\Git\usr\bin\patch.exe" -p1 --no-backup-if-mismatch -d .venv\Lib\site-packages `
    -i "$PWD\patches\fix_loading_mmengine_weights_on_torch26_and_higher.diff"
```

`-d` changes the directory first, so pass `-i` an absolute path. `Hunk #2 succeeded at 34 (offset 1 line).` for `model/__init__.py` is expected.

Verify:

```powershell
Select-String -Path .venv\Lib\site-packages\mmengine\runner\checkpoint.py -Pattern "weights_only=False"
Select-String -Path .venv\Lib\site-packages\mmengine\model\__init__.py -Pattern "is_available"
# → one matching line each means it's applied
```

> Reinstalling mmengine wipes the patch (`uv pip install` reinstalls it only when its version changes or with `--reinstall`); re-apply it afterwards. Applying it to an already-patched venv fails (`Hunk #1 FAILED` and the like), and so does applying it over an older or the upstream version of the diff; in that case restore pristine mmengine first: `uv pip install --reinstall --no-deps mmengine==0.10.7`.

### 5.5 YOLO detection models need no extra packages

A YOLO-backend detection model (`lada-yolo-*`, `zelefans-vr-yolo-v2`) is exported to ONNX by ultralytics before its TensorRT engine is built. `onnx` is a regular dependency and the export runs with `simplify=False`, so neither `onnxslim` nor `onnxruntime` is needed. RF-DETR models ship a pre-built `.onnx` that TensorRT parses directly.


---

## 6. Model weights and assets

Place these files in `$Workspace\jasna\model_weights\` (taken from an upstream release package):

- `lada_mosaic_restoration_model_generic_v1.2.pth`
- `rfdetr-v6.onnx` (the default detector since v0.9.0; `rfdetr-v6-large.onnx` for 4K and `rfdetr-vr-v1.onnx` for VR180 are optional, and the old `rfdetr-v5.onnx` still works)
- `lada_mosaic_detection_model_v4_fast.pt`
- `lada_vr_mosaic_detection_model_v2_accurate.pt` (the weights behind the VR180 YOLO detection model `zelefans-vr-yolo-v2`)

When running from source, Jasna resolves `model_weights\` automatically (Appendix A.1), so placing them there is all that's needed.

Detection models are auto-discovered from the files present in `model_weights\`, so everything else works without the fourth file; `zelefans-vr-yolo-v2` then simply does not appear in the model list and cannot be used for VR180 detection.

The two test clips normally ship with the repository in `$Workspace\jasna\assets\` (`test_clip1_1080p.mp4`, `test_clip1_2160p.mp4`). If absent, extract them from an upstream release.

### 6.1 Optional: RIFE frame-interpolation model (for `--frame-gen`)

Needed only for frame-rate doubling (`--frame-gen {2x,4x}`) with the default `rife` backend. On an RTX 40 series or newer GPU, `--frame-gen-backend rtx` (NVIDIA RTX Video Frame Generation from the `nvidia-vfx` wheel installed in Section 5) needs no weights at all; see [frame_generation.md](frame_generation.md). The RIFE weights are **not bundled** (non-commercial license terms), so create the TorchScript checkpoint yourself with `scripts/make_rife_torchscript.py`.

1. Clone Practical-RIFE and download a 4.x model package (**verified with v4.25**) so that `<repo>\train_log\` contains `RIFE_HDv3.py`, `IFNet_HDv3.py`, `flownet.pkl`:
   ```powershell
   git clone https://github.com/hzwer/Practical-RIFE
   # follow its README to download/extract the model package into Practical-RIFE\train_log\
   ```
2. Convert to TorchScript with the project venv (run in `$Workspace\jasna`):
   ```powershell
   .\.venv\Scripts\python.exe scripts\make_rife_torchscript.py `
       --rife-repo C:\path\to\Practical-RIFE `
       --output model_weights\rife.pth --validate
   ```

When **running from source**, `model_weights\rife.pth` is picked up automatically (same resolver as the other weights; or point `JASNA_MODEL_WEIGHTS_DIR` at a folder containing it). Full walkthrough: [frame_generation.md](frame_generation.md).

### 6.2 Optional: FlashVSR secondary restoration (experimental)

FlashVSR (`--secondary-restoration flashvsr` / `flashvsr-inline`) needs a separate repository checkout with its own venv. Use the fork [`sh202603/FlashVSR_plus`](https://github.com/sh202603/FlashVSR_plus) (it includes the tiny-long fix inline needs and the `--flashvsr-accel` speed-up, and `uv sync` installs its dependencies). Windows is supported; on a 16 GB card use `--flashvsr-tiles 2` for 1080p (inline follows `--max-clip-size`, default 90; the former 32-frame cap is gone). Setup instructions: [flashvsr.md](flashvsr.md).

---

### 6.3 Optional: SeedVR2 primary restoration (experimental)

SeedVR2 (`--restoration-model-name seedvr2`) needs a separate `ComfyUI-SeedVR2_VideoUpscaler` checkout with its own venv plus the ~90 MB LoRA in `model_weights/`; no patch is required and nothing is installed into jasna's venv. Setup instructions: [seedvr2.md](seedvr2.md).

---

### 6.4 Optional: SwiftVR secondary restoration and SwiftVR distill (experimental)

SwiftVR (`--secondary-restoration swiftvr` / `swiftvr-inline`, also the `swiftvr` backend of `--ltx-crop-vsr`) needs the fork [`sh202603/SwiftVR`](https://github.com/sh202603/SwiftVR) (branch `modi`) checked out with its own `uv sync` venv and the ~20 GB checkpoint. The small distilled model (`--secondary-restoration swiftvr-distill`, also the `swiftvr-distill` backend of `--ltx-crop-vsr`) needs nothing but `swiftvr-distill.pt` from [okatti/swiftvr-distill](https://huggingface.co/okatti/swiftvr-distill) in `model_weights\`. Setup instructions and the Windows notes: [swiftvr.md](swiftvr.md).

---

## 7. Run from source / smoke test

With Sections 1–6 done, run Jasna directly from the source checkout inside the venv:

```powershell
cd $Workspace\jasna
python -m jasna --version    # -> 0.11.2+modi
python -m jasna --help
python -m jasna --input assets\test_clip1_1080p.mp4 --output $env:TEMP\out.mp4   # process a short clip
python -m jasna              # launch the GUI (no arguments)
```

> **Launch the GUI with `python -m jasna` (no arguments).** The console script `jasna` is wired straight to `jasna.main:main` and never goes through the GUI dispatch (`jasna\__main__.py`), even with no arguments. Only the frozen build (`jasna.exe`) enters the GUI on a no-argument launch. Prefer `python -m jasna` for the CLI too: the console script `jasna` skips what `__main__.py` sets up first (`CUDA_MODULE_LOADING=LAZY`, `OMP_WAIT_POLICY=passive`, and the guard that stops multiprocessing children).

Check:

- `ffmpeg` / `ffprobe` (v8) are on `PATH` (Section 1.1).
- The NVIDIA driver is **610 or newer** (the startup check rejects older ones).
- The install path is ASCII-only (enforced at startup; watch out for extraction paths under user names containing non-ASCII characters).

> **The first processing run is slow.** On first use the GPU-specific TensorRT engines are compiled (15–60 min). They are cached next to the weights inside `model_weights\` (e.g. `<model>_sub_engines\`) and reused afterwards. Engine caches from the v0.9.1 base onwards stay valid (the engine code changed through v0.11.1, but TensorRT is still 10.16.1.11 and the engine file names are the same); `--max-clip-size` no longer triggers a recompile.

---

## 8. Packaging / frozen builds

This fork ships its own experimental Nuitka build script (`scripts\build_nuitka.py`) that produces a standalone distribution with `jasna.exe` and `jasna-framegen.exe` from the venv of this guide; the procedure, the bundle layout and its limits are in [frozen_build.md](frozen_build.md). The script follows the branch (PyAV media layer, the rfdetr-v6 detector, the CUDA kernel set, the worker scripts of the external restorers), but each new base needs a fresh build and smoke test on Windows.

Upstream's own packaging tooling (also Nuitka-based) lives in a **private submodule (`jasna/protection`)** not included in this fork. Features depending on that submodule (`unet-4x`, SD1.5 inpaint, license activation) do not work in this fork's frozen builds.

---

## Troubleshooting

### PyAV / FFmpeg

- **Every encode fails right at startup with `ValueError: hevc_nvenc did not accept encoder option(s): ['lookahead_level']`**
  PyAV is linked against an FFmpeg built with old nv-codec-headers. This cannot happen with the official PyPI wheel (18.1+ bundles a compatible FFmpeg): install it with `uv pip install --reinstall "av>=18.1"`. If you deliberately self-build, point `INCLUDE` / `LIB` at the BtbN build as in Appendix B.1.

- **`import av` fails with `ImportError: DLL load failed while importing _core`**
  The self-built wheel was installed without the delvewheel repair. Python 3.8+ does not search PATH for the DLL dependencies of extension modules, so keeping the BtbN `bin\` on PATH is not enough. Reinstall the delvewheel-repaired wheel (Appendix B.1), or switch to the official wheel with `uv pip install --reinstall "av>=18.1"`.

- **The PyAV build fails with `fatal error C1083: libavcodec/avcodec.h`**
  The compiler does not know where FFmpeg lives. `PKG_CONFIG_PATH` is ignored on Windows (setup.py never calls pkg-config there). Append the BtbN `include` / `lib` directories to `INCLUDE` / `LIB` as in Appendix B.1.

- **Encoder/decoder initialization fails with a current_ctx-related `TypeError` or similar**
  av is still 18.0.0: either PyPI's old release or the interim self-built wheel, which also reports `18.0.0`. Fix: `uv pip install "av>=18.1"` (Section 4).

- **Jasna refuses to start: wrong ffprobe version**
  The startup check requires `ffprobe` **major version 8**. Add a v8 build's `bin\` to `PATH` (Section 1.1).

- **The torchcodec backend cannot find the FFmpeg DLLs (`avcodec-62.dll` etc.)**
  torchcodec needs the FFmpeg shared DLLs at runtime. Keep the BtbN shared build's `bin\` on `PATH` (Section 1.1).

### jasna install

- **`uv pip install -e .[dev,nvidia]` cannot resolve `python_vali==4.8.8`**
  4.8.8 is not on PyPI. Build the VALI fork wheel (Section 5.2) and re-run with `--find-links $Workspace\vali\dist` (Section 5.3).

- **The VALI build fails with `ModuleNotFoundError: No module named 'pkg_resources'`**
  setuptools 81 or newer is installed. Go back with `uv pip install "setuptools<81"` (Section 5.1).

- **The VALI build fails with `error FTK1011`**
  The path is too long. Move the workspace to a short path, or map it to a short drive with `subst` (Section 5.2).

- **`uv pip install -e .[dev,nvidia]` fails with `No solution found ... no version of torch==2.12.1+cu130`**
  The PyTorch index is missing. Add `--extra-index-url https://download.pytorch.org/whl/cu130`.

- **`uv pip install -e .[dev,nvidia]` fails with `torch-tensorrt==2.12.0+cu130 ... unsatisfiable`**
  uv's default index strategy is first-index only and refuses to satisfy `torch-tensorrt==2.12.0` with the PyTorch index's `2.12.0+cu130` (local version). The transitive dependency `nvidia-cuda-runtime-cu13==0.0.0a0` is also a prerelease. Add `--index-strategy unsafe-best-match --prerelease=allow` (Section 5).

### Runtime

- **Jasna raises `FileNotFoundError` for a `.pth`/`.onnx`/`.pt`**
  Files missing from `model_weights\`, or the resolver is looking elsewhere. Put the Section 6 weights in `$Workspace\jasna\model_weights\`, or set `JASNA_MODEL_WEIGHTS_DIR` to a folder containing them.

- **Startup rejects the NVIDIA driver version**
  The Windows minimum is **610**. Update the driver from NVIDIA.

- **Loading BasicVSR++ fails with `Weights only load failed ... HistoryBuffer`**
  The mmengine patch is missing, or the upstream version (`weights_only=True`) is applied. Restore pristine mmengine and apply the bundled diff as in Section 5.4.

- **The first launch is extremely slow**
  Not a bug. The initial TensorRT engine compilation takes 15–60 minutes. The engines are cached next to the weights inside `model_weights\` and later runs are fast.

---

## Appendix A: branch changes this setup relies on

Everything below is applied in the branch's source; no standalone patch files are shipped (the two bundled patches, for mmengine and FlashVSR_plus, are applied to other packages: Section 5.4 and [flashvsr.md](flashvsr.md)). The full list of what the branch changes is [changes_vs_upstream.md](changes_vs_upstream.md).

### A.1 Automatic resolution of the `model_weights\` directory

`jasna/model_weights_resolver.py` looks for `model_weights\` in this order: the `JASNA_MODEL_WEIGHTS_DIR` environment variable, next to the executable (frozen builds only), the current directory, then next to the `jasna` package (the source tree). The first hit wins and is logged at startup with `--log-level info`. The CLI, the detection registry, the engine paths and the GUI go through it instead of a hardcoded `Path("model_weights")`, so the CLI runs from any directory and Section 6 only has to place the files.

### A.2 DLL search paths for the torchcodec backend

Windows does not search `PATH` for the DLL dependencies of extension modules, so the torchcodec backend modules (`jasna/media/torchcodec_decoder.py`, `torchcodec_encoder.py`) register the torchcodec package directory and `CUDA_PATH\bin` (when set) with `os.add_dll_directory()`, and the branch extends upstream's `jasna/packaging/windows_dll_paths.py` with the `torchcodec`, `tensorrt_rtx_libs` and bundled `tools\` (the frozen build's FFmpeg 8 DLLs) directories. `windows_dll_paths.py` acts only in a frozen build and does nothing when running from source (it returns at once unless `is_frozen()`). The native PyAV path needs none of this. The FFmpeg DLLs torchcodec links by their plain names (`avcodec-62.dll`, ...) still have to come from a shared build on `PATH` (Section 1.1) or from the frozen build's `tools\`.

---

## Appendix B: history of this setup

What changed in the from-source setup at each base, newest first, and when this guide was verified on Windows. Procedures that are no longer needed are summarized; their full text is in this file's git history.

- **v0.11.2 base (not yet verified on Windows)**: the only pin change is `comfy-kitchen` 0.2.36 to 0.2.38 (upstream ships prebuilt abi3 wheels for Windows too, so `uv pip install comfy-kitchen==0.2.38` in the existing venv should be enough); upstream adds the LTX less-overlap mode and a hard-disk install warning whose Windows side queries the volume's seek penalty through `IOCTL_STORAGE_QUERY_PROPERTY`. To verify: the venv update, the full test suite against pristine upstream `b623466f`, the CLI runs of the modi features, and that a run from an SSD prints no hard-disk warning. The Linux verification of this base is in [building_linux.md](building_linux.md).
- **v0.11.1 base (verified 2026-10-16: Windows 11, RTX 5060 Ti, driver 617.42, Python 3.13.9, CUDA 13.2)**: upstream pins torch 2.12.1 / torchvision 0.27.1, adds `comfy-kitchen==0.2.36` and `ultralytics>=8.4.96` (whose YOLO export needs neither onnxruntime nor onnxslim, so Section 5.5 shrank to a note), pins `av==18.1.0` exactly, and selects the decode backend only through `JASNA_DECODE_BACKEND` (the in-code `DECODE_BACKEND` toggle is gone). On Linux this base is verified ([building_linux.md](building_linux.md)), including the VALI fork wheel rebuilt at Kruk2's `0e5c01e` (4.8.8 with the ASF/VC-1 timestamp fixes and the decoder-open GIL release). On Windows a v0.10.0-era venv was updated with the Section 5 command (`[dev,nvidia,torchcodec]` with `--find-links`) and the `0e5c01e` wheel reinstalled with `--force-reinstall --no-deps`; what changed was torch / torchvision, the new `comfy-kitchen` and `nvidia-ml-py`, `nvidia-vfx` 0.2.0.0 and ultralytics 8.4.178, while TensorRT stayed and the existing `.win` engines were reused. The full-suite failure set matched pristine upstream (the two modi-only failures were test defects, since fixed); fifteen CLI runs of the modi features (secondary restorers, SeedVR2, frame generation, `--segments`, `--ltx-trial`, the LTX/frame-gen refusal), four `--ltx-crop-vsr` runs and the GUI (start-up, migration of v0.10.0-era presets, queue processing, the visual check of the player and the rest) passed. `--ltx-trial` builds its native block only when `cl` is on PATH and otherwise runs through torch.compile (a session with vcvars64 applied, such as a Developer PowerShell, attempts the native build). The same day VALI became a required step on Windows and Section 5 was reordered: build packages (including setuptools<81), the VALI wheel build, the jasna install with `--find-links`, then the mmengine patch. For the mmengine patch, the version upstream changed to `weights_only=True` in `7c64105` was found to break loading `v1.2.pth` in a source run, so the fork keeps shipping it with `False`; the section now describes all four files, and the tool changed from the Python `patch` package (which silently skips the `model/__init__.py` hunk) to the GNU patch of Git for Windows.
- **nvidia-vfx 0.2.0.0 pin (2026-10-03, verified on Windows: RTX 5080, driver 616.92)**: the branch pins `nvidia-vfx==0.2.0.0` (upstream leaves it unpinned) for the `VideoFrameGeneration` effect behind `--frame-gen-backend rtx`. 0.2.0.0 bundles no TensorRT (which ended the `libnvinfer` clash that mattered on Linux) and ships its own `cudart64_12.dll`. Venvs created earlier hold 0.1.0.1: `uv pip install nvidia-vfx==0.2.0.0`.
- **PyAV from PyPI (2026-08-24; verified on Windows 2026-10-16)**: av 18.1.0 reached PyPI with the current_ctx API and the explicit CUDA stream support the encoder uses, ending the interim self-built wheel of B.1. That interim wheel reports `18.0.0`, so the Section 5 install replaces it (or `uv pip install "av>=18.1"` at once); if `python -c "import av; print(av.__version__)"` prints `18.0.0`, the interim wheel is still installed. The PyPI Windows wheel (tag `cp311-abi3-win_amd64`) is delvewheel-repaired too and carries `av.libs`, so the presence of `av.libs` tells nothing. The official wheel is verified on Linux and Windows; the last Windows-verified setup (v0.10.0, 2026-08-13) used the interim wheel built from commit `f6f0a5e`, which is contained in 18.1.0.
- **v0.10.0 base (verified 2026-08-13: Windows 11, RTX 5060 Ti, driver 610.88, Python 3.13.9, CUDA 13.2, ffmpeg 8.1)**: the full-pytest failure set matched the bare v0.10.0 baseline; fifteen CLI runs (including `--cq`, chapter/subtitle preservation, `JASNA_DECODE_BACKEND`, post-export commands, near-lock retarget and streaming restart) and the GUI smoke (restored-video player on the VLC and SoftwareClock paths, queue right-click actions, A/B comparison, Segment Editor) passed; the VALI fork wheel 4.8.8 (`f4a67f8`) was built and verified (now Section 5.2). New at this base: the `JASNA_DECODE_BACKEND` environment variable, and the `python_vali==4.8.8` pin, which on Windows made the self-built fork wheel a prerequisite of the `[nvidia]` resolve (Section 5).
- **v0.9.1 base (verified 2026-07-30, RTX 5080)**: the VALI NVDEC decode backend returned as an optional, self-built fork wheel (fork commit `3ad0d54` = 4.8.7; both decode paths through VALI, output bitstream-md5-identical to the PyAV run, coexistence with the av wheel and torchcodec confirmed). The encoder started passing a CUDA stream via `CudaContext(cuda_stream=...)`, which until av 18.1.0 required the interim PyAV wheel. All TensorRT engines recompiled once.
- **v0.8.1 base (verified 2026-07-19)**: the GPU stack moved into the `nvidia` / `amd` extras (hence `.[dev,nvidia]`), `requires-python` relaxed to `>=3.12`.
- **v0.8.0 base (verified 2026-07-18, Windows 11, RTX 5080)**: upstream moved the media layer to PyAV (NVDEC/NVENC) and removed `python_vali` / `PyNvVideoCodec` from the runtime. With their C++ builds went every prerequisite the earlier guide needed: CUDA Toolkit 13.2 and the pinned `CUDA_PATH`, VS Build Tools 2022, the ffmpeg dev build under `C:\Program Files\ffmpeg8\` with its junction workaround (the `FFMPEG_ROOT` hardcode belonged to vali), the build environment variables (`DISTUTILS_USE_SDK`, `CMAKE_GENERATOR`, `CL=/utf-8`, `FFMPEG_DIR`) and `setup-build-env.ps1`, the `setuptools<80` pin, and MKVToolNix. Of these, VS Build Tools, the CUDA Toolkit and `C:\Program Files\ffmpeg8` are needed again since the v0.10.0 base made the VALI fork build a prerequisite of the `[nvidia]` resolve (Section 5.2). Leftovers of these do no harm (`CUDA_PATH` is even consulted by Appendix A.2). The NVIDIA driver minimum rose to 610. The FP8 backend was first verified on Windows here.

### B.1 Interim PyAV wheel build (2026-07 to 2026-08)

Required while av 18.1.0 was unpublished; kept in case a source build is ever needed again. The link target must be an FFmpeg 8 built with nv-codec-headers 12.2+ (the BtbN shared build qualifies; with an older one every encode fails at startup with `ValueError: hevc_nvenc did not accept encoder option(s): ['lookahead_level']`). Run inside a Developer PowerShell for VS 2022. Three Windows-specific points: PyAV's setup.py never calls pkg-config on Windows, so FFmpeg's headers and libraries reach the compiler only through the MSVC `INCLUDE` / `LIB` variables (without them: `fatal error C1083: libavcodec/avcodec.h`); Python 3.8+ does not search `PATH` for an extension module's DLL dependencies, so the FFmpeg DLLs must be vendored into the wheel with delvewheel (otherwise `ImportError: DLL load failed while importing _core`); and the resulting wheel reports `18.0.0`, which the `av==18.1.0` pin replaces on the next Section 5 install unless you reinstall it afterwards.

```powershell
cd $Workspace
git clone https://github.com/PyAV-Org/PyAV.git
cd PyAV
git checkout f6f0a5e                 # verified on Linux and Windows; contained in v18.1.0
$env:INCLUDE = "$env:INCLUDE;$Workspace\ffmpeg-n8.1-shared\include"
$env:LIB     = "$env:LIB;$Workspace\ffmpeg-n8.1-shared\lib"
uv build --wheel                     # -> dist\av-18.0.0-cp311-abi3-win_amd64.whl
uv pip install delvewheel
delvewheel repair dist\av-18.0.0-cp311-abi3-win_amd64.whl `
    --add-path $Workspace\ffmpeg-n8.1-shared\bin -w dist\repaired
uv pip install --reinstall (Get-Item dist\repaired\av-*.whl)
python -c "import av; from av.video.frame import CudaContext; print(av.ffmpeg_version_info); print(hasattr(CudaContext, 'current_ctx'))"
```

delvewheel vendors the DLLs under mangled names in `av.libs\`, a separate in-process copy from the FFmpeg DLLs torchcodec loads; that is fine on Windows, where each DLL has its own symbol namespace.
