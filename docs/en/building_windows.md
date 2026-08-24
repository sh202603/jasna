# Building Jasna for Windows

How to set up Jasna on Windows and run it **from source**.

> **Verification status (2026-08-13, `0.10.0+modi` on upstream `93d0584` = the v0.10.0 tag)**: These instructions are **verified on real hardware on the 93d0584 base** — Windows 11 + RTX 5060 Ti (driver 610.88, Python 3.13.9, CUDA 13.2, ffmpeg 8.1); the full-pytest failure set matches the bare v0.10.0 baseline exactly, fifteen CLI runs (including the v0.10.0 additions: `--cq`, chapter/subtitle preservation, `JASNA_DECODE_BACKEND`, post-export commands, near-lock retarget, and streaming restart) plus the GUI smoke (restored-video player playback/seek/full-screen on both the VLC and SoftwareClock paths, queue right-click actions, A/B comparison, segment editor) all pass, and the Windows build of the VALI fork wheel 4.8.8 is verified in Section 5.3. (The previous verification, on the a7cdaf8 base, was 2026-07-30.) For the Linux side see [building_linux.md](building_linux.md).

> **This guide covers the `v0.9.1+modi` branch.** The GPU stack (**torch 2.12.0+cu130 / torchvision 0.27.0+cu130 / torch-tensorrt 2.12.0+cu130 / tensorrt 10.16.1.11**) is unchanged from the v0.7.2 era, and the pins are already applied in `pyproject.toml` on this branch. TensorRT stays on the **10.16** line because `torch-tensorrt==2.12.0` requires `tensorrt>=10.16.1,<10.17.0` (torch-tensorrt does not support TensorRT 11 yet).

> **v0.8.0 simplified the setup considerably.** Upstream v0.8.0 moved the media layer to PyAV (NVDEC/NVENC), which removed the C++ builds of `python_vali` / `PyNvVideoCodec` entirely. All of the following prerequisites of the old guide are gone with them:
>
> - CUDA Toolkit 13.2 (and pinning `CUDA_PATH`)
> - VS Build Tools 2022 (except when building the PyAV wheel yourself)
> - Extracting an ffmpeg dev build to `C:\Program Files\ffmpeg8\` and the junction workaround (the `FFMPEG_ROOT` hardcode belonged to vali)
> - The build environment variables (`DISTUTILS_USE_SDK`, `CMAKE_GENERATOR`, `CL=/utf-8`, `FFMPEG_DIR`) and `setup-build-env.ps1`
> - The `setuptools<80` pin
> - MKVToolNix (`mkvmerge`)
>
> Leftovers of these in an existing environment do no harm (`CUDA_PATH` can even help: the torchcodec backend's DLL-resolution fallback consults it). On the other hand, **the NVIDIA driver requirement rose to 610+** (the startup check rejects anything below 610 on Windows). The one special step that used to remain, self-building the **PyAV wheel**, ended with av 18.1.0 reaching PyPI (Section 4).
>
> **Main additions on this branch:** 2x/4x frame generation via RIFE ([frame_generation.md](frame_generation.md)), the torchcodec backend ([torchcodec_backend.md](torchcodec_backend.md)), the cuDNN FP8 restoration backend ([fp8_recon.md](fp8_recon.md)), SeedVR2 primary restoration ([seedvr2.md](seedvr2.md)), and FlashVSR secondary restoration ([flashvsr.md](flashvsr.md)). The AV1 / 8-bit / BT.601 and BT.2020 output features of v0.7.2+modi were absorbed into upstream v0.8.0. See [changes_vs_upstream.md](changes_vs_upstream.md) for the full delta.

> **Packaging note:** This fork ships an experimental Nuitka build script (`scripts\build_nuitka.py`), but it is ⚠️ **not yet updated for the v0.8.0 media-layer migration** (it still assumes bundling the old `python_vali` / `PyNvVideoCodec` DLLs and is not expected to work as-is). Details: [Packaging / frozen builds](#8-packaging--frozen-builds).

---

## 1. Prerequisites

| Category | Requirement | winget ID / Source | Notes |
|---|---|---|---|
| OS | Windows 10/11 x64 | n/a | PowerShell 7+ recommended |
| Git | Git for Windows | `Git.Git` | For cloning the repository |
| uv | Latest | `astral-sh.uv` | Manages Python automatically |
| Python | 3.13 | n/a (managed by uv) | Auto-fetched by `uv venv --python 3.13` if missing. v0.8.1 relaxed `requires-python` to `>=3.12`, but this guide is verified on 3.13 only (3.14 is unverified against the cu130 GPU stack) |
| NVIDIA Driver | **610+** | NVIDIA official or the NVIDIA App | The startup check requires 610+ on Windows. GPU must be compute capability 7.5+ |
| ffmpeg / ffprobe | **v8** (runtime CLI) | [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases), `ffmpeg-n8.1-latest-win64-gpl-shared-*.zip` recommended (see below) | The startup check requires `ffprobe` major version 8 |
| VS Build Tools | 2022 (Desktop development with C++) | `Microsoft.VisualStudio.2022.BuildTools` | **Only when building the PyAV wheel yourself** (Section 4 (B)) |

The CUDA Toolkit is no longer needed for the core setup. The torch / tensorrt pip wheels bundle the CUDA runtime, and nothing is compiled in the core setup (since av 18.1.0 the PyAV wheel comes from PyPI, Section 4). The only exceptions are the optional PyAV self-build (Section 4 (B)) and the optional VALI backend build (Section 5.3).

### 1.1 Choosing ffmpeg and adding it to PATH

For the CLI alone (`ffmpeg.exe` / `ffprobe.exe`), gyan.dev's [release-full build](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z) suffices (it is also what the startup check's error message points to). The BtbN **shared build** is recommended because the same extraction also covers:

- the FFmpeg DLLs (`avcodec-62.dll` etc. in `bin\`) that the optional torchcodec backend needs at runtime
- the `include\` (headers) and `lib\` (MSVC import libraries such as `avcodec.lib`) needed when building the PyAV wheel yourself (Section 4 (B))

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

# only when building the PyAV wheel yourself (Section 4 (B)):
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --silent `
  --override "--wait --quiet --add ProductLang En-us --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Install the NVIDIA driver (610+) from NVIDIA directly or via the NVIDIA App.

---

## 2. Clone the repository

Only `jasna` needs to be cloned. `vali` / `PyNvVideoCodec` are no longer used by the v0.8.0 runtime, so checking them out is unnecessary.

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

uv's managed Python is fine here (a system Python was only required by the old guide's vali / PyNvVideoCodec builds, removed in v0.8.0). The old guide's tool pre-install (`cmake ninja scikit-build "setuptools<80" wheel numpy`) is obsolete too.

---

## 4. Install the PyAV wheel

The GPU path uses PyAV's **current_ctx API** (it lets NVDEC/NVENC share the CUDA context torch already initialized) plus, since the v0.9.1 base, the **explicit CUDA-stream support** the encoder passes via `CudaContext(cuda_stream=...)`. **av 18.1.0 (PyPI, 2026-08) is the first release that ships both**, so the official wheel is now the standard path.

### (A) Use the official PyPI wheel (standard path)

The `av>=18.1,<19` requirement in `pyproject.toml` resolves during the Section 5 install; `uv pip install "av>=18.1"` also works on its own. The PyPI binary wheels bundle their own FFmpeg 8 built with new-enough nv-codec-headers (8.1.2 at the time of writing; `hevc_nvenc` accepts `lookahead_level`), so no BtbN build and no delvewheel repair are involved.

Verified with the official 18.1.0 wheel on Linux hardware (2026-08-24: media-layer and full-pipeline e2e tests pass, NVDEC decode and NVENC encode included). Not yet re-verified on Windows hardware; the last Windows-verified setup used the self-built `f6f0a5e` wheel from (B), and that commit is contained in the v18.1.0 release, so the official wheel carries the same code.

For the Windows build of the optional VALI decode backend (fork `python_vali` wheel) see Section 5.3; without it the reader falls back to PyAV, which is fully functional.

> **Migrating an existing venv:** the interim self-built wheel reports version `18.0.0`, which no longer satisfies `av>=18.1`, so the next Section 5 install replaces it with the official wheel automatically. To switch immediately, run `uv pip install "av>=18.1"` once. `Test-Path .venv\Lib\site-packages\av.libs` tells whether the delvewheel-repaired interim wheel is still installed (the path is absent for the official wheel).

### (B) Build from PyAV main yourself (interim procedure, normally no longer needed; verified on hardware)

This was the required path while 18.1.0 was unpublished; it remains here in case a source build is ever needed again. The link target must be an **FFmpeg 8 built with nv-codec-headers 12.2+**: with an older one, `hevc_nvenc` lacks the `lookahead_level` option and every jasna encode fails right at startup with `ValueError: hevc_nvenc did not accept encoder option(s): ['lookahead_level']` (a symptom actually hit with the distro FFmpeg on Linux; the BtbN builds satisfy the requirement).

Check out upstream main `f6f0a5e` (contained in v18.1.0), which has both current_ctx and the CUDA-stream API merged, and link against the BtbN shared build. The only additional prerequisite is VS Build Tools 2022 (run inside a **Developer PowerShell for VS 2022** session).

> **No pkg-config / pkgconf is needed.** PyAV's setup.py **never calls pkg-config on Windows** (that code path is non-Windows only). Setting `PKG_CONFIG_PATH` is silently ignored; the only way FFmpeg's headers and libraries reach the compiler is through the standard MSVC environment variables `INCLUDE` / `LIB`. Building without them fails with `fatal error C1083: libavcodec/avcodec.h: No such file or directory`.

```powershell
cd $Workspace
git clone https://github.com/PyAV-Org/PyAV.git
cd PyAV
# f6f0a5e is verified on Linux and Windows; it is contained in the v18.1.0 release
git checkout f6f0a5e

# cl.exe reads INCLUDE, link.exe reads LIB (append to the Developer PowerShell values)
$env:INCLUDE = "$env:INCLUDE;$Workspace\ffmpeg-n8.1-shared\include"
$env:LIB     = "$env:LIB;$Workspace\ffmpeg-n8.1-shared\lib"
uv build --wheel                     # -> dist\av-18.0.0-cp311-abi3-win_amd64.whl
```

**delvewheel is mandatory for FFmpeg DLL resolution.** Since Python 3.8, Windows does not search PATH for the DLL dependencies of extension modules (`av\_core.pyd`) — only the system directories, the .pyd's own folder, and directories registered via `os.add_dll_directory()`. Keeping the BtbN `bin\` on PATH is therefore not enough: the import fails with `ImportError: DLL load failed while importing _core` (measured). Vendor the FFmpeg DLLs into the wheel with delvewheel before installing:

```powershell
uv pip install delvewheel
delvewheel repair dist\av-18.0.0-cp311-abi3-win_amd64.whl `
    --add-path $Workspace\ffmpeg-n8.1-shared\bin -w dist\repaired
uv pip install --reinstall (Get-Item dist\repaired\av-*.whl)

# Verify: BtbN FFmpeg (n8.1.x) is picked up and the current_ctx API is present
python -c "import av; from av.video.frame import CudaContext; print(av.ffmpeg_version_info); print(hasattr(CudaContext, 'current_ctx'))"
```

delvewheel vendors the DLLs under hash-mangled names in `av.libs\`, so at runtime they are a separate in-process copy from the FFmpeg DLLs torchcodec loads. That is fine on Windows — each DLL has its own symbol namespace, so the same-name library collision that bit Linux cannot happen (confirmed by the full smoke-test matrix).

> **Note: the self-built wheel reports version `18.0.0`, which no longer satisfies the `av>=18.1` requirement in `pyproject.toml`.** The Section 5 install therefore replaces it with the official PyPI wheel; if you really mean to run the self-built one, reinstall it afterwards with `uv pip install --reinstall (Get-Item dist\repaired\av-*.whl)`. To check which one is installed: `Test-Path .venv\Lib\site-packages\av.libs` (present only for the delvewheel-repaired wheel).

---

## 5. Install jasna itself

Since v0.8.1 the GPU stack is split into extras (`nvidia` = the NVIDIA stack, `amd` = the ROCm one). For an NVIDIA build the `nvidia` extra pulls in `torch==2.12.0+cu130` / `torchvision==0.27.0+cu130` / `torch-tensorrt==2.12.0` / `nvidia-vfx`, which are not on the default PyPI index. Point uv at the PyTorch wheel index with `--extra-index-url` and add two more flags:

```powershell
cd $Workspace\jasna
uv pip install -e .[dev,nvidia] `
    --extra-index-url https://download.pytorch.org/whl/cu130 `
    --index-strategy unsafe-best-match `
    --prerelease=allow
```

> **⚠️ On Windows this command fails to resolve as-is.** The `[nvidia]` extra pins `python_vali==4.8.8`, and PyPI has no Windows wheel for it (you get a `No wheels with a matching platform tag` error). Workaround: create the venv with `[dev,torchcodec]` first (no `nvidia`), build and install the fork wheel (4.8.8) per Section 5.3, then re-run the `[dev,nvidia,...]` command above. The installed 4.8.8 satisfies the pin, so the second run resolves (alternatively, keep the wheel in `dist\` and add `--find-links <workspace>\vali\dist` to resolve in one pass; confirmed on hardware).

Why each flag:

- `--extra-index-url https://download.pytorch.org/whl/cu130`: source of the `torch+cu130` / `torchvision+cu130` / `torch-tensorrt+cu130` wheels
- `--index-strategy unsafe-best-match`: needed so `torch-tensorrt==2.12.0` from `pyproject.toml` can be satisfied by `torch-tensorrt==2.12.0+cu130` (local version) on the PyTorch index; the default first-index strategy refuses
- `--prerelease=allow`: the transitive dependency `nvidia-cuda-runtime-cu13==0.0.0a0` is a prerelease

The `[dev]` extra installs `nuitka>=2.4`, `pytest`, `pytest-cov`, `scikit-build`, `cmake`, `ninja`. The `[nvidia]` extra installs the GPU stack (torch cu130 / TensorRT / torch-tensorrt / nvidia-vfx; split out of the required dependencies in v0.8.1) — **omitting it leaves you without torch and the app will not start**.

**Optional: the torchcodec backend.** To use the experimental torchcodec decode/encode path (`--video-backend torchcodec`/`auto`), add the `torchcodec` extra and install `.[dev,nvidia,torchcodec]` with the same flags:

```powershell
uv pip install -e .[dev,nvidia,torchcodec] `
    --extra-index-url https://download.pytorch.org/whl/cu130 `
    --index-strategy unsafe-best-match `
    --prerelease=allow
```

This installs `torchcodec>=0.15.0`. torchcodec needs the FFmpeg DLLs at runtime, so keep the BtbN shared build's `bin\` on PATH (Section 1.1). It is not needed for a normal setup; the default `native` backend (PyAV) works without torchcodec. Details: [torchcodec_backend.md](torchcodec_backend.md).

**Note: the FP8 restoration backend needs no extra install steps.** Its dependencies `nvidia-cudnn-frontend` and (on Windows) `triton-windows` are regular dependencies in `pyproject.toml` and come in with the command above. The cuDNN runtime (9.17+) ships inside the torch cu130 wheels. The feature is a runtime opt-in (`--fp8-recon`, needs an FP8-capable GPU, sm89+) and falls back to the TensorRT engines where unavailable. Verified on hardware (Windows 11 + RTX 5080) on v0.8.0+modi as well (2026-07-18; confirmed `CudnnFP8Upsample: enabled` with `--log-level info` — note the default `--log-level error` suppresses the activation log line). Details: [fp8_recon.md](fp8_recon.md).

**Optional: the TensorRT-RTX flavor.** Installing the `nvidia-rtx` extra *instead of* `nvidia` switches the TensorRT stack to TensorRT-RTX (JIT compilation; engine builds finish in seconds instead of minutes). The two extras are mutually exclusive in one venv, so use a dedicated venv. Engines are cached under `.rtx`-tagged names (`.rtx.win`), so both flavors can share one `model_weights` directory. The mmengine patch (§5.1) applies to this venv too. Note: `python_vali` has no Windows wheel on PyPI; supply a locally built wheel via `--find-links` (see §5). Details and measured numbers (validated on Windows 11 / RTX 5060 Ti): [tensorrt_rtx.md](tensorrt_rtx.md).


### 5.1 Apply the mmengine patch (torch 2.6+ compatibility)

Add `weights_only=False` to the `torch.load` calls inside `mmengine.runner.checkpoint`. From torch 2.6 the default flipped to `weights_only=True`, which breaks loading the existing `.pth` checkpoints without this patch.

Apply `patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff` to the venv's `site-packages`. So that `patch.exe` isn't required, install the Python `patch` package temporarily:

```powershell
cd $Workspace\jasna

uv pip install patch
uv run --no-project python -m patch -p1 -d .venv\Lib\site-packages `
    patches\fix_loading_mmengine_weights_on_torch26_and_higher.diff
uv pip uninstall patch
```

Verify:

```powershell
Select-String -Path .venv\Lib\site-packages\mmengine\runner\checkpoint.py -Pattern "weights_only=False"
# → one matching line means it's applied
```

> This patch is wiped every time `uv pip install -e .[dev,nvidia]` reinstalls `mmengine`. Re-apply it after reinstalls.

### 5.2 ONNX packages (for YOLO detection models)

When using a **YOLO-backend detection model (lada-yolo-\* and zelefans-vr-yolo-v2)**, ultralytics exports the model to ONNX before building the TensorRT engine, which needs three packages that are not installed automatically (RF-DETR models do not need them: their pre-built `.onnx` is parsed by TensorRT directly):

```powershell
uv pip install onnx onnxslim onnxruntime
```

> **Without them**, running with `--detection-model lada-yolo-v4` aborts engine compilation with `ERROR ONNX: export failure ... No module named 'onnx'` → `RuntimeError: Engine compilation subprocess failed`. Installing the three packages into the venv resolves it.

### 5.3 Optional: build the VALI NVDEC decode backend (`python_vali` fork)

Native decode tries the VALI NVDEC decoder first and escalates to PyAV when it is unavailable (since v0.10.0 the `JASNA_DECODE_BACKEND` environment variable — `auto` / `vali` / `pyav-hw` / `pyav-sw` — selects this chain explicitly; modi's `--video-backend` sits one layer above it, and the chain applies when `native` is selected). Without a wheel that has the fork-only API `DecodeSingleSurfaceAsyncDetailed`, the reader logs a warning and falls back to PyAV every time (everything still works without this section). On Windows, resolving the `[nvidia]` extra also needs a 4.8.8 wheel (see the note in Section 5), which makes building the fork wheel a de-facto prerequisite.

Additional prerequisites on top of Section 1:

- **VS Build Tools 2022** (same as Section 4 (B); build inside a cmd / Developer PowerShell session that went through vcvars64)
- **CUDA Toolkit 13.x** (`nvcc`; vali's CMake resolves the toolkit from `CUDA_PATH`)
- **An FFmpeg 8 shared build**: vali's CMake **hardcodes `FFMPEG_ROOT` to `C:\Program Files\ffmpeg8`**, so place a shared build with `include\`/`lib\`/`bin\` there (e.g. gyan.dev full-shared), or create a junction (`New-Item -ItemType Junction -Path "C:\Program Files\ffmpeg8" -Target <extracted dir>`). The BtbN build (Section 1.1) satisfies the same requirement.
- cmake, ninja, scikit-build, a setuptools with `pkg_resources` (<80): all already in the venv via `[dev]` (Section 5).

```powershell
cd $Workspace
git clone https://codeberg.org/Kruk2/vali.git       # skip if you have a checkout
cd vali
git checkout f4a67f8      # = the 4.8.8 pin, has DecodeSingleSurfaceAsyncDetailed and the post-EAGAIN drain fix
git submodule update --init --recursive
```

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

Unlike on Linux, the Windows wheel bundles the FFmpeg DLLs inside `python_vali\` (CMake's `PYPI_BUILD=0` default), so no delvewheel step is needed. Install and verify:

```powershell
uv pip install --force-reinstall --no-deps (Get-Item $Workspace\vali\dist\python_vali-4.8.8-*.whl)
python -c "import python_vali as v; print(hasattr(v.PyDecoder, 'DecodeSingleSurfaceAsyncDetailed'))"   # -> True
# running jasna with --log-level info now prints "Using VALI NVDEC decoder for <file>"
```

> The Windows build and a full pipeline run are verified at fork commit `f4a67f8` (= the 4.8.8 pin, 2026-08-13, RTX 5060 Ti + CUDA 13.2). The previous verification was at fork commit `3ad0d54` (= the 4.8.7 pin, 2026-07-30, RTX 5080 + CUDA 13.2 + gyan.dev 8.1 shared): both decode paths of the pipeline go through VALI, the bundled test clip's output is **bitstream-md5-identical** to the PyAV-decoded run, and coexistence with the av wheel and the torchcodec backend is confirmed (av vendors FFmpeg via delvewheel, vali bundles its own copy, torchcodec resolves via PATH — three separate in-process DLL copies, which is fine on Windows because each DLL has its own symbol namespace).

---

## 6. Model weights and assets

Place these 4 files in `$Workspace\jasna\model_weights\`:

- `lada_mosaic_restoration_model_generic_v1.2.pth`
- `rfdetr-v5.onnx`
- `lada_mosaic_detection_model_v4_fast.pt`
- `lada_vr_mosaic_detection_model_v2_accurate.pt` (added in v0.8.0; the weights behind the VR180 YOLO detection model `zelefans-vr-yolo-v2`)

When running from source, Jasna resolves `model_weights\` automatically (Appendix A.1), so placing them there is all that's needed.

Detection models are auto-discovered from the files present in `model_weights\`, so everything else works without the fourth file; `zelefans-vr-yolo-v2` then simply does not appear in the model list and cannot be used for VR180 detection.

The two test clips normally ship with the repository in `$Workspace\jasna\assets\` (`test_clip1_1080p.mp4`, `test_clip1_2160p.mp4`). If absent, extract them from an upstream release.

### 6.1 Optional: RIFE frame-interpolation model (for `--frame-gen`)

Needed only for frame-rate doubling (`--frame-gen {2x,4x}`). The RIFE weights are **not bundled** (non-commercial license terms), so create the TorchScript checkpoint yourself with `scripts/make_rife_torchscript.py`.

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

FlashVSR (`--secondary-restoration flashvsr` / `flashvsr-inline`) needs a separate repository checkout with its own venv, and the inline mode additionally needs a bundled patch applied to that checkout. Windows is supported (inline is re-verified on v0.8.0+modi on a 16 GB Windows card with `--flashvsr-tiles 2`; v0.8.0 automatically caps `--max-clip-size` to 32 in inline mode, leaving more VRAM headroom than in the v0.7.2 era). Setup instructions: [flashvsr.md](flashvsr.md).

---

### 6.3 Optional: SeedVR2 primary restoration (experimental)

SeedVR2 (`--restoration-model-name seedvr2`) needs a separate `ComfyUI-SeedVR2_VideoUpscaler` checkout with its own venv plus the ~90 MB LoRA in `model_weights/`; no patch is required and nothing is installed into jasna's venv. Setup instructions: [seedvr2.md](seedvr2.md).

---

## 7. Run from source / smoke test

With Sections 1–6 done, run Jasna directly from the source checkout inside the venv:

```powershell
cd $Workspace\jasna
python -m jasna --version    # -> 0.9.1+modi
python -m jasna --help
jasna --input assets\test_clip1_1080p.mp4 --output $env:TEMP\out.mp4   # process a short clip
python -m jasna              # launch the GUI (no arguments)
```

> **Launch the GUI with `python -m jasna` (no arguments).** The console script `jasna` is wired straight to `jasna.main:main` and never goes through the GUI dispatch (`jasna\__main__.py`), even with no arguments. Only the frozen build (`jasna.exe`) enters the GUI on a no-argument launch.

Check:

- `ffmpeg` / `ffprobe` (v8) are on `PATH` (Section 1.1). `mkvmerge` is no longer required.
- The NVIDIA driver is **610 or newer** (the startup check rejects older ones).
- The install path is ASCII-only (enforced at startup; watch out for extraction paths under user names containing non-ASCII characters).

> **The first processing run is slow.** On first use the GPU-specific TensorRT engines are compiled (15–60 min). They are cached next to the weights inside `model_weights\` (e.g. `<model>_sub_engines\`) and reused afterwards. When migrating from v0.7.2, some engines recompile because the default max-clip-size changed (90); the loop_body engine caches keep their names and are reused.

---

## 8. Packaging / frozen builds

This fork ships an experimental Nuitka build script (`scripts\build_nuitka.py`) that in the v0.7.2 era produced a standalone distribution with a single `jasna.exe` ([frozen_build.md](frozen_build.md)).

⚠️ **The script has not been updated for the v0.8.0 media-layer migration.** It still assumes bundling the old `python_vali` / `PyNvVideoCodec` DLLs (CUDA NPP / nvJPEG etc.) and is not expected to work as-is. It will be updated after the Windows hardware verification.

Upstream's own packaging tooling (also Nuitka-based) lives in a **private submodule (`jasna/protection`)** not included in this fork. Features depending on that submodule (`unet-4x`, SD1.5 inpaint, license activation) do not work in this fork's frozen builds.

---

## Troubleshooting

### PyAV / FFmpeg

- **Every encode fails right at startup with `ValueError: hevc_nvenc did not accept encoder option(s): ['lookahead_level']`**
  PyAV is linked against an FFmpeg built with old nv-codec-headers. This cannot happen with the official PyPI wheel (18.1+ bundles a compatible FFmpeg): install it with `uv pip install --reinstall "av>=18.1"`. If you deliberately self-build, point `INCLUDE` / `LIB` at the BtbN build as in Section 4 (B).

- **`import av` fails with `ImportError: DLL load failed while importing _core`**
  The self-built wheel was installed without the delvewheel repair. Python 3.8+ does not search PATH for the DLL dependencies of extension modules, so keeping the BtbN `bin\` on PATH is not enough. Reinstall the delvewheel-repaired wheel from Section 4 (B).

- **The PyAV build fails with `fatal error C1083: libavcodec/avcodec.h`**
  The compiler does not know where FFmpeg lives. `PKG_CONFIG_PATH` is ignored on Windows (setup.py never calls pkg-config there). Append the BtbN `include` / `lib` directories to `INCLUDE` / `LIB` as in Section 4 (B).

- **Encoder/decoder initialization fails with a current_ctx-related `TypeError` or similar**
  av is still 18.0.0: either PyPI's old release or the interim self-built wheel, which also reports `18.0.0`. Fix: `uv pip install "av>=18.1"` (Section 4 (A)).

- **Jasna refuses to start: wrong ffprobe version**
  The startup check requires `ffprobe` **major version 8**. Add a v8 build's `bin\` to `PATH` (Section 1.1). The `mkvmerge` check was removed in v0.8.0.

- **The torchcodec backend cannot find the FFmpeg DLLs (`avcodec-62.dll` etc.)**
  torchcodec needs the FFmpeg shared DLLs at runtime. Keep the BtbN shared build's `bin\` on `PATH` (Section 1.1).

### jasna install

- **`uv pip install -e .[dev,nvidia]` fails with `No solution found ... no version of torch==2.12.0+cu130`**
  The PyTorch index is missing. Add `--extra-index-url https://download.pytorch.org/whl/cu130`.

- **`uv pip install -e .[dev,nvidia]` fails with `torch-tensorrt==2.12.0+cu130 ... unsatisfiable`**
  uv's default index strategy is first-index only and refuses to satisfy `torch-tensorrt==2.12.0` with the PyTorch index's `2.12.0+cu130` (local version). The transitive dependency `nvidia-cuda-runtime-cu13==0.0.0a0` is also a prerelease. Add `--index-strategy unsafe-best-match --prerelease=allow` (Section 5).

### Runtime

- **Jasna raises `FileNotFoundError` for a `.pth`/`.onnx`/`.pt`**
  Files missing from `model_weights\`, or the resolver is looking elsewhere. Put the 3 weights in `$Workspace\jasna\model_weights\` (Section 6), or set `JASNA_MODEL_WEIGHTS_DIR` to a folder containing them.

- **Startup rejects the NVIDIA driver version**
  The Windows minimum is **610** (raised from the v0.7.2+modi era's 591.67+). Update the driver from NVIDIA.

- **The first launch is extremely slow**
  Not a bug. The initial TensorRT engine compilation takes 15–60 minutes. The engines are cached next to the weights inside `model_weights\` and later runs are fast.

---

## Appendix A: build/runtime modifications on this branch

Changes already applied to this branch's source (reference description; no standalone `.patch` files are shipped).

### A.1 Automatic resolution of the `model_weights/` directory

**Problem**: The defaults of `--detection-model-path` / `--restoration-model-path` were the relative `Path("model_weights")`. Running Jasna from an arbitrary folder looked for `<CWD>/model_weights/...` and raised `FileNotFoundError`.

**Fix**: The new `jasna/model_weights_resolver.py` searches for the `model_weights/` folder in priority order:

1. **the folder named by the `JASNA_MODEL_WEIGHTS_DIR` environment variable**: an explicit user choice wins
2. **`model_weights\` next to the executable**: the standard layout of a packaged install
3. **`model_weights\` in the current directory**: wherever you are right now
4. **`model_weights\` next to the jasna source tree**: for developers running via `uv pip install -e .`

The first hit wins; the chosen location is logged at startup with `--log-level info`. `main.py`, `mosaic/detection_registry.py`, `engine_paths.py`, and the GUI (`gui/processor.py`, `gui/engine_preflight.py`) go through the resolver instead of a hardcoded `Path("model_weights")`.

### A.2 DLL load assistance (torchcodec backend only)

With v0.8.0 the native path (PyAV) no longer needs DLL assistance. Today the torchcodec backend modules (`jasna/media/torchcodec_decoder.py` / `torchcodec_encoder.py`) register the torchcodec package directory and `CUDA_PATH\bin` (when set) on the DLL search path, **on Windows only**.

### A.3 BasicVSR++ benchmark TRT API fix

Updates `jasna/benchmark/basicvsrpp_restoration.py` to the new `_preprocess_engine` API (feat_extract + flow merged into one TRT engine) so `--benchmark basicvsrpp` works. Cross-platform.

### Full diff

The complete set of changes on this branch can be inspected by comparing against upstream (`git diff upstream/main..modi`).
