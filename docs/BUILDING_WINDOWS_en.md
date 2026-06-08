# Building Jasna for Windows

How to set up the Jasna build dependencies on Windows and run jasna **from source**.

> **This guide covers the `v0.7.1+modi` branch.** It builds jasna against the GPU stack — **torch 2.12.0+cu130 / torchvision 0.27.0+cu130 / torch-tensorrt 2.12.0+cu130 / tensorrt 10.16.1.11** — already pinned in `pyproject.toml` on this branch. TensorRT stays on the **10.16** line because `torch-tensorrt==2.12.0` requires `tensorrt>=10.16.1,<10.17.0`; TensorRT 11 is not yet supported by torch-tensorrt.
>
> **New on this branch:** AV1 output, 8-bit (NV12) output, and BT.601/BT.2020 colorspace preservation (see [CODECS_AND_COLORSPACE_en.md](CODECS_AND_COLORSPACE_en.md)); and 2x/4x frame generation via RIFE (see [FRAME_GENERATION_en.md](FRAME_GENERATION_en.md)).

> **Packaging note:** This public fork builds the native GPU dependencies and runs jasna **from source**. It does **not** ship Nuitka packaging tooling to produce a frozen/packaged binary — that tooling is private (the same arrangement as upstream). If you want a pre-packaged binary instead of running from source, use upstream Kruk2/jasna's official releases. See [Packaging / frozen builds](#10-packaging--frozen-builds).

---

## 1. Prerequisites

> ⚠ **Important note up front about where to install ffmpeg**: `vali/src/CMakeLists.txt` hardcodes `FFMPEG_ROOT` as `C:/Program Files/ffmpeg8`, and this is **not** overridable via `-DFFMPEG_ROOT=...` (CMAKE_ARGS). The simplest path is therefore to **extract the ffmpeg dev build to `C:\Program Files\ffmpeg8\`**. If you already have ffmpeg elsewhere, or you cannot write to `C:\Program Files\`, you'll need to either create a junction or edit one line of vali's source (see Section 5.2). Decide on the install location now — discovering this mid-build forces a rebuild.

| Category | Requirement | winget ID / Source | Notes |
|---|---|---|---|
| OS | Windows 10/11 x64 | — | PowerShell 7+ recommended |
| Git | Git for Windows | `Git.Git` | Used for fetching submodules |
| uv | Latest | `astral-sh.uv` | Manages Python automatically (see below) |
| Python | 3.13+ | — (managed by uv) | uv auto-installs the required Python when you run `uv venv --python 3.13`; no need to install Python separately |
| CUDA Toolkit | **13.2** | NVIDIA official ([developer.nvidia.com](https://developer.nvidia.com/cuda-downloads)) | Default `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`. `CUDA_PATH` must point to 13.2 (Section 3.1). winget is discouraged because the package is large |
| NVIDIA Driver | 591.67+ (59x series) | NVIDIA official or GeForce Experience | GPU must be compute capability 7.5+ |
| VS Build Tools | **2022** (Desktop development with C++) | `Microsoft.VisualStudio.2022.BuildTools` | Required for the C++ builds of `vali` / `PyNvVideoCodec` |
| ffmpeg / ffprobe | **v8 shared dev build** | `Gyan.FFmpeg.Shared` | Extract a dev build that contains `include/`, `lib/`, `bin/` to **`C:\Program Files\ffmpeg8\`** (workaround needed for other paths, as noted above). `vali` / `PyNvVideoCodec` link against this at build time (details in Section 5); the v8 CLI is also required at runtime |
| MKVToolNix | `mkvmerge` (runtime) | [mkvtoolnix.download](https://mkvtoolnix.download/) | Needed on `PATH` when running jasna, not for building the dependencies |

Run any step that requires a C++ compile inside a **Developer PowerShell for VS 2022** session, or in a shell where `vcvars64.bat` has been sourced.

### 1.1 One-shot install via winget

Everything except the CUDA Toolkit and NVIDIA Driver can be installed at once via winget:

```powershell
winget install --id Gyan.FFmpeg.Shared              -e --source winget
winget install --id Git.Git                         -e --source winget
winget install --id astral-sh.uv                    -e --source winget
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --silent `
  --override "--wait --quiet --add ProductLang En-us --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Caveats:

- **ffmpeg**: `Gyan.FFmpeg.Shared` installs to winget's default location (something like `C:\Users\<user>\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_*\ffmpeg-*-full_build-shared\`), **not** `C:\Program Files\ffmpeg8\`. After installing, locate the actual path with `Get-ChildItem $env:LOCALAPPDATA\Microsoft\WinGet\Packages\ -Filter "Gyan.FFmpeg*" -Recurse -Directory` etc., and create a junction at `C:\Program Files\ffmpeg8\` pointing to it (Section 5.2 (A)). That's the easiest fix.
- **Python is not required**: uv will fetch it automatically the first time you run `uv venv --python 3.13`. To pre-install explicitly, use `uv python install 3.13`.
- **CUDA Toolkit and NVIDIA Driver must be installed manually**. CUDA requires a pinned version (13.2) which is hard to manage via the `Nvidia.CUDA` winget package, so downloading from NVIDIA directly is more reliable.
- **MKVToolNix is not needed to build the dependencies**, but `mkvmerge` must be on `PATH` when you actually run jasna (install MKVToolNix and add it to `PATH`).

---

## 2. Clone the repositories

Place the three repositories **under the same parent directory**. The rest of this guide refers to that working root via the PowerShell variable `$Workspace`. **Replace the path with your own** (the example uses `C:\jasna-dev`).

```powershell
$Workspace = "C:\jasna-dev"          # ← any directory of your choice
mkdir $Workspace -Force | Out-Null
cd $Workspace

git clone https://codeberg.org/Kruk2/vali.git
git clone https://codeberg.org/Kruk2/PyNvVideoCodec.git
git clone -b modi https://github.com/sh202603/jasna.git   # the modi branch, this fork's default (patches pre-applied)

# vali has a submodule (extern/dlpack), initialize it
cd vali
git submodule update --init --recursive
cd ..
```

Layout:

```
<Workspace>\
  vali\
  PyNvVideoCodec\
  jasna\        <- the working root from here on
```

> Whenever you run a command that uses `$Workspace` in a **new PowerShell session, re-set `$Workspace = "..."`** (the helper script in Section 3.5 can automate this).

---

## 3. Build environment setup (env vars + VS Build Tools)

Set up the following **in the same PowerShell session** to support the C++ builds of `vali` / `PyNvVideoCodec`. Sections 4 onward should be run in that same session.

### 3.0 Build env-var quick reference

The full list of environment variables used in the setup. See each section for details. **The helper script in Section 3.5 sets them all at once.**

| Env var | Example value | Consumer | Set in section |
|---|---|---|---|
| `CUDA_PATH` / `CUDA_HOME` | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2` | nvcc, vali CMake, jasna runtime | 3.1 |
| `PATH` (prepend) | `<CUDA_PATH>\bin;<CUDA_PATH>\libnvvp;<ffmpeg>\bin` | shell | 3.1 / 3.2 |
| `DISTUTILS_USE_SDK` | `1` | setuptools | 3.4 |
| `CMAKE_GENERATOR` | `Ninja` | CMake | 3.4 |
| `CL` | `/utf-8` | MSVC `cl.exe` (required for PyNvVideoCodec build) | 3.4 |
| `FFMPEG_DIR` | `C:/Program Files/ffmpeg8` **(★forward slashes)** | PyNvVideoCodec CMake | 5 / 7 |
| `CMAKE_ARGS` | `-DFFMPEG_ROOT=H:/ffmpeg` | (optional: when building vali with ffmpeg in a non-default path) | 5.2 |

### 3.1 Pin CUDA_PATH

This project uses **CUDA 13.2**. Set `CUDA_PATH` so nvcc/vali pick up the right toolkit, and so the jasna runtime can locate the CUDA `bin/` directory for the native libraries (Section 6 / Appendix A.2).

> Note: `torch==2.12.0+cu130` in `pyproject.toml` is built against the CUDA 13.0 ABI, but the CUDA 13.x line is minor-version compatible and runs fine on a 13.2 runtime.

```powershell
$env:CUDA_PATH      = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:CUDA_HOME      = $env:CUDA_PATH
$env:PATH           = "$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:PATH"
```

Verify:

```powershell
nvcc --version          # should show "release 13.2"
Get-Command nvcc | Select-Object Source
```

To make it permanent, register it as a user env var with `setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"` (takes effect in new sessions).

### 3.2 Add ffmpeg to PATH (required for running jasna)

When running jasna from source, the v8 `ffmpeg`/`ffprobe` CLI must be on `PATH` (there is no bundled copy). Add the ffmpeg `bin` directory:

```powershell
$env:PATH = "C:\Program Files\ffmpeg8\bin;$env:PATH"
ffmpeg -version          # should show ffmpeg version 8.x
```

> If ffmpeg is in a non-default location, the **build-time** settings (vali / PyNvVideoCodec) are described in Section 5.

### 3.3 Load the VS Build Tools environment

This brings the C++ compiler (`cl.exe`) and the Windows SDK into the current shell session. Required for `pip install`-ing `vali` / `PyNvVideoCodec`.

**From PowerShell (recommended):**

```powershell
& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1' `
    -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
```

> `Launch-VsDevShell.ps1` by default `cd`s into the VS source tree. `-SkipAutomaticLocation` preserves the current directory.

**From cmd.exe:**

```cmd
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

> If you're using a different edition (Community / Professional / Enterprise), substitute the `BuildTools` part accordingly.

Verify:

```powershell
cl.exe                  # should print the Microsoft (R) C/C++ Optimizing Compiler banner
where.exe cl.exe        # should resolve to a path under the MSVC toolset
```

### 3.4 Auxiliary build env vars

These ensure scikit-build / setuptools reliably use MSVC.

```powershell
$env:DISTUTILS_USE_SDK = "1"        # tell setuptools to honor the existing MSVC environment
$env:CMAKE_GENERATOR   = "Ninja"    # pin Ninja as the CMake generator (optional)
$env:CL                = "/utf-8"   # avoid C4819 → C2220 on a Japanese-Windows (CP932) host (required for PyNvVideoCodec)
```

> **Why `$env:CL = "/utf-8"` matters**: On a Japanese version of Windows the system code page is CP932 (Shift-JIS). CUDA / ffmpeg headers contain UTF-8 author names with accents etc. that can't be represented in CP932, so MSVC emits the `C4819` warning. `PyNvVideoCodec` builds with `/WX` (warnings-as-errors), so the warning is promoted to `error C2220` and the build fails. Passing `/utf-8` makes MSVC interpret sources as UTF-8 and the warning never fires. The `CL` env var applies to every `cl.exe` invocation, including those launched by `nvcc` via `-Xcompiler`.

### 3.5 Session loader helper script (optional)

To avoid retyping the commands, save the following as **`$Workspace\setup-build-env.ps1`** and dot-source it on every build: `. $Workspace\setup-build-env.ps1` (or `cd $Workspace; . .\setup-build-env.ps1`).

`$Workspace` is derived from the script's own location, so moving or copying the workspace doesn't require editing the script.

```powershell
# setup-build-env.ps1
$Workspace                 = $PSScriptRoot          # treat the script directory as Workspace

$env:CUDA_PATH             = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:CUDA_HOME             = $env:CUDA_PATH
$env:DISTUTILS_USE_SDK     = "1"
$env:CMAKE_GENERATOR       = "Ninja"
$env:CL                    = "/utf-8"                          # MSVC character-set workaround for Japanese Windows
$env:FFMPEG_DIR            = "C:/Program Files/ffmpeg8"        # for PyNvVideoCodec CMake; ★forward slashes required (see Section 7)
$env:PATH                  = "$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;C:\Program Files\ffmpeg8\bin;$env:PATH"

& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1' `
    -Arch amd64 -HostArch amd64 -SkipAutomaticLocation

# Re-export $Workspace into the calling session
Set-Variable -Name Workspace -Value $Workspace -Scope Global
```

---

## 4. Create the Python virtual environment

Create a `.venv` under the `jasna` repository. All subsequent `pip` / `uv pip` commands should be run with this venv activated.

```powershell
cd $Workspace\jasna
uv venv --python 3.13
.\.venv\Scripts\Activate.ps1
```

Install the build-side tools first (so that `--no-build-isolation` picks them up from the venv):

```powershell
uv pip install cmake ninja scikit-build "setuptools<80" wheel numpy
```

> `setuptools` ≥ 80 no longer ships `pkg_resources` by default, and there is no standalone `pkg_resources` package on PyPI. `vali` / `PyNvVideoCodec` have `from pkg_resources import ...` in their `setup.py`, so you need to pin `setuptools<80` (currently resolves to e.g. `79.0.1`).

---

## 5. Place an ffmpeg v8 shared dev build

The next-section build of `vali` links against ffmpeg's headers and import libraries, so **a "shared dev build" with `include/` and `lib/` (in addition to `bin/`) is required**. The same `bin/` provides the v8 `ffmpeg`/`ffprobe` CLI that jasna needs on `PATH` at runtime (Section 3.2).

### 5.1 Obtain and extract

- Recommended sources: the "shared" build from [gyan.dev (Windows builds)](https://www.gyan.dev/ffmpeg/builds/) (e.g. `ffmpeg-8.x-full_build-shared.7z`), or [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases) `*-win64-lgpl-shared-8.x.zip`
- Extract to: **`C:\Program Files\ffmpeg8\`** (as noted below, vali's CMake hardcodes this path, so other locations need a workaround)
- Required layout after extraction:
  ```
  C:\Program Files\ffmpeg8\
    bin\        # ffmpeg.exe, ffprobe.exe, avcodec-*.dll, avformat-*.dll, ...
    include\    # libavcodec\, libavformat\, libavutil\, libswresample\, libswscale\
    lib\        # avcodec.lib, avformat.lib, avutil.lib, swresample.lib, swscale.lib
  ```

### 5.2 If you must use a non-default location

`vali/src/CMakeLists.txt` runs `set(FFMPEG_ROOT "C:/Program Files/ffmpeg8")` **without an `if(NOT DEFINED ...)` guard**, so passing `-DFFMPEG_ROOT=...` via CMAKE_ARGS gets overwritten. Workarounds:

**(A) Create a junction (recommended; requires admin)**

```powershell
# From an elevated PowerShell
New-Item -ItemType Junction -Path "C:\Program Files\ffmpeg8" -Target "H:\ffmpeg"
```

**(B) Edit a single line of vali's source to make it conditional**

```powershell
cd $Workspace\vali
(Get-Content src\CMakeLists.txt) -replace `
  '^set\(FFMPEG_ROOT "C:/Program Files/ffmpeg8"\)$', `
  "if(NOT DEFINED FFMPEG_ROOT)`r`n  set(FFMPEG_ROOT `"C:/Program Files/ffmpeg8`")`r`nendif()" |
  Set-Content src\CMakeLists.txt

$env:CMAKE_ARGS = "-DFFMPEG_ROOT=H:/ffmpeg"   # forward slashes
```

### 5.3 Two build-time consumers of the ffmpeg directory

During the build, `vali` and `PyNvVideoCodec` each look up ffmpeg via a different variable. If you stick with the default path (`C:\Program Files\ffmpeg8`), the "Default behavior" column applies; if you place it elsewhere, **you must configure each consumer individually**.

| Consumer | Variable read | Default behavior | When using a non-default path |
|---|---|---|---|
| **`vali` CMake** | `FFMPEG_ROOT` (CMake) | `vali/src/CMakeLists.txt` hardcodes `C:/Program Files/ffmpeg8` (no guard) | Section 5.2 (junction or edit vali's source) |
| **`PyNvVideoCodec` CMake** | `FFMPEG_DIR` (env var) | If unset, falls back to the bundled `external/ffmpeg/` (no `lib/` → build fails) | Set `$env:FFMPEG_DIR = "C:/Program Files/ffmpeg8"` (★forward slashes; see Section 7) |

> At runtime, jasna additionally needs the v8 `ffmpeg`/`ffprobe` CLI on `PATH` — add the `bin/` directory (Section 3.2).

---

## 6. Build and install vali

> **Prerequisites**: Section 3 (CUDA_PATH / VS Build Tools / env vars) + Section 4 (venv + `setuptools<80`) + Section 5 (ffmpeg dev build at `C:\Program Files\ffmpeg8\` or via junction) must be complete.

```powershell
cd $Workspace\vali
uv pip install . --no-build-isolation
```

Verify:

```powershell
python -c "import os; os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'],'bin')); import python_vali; print(python_vali.__file__)"
```

---

## 7. Build and install PyNvVideoCodec

> **Prerequisites**: Same as Section 6 (CUDA / VS Build Tools / venv / ffmpeg dev build), plus `$env:CL = "/utf-8"` (3.4).

PyNvVideoCodec's CMake reads the **environment variable `FFMPEG_DIR`**. When unset, it falls back to the bundled `external/ffmpeg/` (which only has headers and source, no `lib/`) and fails with `avcodec_library-NOTFOUND`. Always point it at your ffmpeg dev build's top directory **with forward slashes** before building:

```powershell
$env:FFMPEG_DIR = "C:/Program Files/ffmpeg8"   # ★forward slashes are required
$env:CL         = "/utf-8"                     # required on Japanese Windows (see Section 3.4)
cd $Workspace\PyNvVideoCodec
uv pip install . --no-build-isolation
```

> **Forward slashes are required**: At install time, CMake embeds the `FFMPEG_DIR` value into a generated script as a quoted string. A path with backslashes (`C:\Program Files\ffmpeg8`) makes CMake see `\P` and similar as invalid escape sequences, failing with `Syntax error in cmake code`.

> Note that this is a different variable name and a different mechanism from vali's `FFMPEG_ROOT` (CMake variable). PyNvVideoCodec only looks at the env var.

> On Japanese Windows, without `$env:CL = "/utf-8"` you'll hit `error C2220: warning treated as error` / `warning C4819`. See Section 3.4.

> If you change settings and want to rebuild, wipe the old cache first: `Remove-Item -Recurse -Force _skbuild`

Verify:

```powershell
python -c "import PyNvVideoCodec; print(PyNvVideoCodec.__file__)"
```

---

## 8. Install jasna itself

> **Prerequisites**: `python_vali` (Section 6) and `PyNvVideoCodec` (Section 7) must be installed in the venv.

`pyproject.toml` depends on `torch==2.12.0+cu130` / `torchvision==0.27.0+cu130` / `torch-tensorrt==2.12.0`. These are not on the default PyPI, so you need to **point uv at the PyTorch wheel index and add two more flags**:

```powershell
cd $Workspace\jasna
uv pip install -e .[dev] `
    --extra-index-url https://download.pytorch.org/whl/cu130 `
    --index-strategy unsafe-best-match `
    --prerelease=allow
```

Why each flag is needed:

- `--extra-index-url https://download.pytorch.org/whl/cu130`: source for the `torch+cu130` / `torchvision+cu130` / `torch-tensorrt+cu130` wheels
- `--index-strategy unsafe-best-match`: required so uv can satisfy the pyproject constraint `torch-tensorrt==2.12.0` with `torch-tensorrt==2.12.0+cu130` (a local-version-bearing release) on the PyTorch index. The default first-index strategy rejects it.
- `--prerelease=allow`: required because the transitive dep `nvidia-cuda-runtime-cu13==0.0.0a0` is a pre-release.

The `[dev]` extra installs `nuitka>=2.4`, `pytest`, `pytest-cov`, `scikit-build`, `cmake`, `ninja`.

Smoke test:

```powershell
jasna --help
```

### 8.1 Apply the mmengine patch (torch 2.6+ compatibility)

Adds `weights_only=False` to the `torch.load` call inside `mmengine.runner.checkpoint`. Starting with torch 2.6, the default became `weights_only=True`, so without this patch existing `.pth` checkpoints fail to load.

Apply `patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff` to the `site-packages` inside the venv. Use the Python `patch` package as a temporary install so you don't need `patch.exe` on the system.

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
# → one line should match if applied successfully
```

> Every time you re-run `uv pip install -e .[dev]` it will reinstall `mmengine` and clobber this patch. Re-apply afterwards.

### 8.2 ONNX packages (for the YOLO detection model)

If you use a **YOLO (lada-yolo-\*) detection model**, ultralytics exports the model to ONNX before building the TensorRT engine. This needs three packages that are not pulled in automatically. (RF-DETR models do not need them — TensorRT parses the prebuilt `.onnx` directly.)

```powershell
cd $Workspace\jasna
uv pip install onnx onnxslim onnxruntime
```

When running from source, ultralytics auto-downloads these on the first export if they are missing — but installing them ahead of time avoids a stall on the first YOLO run.

> **If they are missing**, running `jasna --detection-model lada-yolo-v4 ...` aborts during engine compilation with `ERROR ONNX: export failure ... No module named 'onnx'` → `RuntimeError: Engine compilation subprocess failed`. Install the three packages into the venv to fix it.

---

## 9. Place model weights and assets

Drop the following 3 files into `$Workspace\jasna\model_weights\`:

- `lada_mosaic_restoration_model_generic_v1.2.pth`
- `rfdetr-v5.onnx`
- `lada_mosaic_detection_model_v4_fast.pt`

When running from source, jasna resolves `model_weights\` automatically (Appendix A.1), so the files just need to live here.

The 2 test clips usually ship with the repository under `$Workspace\jasna\assets\` (`test_clip1_1080p.mp4`, `test_clip1_2160p.mp4`); if they are absent, extract them from an upstream release.

### 9.1 Optional: RIFE frame-interpolation model (for `--frame-gen`)

Only needed for frame-rate up-conversion (`--frame-gen {2x,4x}`). The RIFE weights are **not bundled** (they carry non-commercial terms), so you create a TorchScript checkpoint yourself with `make_rife_torchscript.py`.

1. Clone Practical-RIFE and download a 4.x model package (verified with **v4.25**) so that `<repo>\train_log\` contains `RIFE_HDv3.py`, `IFNet_HDv3.py`, `flownet.pkl`:
   ```powershell
   git clone https://github.com/hzwer/Practical-RIFE
   # download + unzip the model package into Practical-RIFE\train_log\ per its README
   ```
2. Convert to TorchScript using the project venv (run from `$Workspace\jasna`):
   ```powershell
   .\.venv\Scripts\python.exe make_rife_torchscript.py `
       --rife-repo C:\path\to\Practical-RIFE `
       --output model_weights\rife.pth --validate
   ```

Running **from source** picks up `model_weights\rife.pth` automatically (same resolver as the other weights; or point `JASNA_MODEL_WEIGHTS_DIR` at a folder containing it). Full procedure: [docs/FRAME_GENERATION_en.md](FRAME_GENERATION_en.md).

---

## 10. Packaging / frozen builds

There is currently **no public way to produce a packaged/frozen binary from this fork.** Upstream switched its build from PyInstaller to **Nuitka**, but it does not publish any Nuitka build script or instructions — the actual packaging tooling lives in a **private submodule (`jasna/protection`)** that is not part of this public fork. The fork's old PyInstaller build scripts (`build_exe.py`, `jasna.spec`) have been **removed**.

The supported public path is therefore to **run jasna from source** (Section 11). If you need a pre-packaged binary, use upstream Kruk2/jasna's official releases.

---

## 11. Run from source / verify

With Sections 1–9 complete, run jasna directly from the source checkout in the venv:

```powershell
cd $Workspace\jasna
jasna --version          # -> 0.7.1+modi
jasna --help
jasna --input assets\test_clip1_1080p.mp4 --output $env:TEMP\out.mp4   # process a short clip
jasna                    # launches the GUI (no args)
```

`model_weights\` is resolved automatically (Appendix A.1). Make sure the v8 `ffmpeg`/`ffprobe` CLI and `mkvmerge` are on `PATH` (Sections 3.2, 1), and that `CUDA_PATH` points at CUDA 13.2 so the native libraries can find the CUDA `bin\` directory (Appendix A.2).

---

## Troubleshooting

### Environment / setup

- **`uv pip install .` fails with a CMake error or `cl.exe not found`**
  You're running in a session where Section 3.3 (the VS Build Tools load) wasn't done. Check `where.exe cl.exe` to confirm `cl.exe` is on `PATH`.

- **CMake detects the wrong CUDA version (e.g. 13.0 when you expect 13.2)**
  Either `CUDA_PATH` (Section 3.1) isn't set, or another version is earlier in `PATH`. Inspect `Get-Command nvcc | Select-Object Source` to see which one is being picked up.

- **`vali` / `PyNvVideoCodec` build fails with `ModuleNotFoundError: No module named 'pkg_resources'`**
  `setuptools` ≥ 80 dropped `pkg_resources` from its default install, and PyPI has no standalone `pkg_resources` package (`uv pip install pkg_resources` also fails with "No solution found"). Downgrade setuptools:
  ```powershell
  uv pip install "setuptools<80"     # e.g. 79.0.1
  ```
  Then re-run `uv pip install . --no-build-isolation`. If you pinned `"setuptools<80"` up front in Section 4, this never comes up.

### vali build

- **`vali` build fails with CMake errors like `AVCODEC_INCLUDE_DIRS / AVCODEC_LIB ... NOTFOUND`**
  Either the ffmpeg **shared dev build** isn't extracted to `C:\Program Files\ffmpeg8\`, or only the `bin/` part is present. Lay it out per Section 5 with `include/`, `lib/`, `bin/`. `vali/src/CMakeLists.txt` hardcodes `FFMPEG_ROOT` to `C:/Program Files/ffmpeg8` and **does not honor `-DFFMPEG_ROOT=...` (`CMAKE_ARGS`)**. If you must keep ffmpeg elsewhere, use the junction or the vali source edit from Section 5.2.

### PyNvVideoCodec build

- **`PyNvVideoCodec` build fails with CMake errors like `avcodec_library-NOTFOUND`**
  PyNvVideoCodec reads the **environment variable `FFMPEG_DIR`**. When unset, it falls back to the bundled `external/ffmpeg/` (no `lib/`) and fails. Per Section 7, set `$env:FFMPEG_DIR = "C:/Program Files/ffmpeg8"` before building (forward slashes required). This is a different variable from vali's `FFMPEG_ROOT` (which is a CMake variable).

- **`PyNvVideoCodec` build fails at install with `Syntax error in cmake code ... Invalid character escape '\P'`**
  `$env:FFMPEG_DIR` was set with a backslash path (`C:\Program Files\ffmpeg8`). CMake's install script treats it as a quoted string and rejects `\P` as an invalid escape. Switch to forward slashes (`"C:/Program Files/ffmpeg8"`), delete `_skbuild`, and rebuild.

- **On Japanese Windows, `PyNvVideoCodec` build fails with `error C2220: warning treated as error` / `warning C4819: <can't be represented in code page (932)>`**
  CUDA / ffmpeg headers contain UTF-8 characters (accented author names etc.) that can't be expressed in CP932; MSVC emits `C4819`, and `/WX` promotes it to an error. Set `$env:CL = "/utf-8"` before building — MSVC then interprets sources as UTF-8 and the warning vanishes. Setting it once for the whole build session (per Section 3.4) is the easy approach.

### jasna install

- **`uv pip install -e .[dev]` fails with `No solution found ... no version of torch==2.12.0+cu130`**
  The PyTorch index wasn't specified. Add `--extra-index-url https://download.pytorch.org/whl/cu130`.

- **`uv pip install -e .[dev]` fails with `torch-tensorrt==2.12.0+cu130 ... unsatisfiable`**
  uv's default index strategy is first-index-only, and it refuses to satisfy the pyproject constraint `torch-tensorrt==2.12.0` with the PyTorch index's `2.12.0+cu130` (a local-version release). On top of that, the transitive dep `nvidia-cuda-runtime-cu13==0.0.0a0` is a pre-release. Add `--index-strategy unsafe-best-match --prerelease=allow` (see the command in Section 8).

### Runtime (running from source)

- **Jasna fails with `FileNotFoundError` for a `.pth`/`.onnx`/`.pt`**
  A file in `model_weights\` is missing, or the resolver looked in the wrong place. Make sure the 3 weight files are in `$Workspace\jasna\model_weights\` (Section 9), or set `JASNA_MODEL_WEIGHTS_DIR` to the folder that holds them.

- **Jasna refuses to start: ffmpeg/ffprobe wrong version, or mkvmerge missing**
  Jasna requires `ffmpeg`/`ffprobe` **major version 8** and `mkvmerge` on `PATH`. Add the ffmpeg `bin\` directory (Section 3.2) and install MKVToolNix (Section 1).

- **`import python_vali` fails with a DLL load error**
  Confirm that `CUDA_PATH` points at CUDA 13.2. The native libs also need the CUDA `bin\` directory registered on the DLL search path; jasna does this automatically (Appendix A.2), but for a standalone import call `os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'],'bin'))` first (the vali README documents this).

- **The very first launch is extremely slow**
  This is normal. The first-time TensorRT engine compilation takes 15–60 minutes. The result is cached under `~/.jasna/engines/` and subsequent launches are fast.

---

## Appendix A: build/runtime changes on this branch

A breakdown of the Windows-build/runtime fixes, already applied to this branch's source (these are documented here for reference; no standalone `.patch` file is shipped).

### A.1 Auto-resolution of the `model_weights/` directory

**Problem**: The defaults for `--detection-model-path` / `--restoration-model-path` are the relative path `Path("model_weights")`. If you run jasna from another folder, it looks under `<CWD>/model_weights/...` and dies with `FileNotFoundError`.

**Fix**:

- **New file `jasna/model_weights_resolver.py`**: searches for the `model_weights/` folder in this priority order:
  1. **The folder named by the `JASNA_MODEL_WEIGHTS_DIR` env var** — highest priority, an explicit "use this" from the user
  2. **A `model_weights\` folder next to the running executable** — the standard layout for a packaged install
  3. **A `model_weights\` folder in the directory you ran the command from** — wherever `cd` put you in PowerShell
  4. **A `model_weights\` folder beside the jasna source tree** — for developers running from a `uv pip install -e .` checkout

  The first one found wins. The resolved path is logged at startup with `--log-level info` (duplicate entries are suppressed).
- **`jasna/main.py`**: `--restoration-model-path`'s default changed to `""`. When unset, resolution goes through the resolver. Help text mentions the search order.
- **`jasna/mosaic/detection_registry.py`**: `detection_model_weights_path()` / `discover_available_detection_models()` go through the resolver.
- **`jasna/engine_paths.py`**: `UNET4X_ONNX_PATH` is computed from the resolver.
- **`jasna/gui/processor.py`, `jasna/gui/engine_preflight.py`**: hardcoded `Path("model_weights")` calls in the GUI replaced with resolver calls.

### A.2 DLL load helper for Windows runtime

**Problem**: On Windows the DLL search path is restricted, and dependent DLLs (including CUDA) might not be found when `python_vali` is loaded.

**Fix** (`jasna/media/video_decoder.py`):

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

Before `python_vali` is loaded, register both its sibling DLL directory and the CUDA `bin/` directory on the DLL search path.

### A.3 Update the BasicVSR++ benchmark to the new TRT API

**Problem**: `jasna/restorer/basicvsrpp_sub_engines.py` already uses the newer `_preprocess_engine` API (which fuses feat_extract and flow into a single TRT engine), but the benchmark code in `jasna/benchmark/basicvsrpp_restoration.py` was still calling the old `_feat_extract_engine` + separate-flow API. Running `--benchmark basicvsrpp` therefore failed with `AttributeError`.

**Fix** (`jasna/benchmark/basicvsrpp_restoration.py`):

- Replace `split._feat_extract_engine` + a separate `compute_flow` call with a single `split._preprocess_engine` call.
- Remove the now-unused `torch.nn.functional` import.

### Full diff

10 files (9 modifications + 1 new), about 290 lines. For the complete content, diff this branch against upstream (`git diff upstream/main..modi`).
