# Building the frozen distribution (Windows, experimental)

This fork can produce a standalone frozen distribution at `dist_nuitka\jasna\` with a single `jasna.exe`: run it **with arguments** for the CLI, **without arguments** for the GUI (the console window is detached automatically). End users need no Python installation.

Background: upstream switched its packaging from PyInstaller to Nuitka, but its packaging tooling lives in a private submodule (`jasna/protection`) and is not part of this fork. `scripts\build_nuitka.py` is this fork's own, self-contained replacement. It is currently Windows-only.

> **The bundle is for use on your own machines (or within your organization) — do not redistribute it to third parties.** jasna itself is AGPL-3.0, so passing the bundle on would obligate you to provide the complete corresponding source including your modifications. More restrictively, the bundle contains NVIDIA proprietary components (the TensorRT runtime libraries, `nvvfx`) whose license terms do not cleanly permit passing them on as part of this bundle. RIFE weights (`model_weights\rife*`) carry a non-commercial license and are therefore **excluded from the bundle by default**; pass `--bundle-rife` only for a build you use yourself.

---

## 1. Prerequisites

- A working **from-source setup** per [BUILDING_WINDOWS_en.md](BUILDING_WINDOWS_en.md): the venv with all runtime dependencies installed, plus the `[dev]` extra (`uv pip install -e .[dev]`), which provides `nuitka>=2.4`. Visual Studio Build Tools (MSVC) must be available — Nuitka compiles jasna to C.
- **CUDA Toolkit 13.x installed and `CUDA_PATH` set.** The NPP / nvJPEG runtime DLLs (`nppc64_13.dll`, `nppicc64_13.dll`, ..., `nvjpeg64_13.dll`) are copied from `%CUDA_PATH%\bin\x64` into the distribution. They are not shipped by any pip wheel, and the frozen app deliberately strips the CUDA Toolkit from `PATH` at runtime, so a build without them produces a distribution whose video decode fails on machines without a local CUDA Toolkit.
- The required model weights in `model_weights\`:
  `lada_mosaic_restoration_model_generic_v1.2.pth`, `rfdetr-v5.onnx`, `lada_mosaic_detection_model_v4_fast.pt`.
- Optional: `ffmpeg` / `ffprobe` (major version 8) and `mkvmerge` on `PATH`. If found, they are bundled into `tools\` and `mkvtoolnix\`; if not, the build warns and the frozen app falls back to the end user's `PATH` at runtime.
- Disk: the finished distribution is **~8 GB** (torch + CUDA stack dominate).

## 2. Building

From the project root, in the venv:

```powershell
.\.venv\Scripts\python.exe scripts\build_nuitka.py               # full build
.\.venv\Scripts\python.exe scripts\build_nuitka.py --skip-nuitka # reuse the compiled exe, redo only the bundle steps
.\.venv\Scripts\python.exe scripts\build_nuitka.py --bundle-rife # also bundle RIFE weights (personal-use builds only)
```

The first build takes noticeably longer (MSVC compiles ~150 C files; results are cached by clcache, so rebuilds compile only what changed). Most of the wall-clock time is the ~8 GB copy of third-party packages.

The script ends with an automatic smoke test (`jasna.exe --version` must print the project version, `--help` must print the CLI usage). Output: `dist_nuitka\jasna\`.

## 3. How the build works

The design principle: **only jasna's own code is compiled; every third-party package is copied verbatim.**

- Nuitka runs in standalone mode on `jasna\__main__.py` with `--include-package=jasna` (jasna loads several of its own modules via `importlib`, which static analysis cannot see, so the whole package is force-included). Every third-party top-level package found in site-packages is excluded from compilation via `--nofollow-import-to` and instead copied as-is onto the distribution root.
- This works because a Nuitka standalone binary's `sys.path` is exactly its own directory: flat-copied packages import normally. The flat layout is also the contract expected by `jasna\packaging\windows_dll_paths.py`, which registers `torch\lib`, `tensorrt_libs`, `python_vali`, `PyNvVideoCodec`, `nvvfx`, and top-level `*.libs` directories for DLL search at startup.
- The exe is built **console-subsystem** (`--windows-console-mode=force`) so the CLI blocks the shell and stdout/stderr work; a GUI launch calls `FreeConsole()` itself (`os_utils.drop_console_window`).
- `--deployment` is set because jasna relaunches `sys.executable` (engine-compilation subprocess, multiprocessing spawn); Nuitka's non-deployment self-execution guard would break that.

Three pitfalls the script handles — keep them in mind when touching it:

1. **Stdlib on Nuitka's no-auto-inclusion list.** Nuitka only bundles stdlib modules that *compiled* code imports. Imports made at runtime by the flat-copied packages (e.g. torch importing `unittest.mock` and `uuid`) are invisible to it, and a large blacklist (`unittest`, `uuid`, `logging`, `socket`, `ssl`, ...) is never auto-included. The script therefore includes everything on that list that exists on the platform, minus obvious junk (see `STDLIB_INCLUDE_SKIP`).
2. **`python3.dll`.** Nuitka ships `python313.dll` but not the stable-ABI forwarder `python3.dll`. Extensions built against the limited API (e.g. psutil's `_psutil_windows.pyd`) link `python3.dll` and fail to load without it. The script copies it from the base CPython installation.
3. **CUDA NPP / nvJPEG DLLs** — see Prerequisites above.

TensorRT engines (`*.engine`, `*_sub_engines\`) are **deliberately not bundled**: they are specific to the GPU and TensorRT version and are regenerated into `model_weights\` on the end user's first run (the usual 15–60 min first-run compilation).

## 4. Distribution layout

```
dist_nuitka\jasna\
├── jasna.exe               # single entry point (args → CLI, no args → GUI)
├── python313.dll, python3.dll, vcruntime140*.dll
├── npp*_13.dll, nvjpeg64_13.dll   # CUDA runtime DLLs from the Toolkit
├── torch\, torchvision\, tensorrt_libs\, python_vali\, PyNvVideoCodec\, nvvfx\, ...
├── numpy.libs\, scipy.libs\, av.libs\    # must stay at the root (DLL search contract)
├── *.dist-info\            # kept for importlib.metadata version lookups
├── tcl\, tk\               # tkinter data (tk-inter plugin)
├── model_weights\          # weights, no engines
├── assets\                 # test clips
├── tools\                  # bundled ffmpeg/ffprobe (if found at build time)
└── mkvtoolnix\             # bundled mkvmerge (if found at build time)
```

## 5. Constraints of the frozen build

- **`unet-4x`, SD1.5 inpaint and license activation do not work** — they need the private `jasna/protection` submodule, which is empty in this fork. This matches running from source; the default pipeline (detection + BasicVSR++ + rtx-super-res/tvai secondaries) is unaffected.
- **FP8 recon (`--fp8-recon`) is unverified in the frozen build**: it relies on triton JIT-compiling kernels on the end user's machine.
- The distribution must be extracted to a **user-writable, ASCII-only path** (TRT engines are written into `model_weights\` next to the exe; the ASCII requirement is the same RTX Super-Res limitation as for source installs).
- End users still need an NVIDIA GPU (compute ≥ 7.5) and driver ≥ 590. `ffmpeg`/`mkvmerge` are only needed on their `PATH` if they were not bundled at build time.

## 6. Verifying a build

```powershell
cd dist_nuitka\jasna
.\jasna.exe --version                 # -> 0.7.2+modi (script already asserts this)
.\jasna.exe --input assets\test_clip1_1080p.mp4 --output %TEMP%\out.mkv
```

The first processing run also exercises the engine-compilation subprocess path (`jasna.exe --compile-engines ...` relaunches itself). Then double-click `jasna.exe`: the console window must disappear and the GUI appear. For a realistic end-user test, run on a machine (or shell) whose `PATH` contains neither the venv nor a CUDA Toolkit — the bundled DLLs must be picked up.

## 7. Troubleshooting

- **`Error: No CUDA device` although the GPU works** — this message can be misleading: `os_utils.check_nvidia_gpu()` swallows `ImportError`, so a torch import failure inside the frozen app (typically a missing stdlib module) is reported as "no CUDA". Check for a genuinely missing module first (see next item) before suspecting the GPU stack.
- **`ModuleNotFoundError: <stdlib module>`** — the module is on Nuitka's no-auto-inclusion list and got skipped. Remove it from `STDLIB_INCLUDE_SKIP` (or extend the include logic) in `scripts\build_nuitka.py` and rebuild; only jasna recompiles, so this is quick.
- **`ImportError: DLL load failed while importing <ext>`** — inspect the extension's imports (e.g. with `pefile`). If it links `python3.dll`, the forwarder is missing from the dist root. Otherwise a dependent DLL is not on the frozen app's search path (dist root, `torch\lib`, `*.libs`, System32).
- **`... nppicc64_13.dll was not found`** (or another `npp*`/`nvjpeg` DLL) — the build machine had no `CUDA_PATH` when the script ran; rebuild with the CUDA Toolkit installed.
- **Silent wrong behavior after changing the nofollow logic** — check `build\nuitka\report.xml`: the script warns if any module was compiled from site-packages (it should never happen; third-party code must be copied, not compiled — `cv2` and friends break when compiled).
