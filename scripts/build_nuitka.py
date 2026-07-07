"""Nuitka standalone build for the +modi fork -> dist_nuitka/jasna/.

Strategy: only jasna's own code is compiled by Nuitka; every third-party
package is copied verbatim from site-packages onto the dist root, where the
standalone binary imports it normally (a Nuitka standalone binary's sys.path
is exactly its own directory). This flat layout is the contract expected by
jasna/packaging/windows_dll_paths.py (torch/lib, tensorrt_libs, *.libs, ...
next to the binary) and avoids compiling the GPU stack.

A single console-subsystem `jasna.exe` is produced: with arguments it runs the
CLI, without arguments it detaches the console (FreeConsole) and starts the
GUI (see jasna/__main__.py argv0/arg dispatch).

Run from the project root with the venv interpreter:

    .\\.venv\\Scripts\\python.exe scripts\\build_nuitka.py [--skip-nuitka]

--skip-nuitka reuses an existing build/nuitka/__main__.dist (useful when only
the copy/bundle steps changed).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE_PACKAGES = Path(sysconfig.get_paths()["purelib"])
BUILD_DIR = ROOT / "build" / "nuitka"
REPORT_PATH = BUILD_DIR / "report.xml"
DIST = ROOT / "dist_nuitka" / "jasna"  # dist_nuitka/ is already gitignored

REQUIRED_WEIGHTS = [
    "lada_mosaic_restoration_model_generic_v1.2.pth",
    "rfdetr-v5.onnx",
    "lada_mosaic_detection_model_v4_fast.pt",
]

# Build/dev-only distributions: neither copied to the dist nor bundled.
# setuptools/pkg_resources stay in (torch._inductor / triton / pkg_resources
# consumers may want them at runtime); yapf stays in (mmengine.config uses it).
EXCLUDED_PACKAGES = {
    "nuitka",
    "pyinstaller",
    "_pyinstaller_hooks_contrib",
    "pyinstaller_hooks_contrib",
    "altgraph",
    "pefile",
    "peutils",
    "ordlookup",
    "win32ctypes",
    "pywin32_ctypes",
    "pytest",
    "_pytest",
    "py",
    "pluggy",
    "iniconfig",
    "coverage",
    "pytest_cov",
    "pip",
    "wheel",
    "cmake",
    "ninja",
    "skbuild",
    "scikit_build",
    "pylab",
    "isympy",
    # stray non-package top-level dirs shipped by some wheels
    "benchmarks",
    "include",
    "samples",
    "tests",
    "yapftests",
}
EXCLUDED_DISTS = {
    "nuitka",
    "pyinstaller",
    "pyinstaller_hooks_contrib",
    "altgraph",
    "pefile",
    "pywin32_ctypes",
    "pytest",
    "pytest_cov",
    "py",
    "pluggy",
    "iniconfig",
    "coverage",
    "pip",
    "wheel",
    "cmake",
    "ninja",
    "scikit_build",
    "jasna",  # the app itself is compiled into the binary
}

# Stdlib on Nuitka's no-auto-inclusion list is only bundled when *compiled*
# code imports it — imports made by the flat-copied (nofollow'd) third-party
# packages are invisible to Nuitka, so e.g. torch's `unittest.mock`/`uuid`
# imports came up missing and surfaced as a bogus "No CUDA device" (the
# ImportError is swallowed by os_utils.check_nvidia_gpu). Rather than chase
# one ModuleNotFoundError at a time, include everything on that list that
# exists on this platform, minus obvious junk. Bytecode inclusion is cheap.
STDLIB_INCLUDE_SKIP = {
    "antigravity",
    "this",
    "idlelib",
    "lib2to3",
    "turtle",
    "turtledemo",
    "venv",
    "ensurepip",
    "msilib",
    "site",
    "sitecustomize",
    "distutils",  # gone in 3.13; find_spec only hits the setuptools shim
    "_distutils_system_mod",
    "asyncio.test_utils",
    "importlib._bootstrap",  # frozen into the interpreter itself
    "importlib._bootstrap_external",
    # unix-only
    "curses",
    "_curses",
    "_curses_panel",
    "termios",
    "pty",
    "tty",
    "grp",
    "readline",
    "_posixsubprocess",
    "_crypt",
    "_dbm",
}


def stdlib_include_options() -> list[str]:
    from importlib.util import find_spec

    from nuitka.importing.StandardLibrary import _stdlib_no_auto_inclusion_list

    options: list[str] = []
    for name in sorted(set(_stdlib_no_auto_inclusion_list)):
        if name in STDLIB_INCLUDE_SKIP or name.split(".")[0] in STDLIB_INCLUDE_SKIP:
            continue
        try:
            spec = find_spec(name)
        except (ImportError, ValueError):
            continue
        if spec is None:
            continue
        if spec.submodule_search_locations:
            options.append(f"--include-package={name}")
        else:
            options.append(f"--include-module={name}")
    return options


def log(msg: str) -> None:
    print(f"[build_nuitka] {msg}", flush=True)


def fail(msg: str) -> None:
    raise SystemExit(f"[build_nuitka] ERROR: {msg}")


def preflight() -> None:
    if sys.prefix == sys.base_prefix:
        fail("run this with the project venv interpreter (.venv/Scripts/python.exe)")
    try:
        import nuitka  # noqa: F401
    except ImportError:
        fail("nuitka is not installed in this environment (uv pip install -e .[dev])")
    for name in REQUIRED_WEIGHTS:
        if not (ROOT / "model_weights" / name).is_file():
            fail(f"required model weight missing: model_weights/{name}")


def _dist_info_name(dirname: str) -> str:
    """'torch-2.12.0+cu130.dist-info' -> 'torch' (normalized)."""
    stem = dirname[: -len(".dist-info")]
    return stem.split("-")[0].lower().replace("-", "_")


def enumerate_site_packages() -> tuple[list[Path], list[str]]:
    """Scan site-packages top level.

    Returns (entries to copy onto the dist root, module names to pass to
    --nofollow-import-to). `*.libs` dirs, `*.dist-info` dirs and non-importable
    loose files are copied but not nofollow-listed.
    """
    copy_entries: list[Path] = []
    nofollow: list[str] = []
    for entry in sorted(SITE_PACKAGES.iterdir(), key=lambda p: p.name.lower()):
        name = entry.name
        low = name.lower()
        if name == "__pycache__" or low.startswith(("__editable__", "~", "_virtualenv")):
            continue
        if entry.is_dir():
            if low.endswith(".dist-info"):
                if _dist_info_name(name) not in EXCLUDED_DISTS:
                    copy_entries.append(entry)
                continue
            if low.endswith(".libs"):
                # windows_dll_paths.py scans *.libs directly at the dist root
                copy_entries.append(entry)
                continue
            if low in EXCLUDED_PACKAGES:
                continue
            copy_entries.append(entry)
            nofollow.append(name)
            continue
        # loose top-level files
        suffix = entry.suffix.lower()
        if suffix in {".pth", ".whl", ".virtualenv"}:
            continue  # .pth files are not processed on a plain sys.path entry anyway
        if suffix == ".py":
            if entry.stem.lower() in EXCLUDED_PACKAGES:
                continue
            copy_entries.append(entry)
            nofollow.append(entry.stem)
        elif suffix in {".pyd", ".dll"}:
            copy_entries.append(entry)
            mod = name.split(".")[0]
            if suffix == ".pyd" and mod.isidentifier():
                nofollow.append(mod)
        # anything else (LICENSE stragglers etc.) is skipped silently
    return copy_entries, nofollow


def run_nuitka(nofollow: list[str]) -> None:
    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "--mode=standalone",
        f"--output-dir={BUILD_DIR}",
        "--output-filename=jasna.exe",
        # Console subsystem so the CLI blocks cmd and stdout works; the GUI
        # path calls FreeConsole() itself (os_utils.drop_console_window).
        "--windows-console-mode=force",
        "--enable-plugin=tk-inter",
        # Force-include the whole app: restorer._LAZY_EXPORTS,
        # detection_registry and engine_compiler load jasna.* via importlib,
        # which static analysis cannot see.
        "--include-package=jasna",
        # Empty private submodule; never bundle, the lazy importers degrade
        # gracefully at runtime (unet-4x / SD1.5 / license paths only).
        "--nofollow-import-to=jasna.protection",
        # jasna relaunches sys.executable itself (--compile-engines,
        # multiprocessing spawn); the non-deployment self-execution guard
        # would break that.
        "--deployment",
        f"--report={REPORT_PATH}",
        "--assume-yes-for-downloads",
        f"--jobs={os.cpu_count() or 4}",
        "--lto=no",
    ]
    cmd += stdlib_include_options()
    cmd += [f"--nofollow-import-to={n}" for n in sorted(set(nofollow))]
    cmd.append(str(ROOT / "jasna" / "__main__.py"))
    log(f"running Nuitka ({len(nofollow)} nofollow packages), this takes a while...")
    subprocess.run(cmd, check=True, cwd=ROOT)


def check_report() -> None:
    """Warn if any compiled/bundled module came from site-packages (nofollow leak)."""
    if not REPORT_PATH.is_file():
        log("WARNING: report.xml not found, skipping leak check")
        return
    site = str(SITE_PACKAGES).lower()
    offenders: set[str] = set()
    for elem in ET.parse(REPORT_PATH).getroot().iter("module"):
        source = (elem.get("source_path") or elem.get("filename") or "").lower()
        if site in source:
            offenders.add(elem.get("name") or source)
    if offenders:
        log("WARNING: modules compiled from site-packages (nofollow leak?):")
        for name in sorted(offenders):
            log(f"  - {name}")


def place_dist() -> None:
    src = BUILD_DIR / "__main__.dist"
    if not (src / "jasna.exe").is_file():
        fail(f"Nuitka output not found: {src / 'jasna.exe'}")
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.parent.mkdir(parents=True, exist_ok=True)
    # copy (not move) so --skip-nuitka reruns can reuse build/nuitka/__main__.dist
    shutil.copytree(src, DIST)
    # Nuitka ships python313.dll but not the stable-ABI forwarder python3.dll;
    # flat-copied extensions built against the limited API (e.g. psutil's
    # _psutil_windows.pyd) link python3.dll and fail to load without it.
    py3_dll = Path(sys.base_prefix) / "python3.dll"
    if py3_dll.is_file():
        shutil.copy2(py3_dll, DIST / py3_dll.name)
    else:
        log("WARNING: python3.dll not found next to the base interpreter; "
            "stable-ABI extensions (psutil, ...) will fail to load")
    log(f"dist placed at {DIST}")


def copy_third_party(copy_entries: list[Path]) -> None:
    log(f"copying {len(copy_entries)} site-packages entries onto the dist root "
        "(torch + CUDA stack: expect 15+ GB, this is slow)...")
    ignore = shutil.ignore_patterns("__pycache__")
    for entry in copy_entries:
        target = DIST / entry.name
        if target.exists():
            fail(f"name collision with Nuitka output: {entry.name}")
        if entry.is_dir():
            shutil.copytree(entry, target, ignore=ignore)
        else:
            shutil.copy2(entry, target)


def copy_weights_and_assets(bundle_rife: bool) -> None:
    weights_out = DIST / "model_weights"
    weights_out.mkdir(exist_ok=True)
    for entry in (ROOT / "model_weights").iterdir():
        # TensorRT engines are GPU/TRT-version specific local artifacts; the
        # app regenerates them next to the binary on first run.
        if entry.suffix == ".engine" or entry.name.endswith("_sub_engines"):
            continue
        if entry.name == "PUT_WEIGHTS_HERE":
            continue
        # RIFE weights carry a non-commercial license; keep them out of the
        # bundle unless explicitly requested for a personal-use build.
        if entry.name.lower().startswith("rife") and not bundle_rife:
            log(f"skipping model_weights/{entry.name} (non-commercial RIFE "
                "weights; pass --bundle-rife to include)")
            continue
        if entry.is_dir():
            shutil.copytree(entry, weights_out / entry.name, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, weights_out / entry.name)
    assets_out = DIST / "assets"
    assets_out.mkdir(exist_ok=True)
    for clip in ["test_clip1_1080p.mp4", "test_clip1_2160p.mp4"]:
        src = ROOT / "assets" / clip
        if src.is_file():
            shutil.copy2(src, assets_out / clip)
        else:
            log(f"WARNING: assets/{clip} missing, skipped")


# CUDA NPP / nvJPEG runtime DLLs used by python_vali's NV12<->RGB conversion.
# Not shipped by any pip wheel, and windows_dll_paths.py deliberately strips
# the CUDA Toolkit from PATH in the frozen app, so they must sit at the dist
# root (same list the old PyInstaller spec bundled).
CUDA_RUNTIME_DLLS = [
    "nppc64_13.dll",
    "nppial64_13.dll",
    "nppicc64_13.dll",
    "nppidei64_13.dll",
    "nppig64_13.dll",
    "nvjpeg64_13.dll",
]


def bundle_cuda_npp_dlls() -> None:
    cuda_path = os.environ.get("CUDA_PATH")
    if not cuda_path:
        log("WARNING: CUDA_PATH not set; NPP/nvJPEG DLLs not bundled — "
            "video decode will fail on machines without a CUDA Toolkit on PATH")
        return
    cuda_bin = Path(cuda_path) / "bin" / "x64"
    for dll in CUDA_RUNTIME_DLLS:
        src = cuda_bin / dll
        if src.is_file():
            shutil.copy2(src, DIST / dll)
        else:
            log(f"WARNING: {src} not found, not bundled")


def bundle_external_tools() -> None:
    # ffmpeg/ffprobe -> tools/ ; os_utils falls back to PATH when absent, so a
    # miss is only a warning.
    tools_out = DIST / "tools"
    copied_dll_dirs: set[Path] = set()
    for exe_name in ["ffmpeg", "ffprobe"]:
        found = shutil.which(exe_name)
        if not found:
            log(f"WARNING: {exe_name} not on PATH, not bundled (runtime PATH fallback applies)")
            continue
        src = Path(found)
        tools_out.mkdir(exist_ok=True)
        shutil.copy2(src, tools_out / src.name)
        if src.parent not in copied_dll_dirs:  # shared-build DLLs, if any
            for dll in src.parent.glob("*.dll"):
                shutil.copy2(dll, tools_out / dll.name)
            copied_dll_dirs.add(src.parent)
    mkvmerge = shutil.which("mkvmerge")
    if mkvmerge:
        src = Path(mkvmerge)
        mkv_out = DIST / "mkvtoolnix"
        mkv_out.mkdir(exist_ok=True)
        shutil.copy2(src, mkv_out / src.name)
        for dll in src.parent.glob("*.dll"):
            shutil.copy2(dll, mkv_out / dll.name)
    else:
        log("WARNING: mkvmerge not on PATH, not bundled (runtime PATH fallback applies)")


def smoke_test() -> None:
    from importlib.metadata import version

    expected = version("jasna")
    exe = DIST / "jasna.exe"
    out = subprocess.run(
        [str(exe), "--version"], capture_output=True, text=True, timeout=180, cwd=DIST
    )
    got = (out.stdout + out.stderr).strip()
    if out.returncode != 0 or expected not in got:
        fail(f"--version smoke test failed (rc={out.returncode}): {got!r}, expected {expected!r}")
    log(f"--version OK: {got}")
    out = subprocess.run(
        [str(exe), "--help"], capture_output=True, text=True, timeout=180, cwd=DIST
    )
    if out.returncode != 0 or "--input" not in out.stdout:
        fail(f"--help smoke test failed (rc={out.returncode})")
    log("--help OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-nuitka",
        action="store_true",
        help="reuse existing build/nuitka/__main__.dist, redo only the bundle steps",
    )
    parser.add_argument(
        "--bundle-rife",
        action="store_true",
        help="include RIFE weights (model_weights/rife*) in the bundle; excluded "
        "by default because they carry a non-commercial license",
    )
    args = parser.parse_args()

    preflight()
    copy_entries, nofollow = enumerate_site_packages()
    if not args.skip_nuitka:
        if BUILD_DIR.exists():
            shutil.rmtree(BUILD_DIR)
        run_nuitka(nofollow)
        check_report()
    place_dist()
    copy_third_party(copy_entries)
    copy_weights_and_assets(bundle_rife=args.bundle_rife)
    bundle_cuda_npp_dlls()
    bundle_external_tools()
    smoke_test()
    log(f"done: {DIST}")


if __name__ == "__main__":
    main()
