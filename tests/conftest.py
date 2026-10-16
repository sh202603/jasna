"""Load NVIDIA TensorRT DLLs first when the active environment provides them."""
import gc
import sys
import time
import sys
from importlib.util import find_spec

import pytest

# Standard TensorRT or the TensorRT-RTX flavor — either provides the raw TRT
# API (via jasna.trt._backend) and torch_tensorrt, so the TRT test files apply.
_HAS_TENSORRT = find_spec("tensorrt") is not None or find_spec("tensorrt_rtx") is not None

if _HAS_TENSORRT and find_spec("tensorrt_libs") is not None:
    import tensorrt_libs  # noqa: F401 — locks in tensorrt_libs nvinfer_10.dll before nvvfx

    if sys.platform != "win32":
        # On Linux importing the package does not dlopen the .so's, so the line
        # above pins nothing. Importing the restorer module runs its
        # _preload_tensorrt_runtime(), claiming the global symbol scope for the pip
        # libnvinfer before anything imports nvvfx — test_e2e's collection-time
        # availability probe loads nvvfx's bundled (older) libnvinfer.so.10, which
        # would otherwise win and break TestTensorrtLoadOrder / engine loading.
        import jasna.restorer.rtx_superres_secondary_restorer  # noqa: F401

collect_ignore = [] if _HAS_TENSORRT else [
    "test_basicvsrpp_sub_engines.py",
    "test_rtx_superres_restorer.py",
    "test_torch_tensorrt_export.py",
    "test_trt_runner.py",
    "test_trt_utils.py",
    "test_unet4x_secondary_restorer.py",
]


def pytest_configure(config):
    """Turn off CustomTkinter's per-monitor DPI detection on Windows, as ``run_gui()`` does.

    Otherwise every CTk window multiplies its sizes by the test machine's display scaling,
    so GUI tests would pass or fail depending on the monitor. With it off, the only scaling
    is the factor a test sets (see ``hidpi``), exactly as on Linux.
    """
    if sys.platform == "win32":
        import customtkinter as ctk

        ctk.deactivate_automatic_dpi_awareness()


@pytest.fixture
def show_window():
    """Process Tk events until a window is viewable, so its widgets have real sizes.

    On Windows CustomTkinter withdraws every new window to recolor its title bar and maps
    it again from an ``after`` callback; until then Tk reports 1x1 geometry, and a single
    ``update()`` may run before that callback.
    """

    def show(window) -> None:
        deadline = time.monotonic() + 2.0
        window.update()
        while not window.winfo_viewable() and time.monotonic() < deadline:
            window.update()
        assert window.winfo_viewable()

    return show


@pytest.fixture
def tk_garbage_collected():
    """Finalize Tk objects left by earlier GUI tests on the main thread.

    Code that runs ``gc.collect()`` on a worker thread (the GUI processor does after every
    job) would otherwise finalize them there, and each Tk call off the main thread waits a
    second for a main loop that tests never run, outlasting the tests' join timeouts.
    """
    gc.collect()


@pytest.fixture(autouse=True)
def _isolate_model_weights_dir(monkeypatch):
    """Tests resolve model_weights CWD-relative (usually after chdir to tmp_path).

    A machine-level JASNA_MODEL_WEIGHTS_DIR overrides that resolution, so tests
    that write fake engine/weight files (e.g. engine preflight's 1-byte
    placeholders) would clobber the user's real engines and checkpoints.
    """
    monkeypatch.delenv("JASNA_MODEL_WEIGHTS_DIR", raising=False)


@pytest.fixture
def hidpi(request):
    """Reproduce Windows display scaling on a platform whose DPI factor is always 1.

    CustomTkinter multiplies its detected per-monitor DPI factor by these process-global
    factors, so setting them makes geometry()/minsize() and CTk widget sizes behave exactly
    as on a scaled Windows monitor while winfo_* keeps reporting physical pixels - the
    asymmetry behind issue #241. They are class attributes on ScalingTracker and leak into
    every later test unless reset.
    """
    import customtkinter as ctk

    factor = request.param
    ctk.set_widget_scaling(factor)
    ctk.set_window_scaling(factor)
    try:
        yield factor
    finally:
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)


@pytest.fixture
def no_gpu_cleanup(monkeypatch):
    """Skip the per-job torch cleanup: it initializes CUDA, which can outlast thread joins in a busy run."""
    monkeypatch.setattr("jasna.gui.processor._cleanup_torch", lambda torch_mod: None)
