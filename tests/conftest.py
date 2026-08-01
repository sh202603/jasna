"""Load NVIDIA TensorRT DLLs first when the active environment provides them."""
import sys
from importlib.util import find_spec

import pytest

_HAS_TENSORRT = find_spec("tensorrt") is not None

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
