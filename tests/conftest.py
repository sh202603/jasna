"""Load NVIDIA TensorRT DLLs first when the active environment provides them."""
import sys
from importlib.util import find_spec

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
