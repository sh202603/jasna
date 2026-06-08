"""Ensure the pip TensorRT runtime is pinned before nvvfx can override it."""
import sys

import tensorrt_libs  # noqa: F401 — locks in tensorrt_libs nvinfer_10.dll before nvvfx

if sys.platform != "win32":
    # On Linux importing the package does not dlopen the .so's, so the line
    # above pins nothing. Importing the restorer module runs its
    # _preload_tensorrt_runtime(), claiming the global symbol scope for the pip
    # libnvinfer before anything imports nvvfx — test_e2e's collection-time
    # availability probe loads nvvfx's bundled (older) libnvinfer.so.10, which
    # would otherwise win and break TestTensorrtLoadOrder / engine loading.
    import jasna.restorer.rtx_superres_secondary_restorer  # noqa: F401
