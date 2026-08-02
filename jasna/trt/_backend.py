"""TensorRT flavor shim — the sole import point for the TRT python module.

``torch-tensorrt-rtx`` aliases ``tensorrt`` into ``sys.modules`` when
``torch_tensorrt`` is imported, so a bare ``import tensorrt`` in an RTX venv
succeeds or fails depending on import order. Deriving the flavor from
:func:`jasna.engine_paths.trt_flavor` (a ``find_spec`` probe, immune to the
alias) and importing accordingly keeps the runtime module and the
flavor-tagged engine paths agreeing by construction.

A forced ``JASNA_TRT_FLAVOR`` pointing at a missing wheel fails loudly here
with ImportError — intended, that override is emergency-only.
"""
from jasna.engine_paths import trt_flavor

TRT_FLAVOR = trt_flavor()

if TRT_FLAVOR == "rtx":
    import tensorrt_rtx as trt  # noqa: F401
else:
    import tensorrt as trt  # noqa: F401
