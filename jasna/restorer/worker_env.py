"""Environment for worker processes that run in another project's venv.

The FlashVSR, SwiftVR and SeedVR2 workers import their own torch, CUDA and
cuDNN. jasna's process environment must not leak into them:

- ``PYTHONPATH`` would make the worker import jasna's packages.
- ``LD_LIBRARY_PATH`` can carry library dirs from jasna's own venv: the FP8
  upsample backend puts torch's cuDNN in front (see ``fp8_upsample``) and
  nvvfx prepends its libs dir. A worker with a different torch then loads
  jasna's cuDNN sub-libraries and fails at its first convolution
  (``CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED``).
"""
from __future__ import annotations

import os
import sys


def external_worker_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """A copy of ``environ`` (default ``os.environ``) without ``PYTHONPATH`` and
    without the ``LD_LIBRARY_PATH`` entries that point into this interpreter's
    installation."""
    env = dict(os.environ if environ is None else environ)
    env.pop("PYTHONPATH", None)
    library_path = env.get("LD_LIBRARY_PATH")
    if library_path:
        own_prefix = os.path.realpath(sys.prefix) + os.sep
        kept = [
            entry
            for entry in library_path.split(os.pathsep)
            if entry and not os.path.realpath(entry).startswith(own_prefix)
        ]
        if kept:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(kept)
        else:
            del env["LD_LIBRARY_PATH"]
    return env
