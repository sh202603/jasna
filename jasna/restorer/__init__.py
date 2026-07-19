"""Restoration package.

The restorer classes pull in torch and the full restoration stack, a multi-second
import. They are exposed lazily so that reaching a lightweight helper in this package
(e.g. ``sd15_download.bundle_present``, a filesystem check the GUI runs at startup)
does not pay that cost.
"""
import importlib
import sys
from pathlib import Path

from jasna._frozen import is_frozen

_LAZY_EXPORTS = {
    "BasicvsrppMosaicRestorer": "jasna.restorer.basicvsrpp_mosaic_restorer",
    "DenoiseStep": "jasna.restorer.denoise",
    "DenoiseStrength": "jasna.restorer.denoise",
    "RestorationPipeline": "jasna.restorer.restoration_pipeline",
}

__all__ = list(_LAZY_EXPORTS)


def bundled_script_path(name: str) -> Path:
    """Path of a restorer script handed as a real .py file to an external interpreter.

    The FlashVSR worker/driver scripts are executed by the FlashVSR venv's own
    Python, so they must exist on disk. The compiled (Nuitka) binary carries no
    source files; the frozen build copies them to <dist>/jasna/restorer/ and they
    are resolved there instead of next to this package's __file__.
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent / "jasna" / "restorer" / name
    return Path(__file__).resolve().with_name(name)


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value
