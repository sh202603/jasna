"""Restoration package (modi: only the helper for scripts run by an external interpreter lives here)."""
import sys
from pathlib import Path

from jasna._frozen import is_frozen


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
