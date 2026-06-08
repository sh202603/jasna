"""Locate the ``model_weights/`` directory.

Searched in order:
  1. ``$JASNA_MODEL_WEIGHTS_DIR`` (explicit user override)
  2. Directory next to ``sys.executable`` (PyInstaller/Nuitka dist build)
  3. Current working directory (dev workflow, backward compatible)
  4. Directory next to the ``jasna`` package (editable install / PATH-launched dev)

The first existing directory wins. If none exist, the CWD candidate is
returned so callers fail with a clear "file not found" pointing at the
relative path the user most likely expects.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from jasna._frozen import is_frozen

_LOGGER = logging.getLogger(__name__)
_LAST_LOGGED: tuple[Path, str] | None = None


def _candidates() -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    env = os.environ.get("JASNA_MODEL_WEIGHTS_DIR")
    if env:
        out.append(("env JASNA_MODEL_WEIGHTS_DIR", Path(env).expanduser()))
    if is_frozen():
        out.append(("next to executable", Path(sys.executable).resolve().parent / "model_weights"))
    out.append(("current working directory", Path("model_weights")))
    out.append(("package parent", Path(__file__).resolve().parent.parent / "model_weights"))
    return out


def resolve_model_weights_dir() -> Path:
    global _LAST_LOGGED
    chosen_label = "current working directory (fallback, missing)"
    chosen: Path = Path("model_weights")
    for label, p in _candidates():
        if p.is_dir():
            chosen_label = label
            chosen = p
            break
    key = (chosen, chosen_label)
    if _LAST_LOGGED != key:
        _LOGGER.info("Resolved model_weights dir: %s (from: %s)", chosen, chosen_label)
        _LAST_LOGGED = key
    return chosen


def resolve_model_weights_file(filename: str) -> Path:
    return resolve_model_weights_dir() / filename
