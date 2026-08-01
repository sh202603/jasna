"""Discovery and EMA inspection of BasicVSR++ restoration checkpoints.

Core-side (not GUI-only) so the CLI and evaluation scripts can reuse it.
``checkpoint_has_ema_weights`` exists because ``load_model`` is hardwired to
``is_use_ema=True``: a checkpoint without ``generator_ema.*`` keys loads
silently with random weights, so callers must reject such checkpoints before
building a session.
"""

from __future__ import annotations

from pathlib import Path

_EMA_KEY_PREFIX = "generator_ema."

# rife.pth is the frame-generation model; detection weights are .pt/.onnx so
# they never collide with the *.pth glob.
_NON_RESTORATION_STEMS = frozenset({"rife"})

_ema_cache: dict[tuple[str, int, int], bool] = {}
_EMA_CACHE_MAX = 64


def discover_restoration_checkpoints(weights_dir: Path | None = None) -> list[Path]:
    """All candidate BasicVSR++ checkpoints (``*.pth``) directly under the
    model weights directory, sorted by name."""
    if weights_dir is None:
        from jasna.engine_paths import model_weights_dir

        weights_dir = model_weights_dir()
    if not weights_dir.is_dir():
        return []
    return sorted(
        path
        for path in weights_dir.glob("*.pth")
        if path.is_file() and path.stem.lower() not in _NON_RESTORATION_STEMS
    )


def checkpoint_has_ema_weights(path: Path) -> bool:
    """True when the checkpoint carries ``generator_ema.*`` weights.

    Loads the checkpoint (mmap when the format allows it), so this is slow and
    must only run on a background worker thread, never the Tk thread. Results
    are cached by ``(path, mtime, size)``.
    """
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _ema_cache.get(key)
    if cached is not None:
        return cached

    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (RuntimeError, ValueError):
        # Legacy (non-zipfile) serialization does not support mmap.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    result = any(str(k).startswith(_EMA_KEY_PREFIX) for k in state_dict)

    if len(_ema_cache) >= _EMA_CACHE_MAX:
        _ema_cache.pop(next(iter(_ema_cache)))
    _ema_cache[key] = result
    return result
