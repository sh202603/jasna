"""+modi: import-light helpers for RelayVSR (``jasna-postprocess --upscale relayvsr``).

Path resolution and the checkpoint pairs. Free of torch, so the argument checks of
``jasna.postprocess_cli`` run before anything heavy is imported.

RelayVSR (streaming video super-resolution: sparse Wan keyframes and a small conditional
decoder) is not bundled: the user supplies a checkout of the fork ``sh202603/RelayVSR`` (branch
``windows-support``, which adds ``build_stream()`` and the 2x checkpoints), its ``uv sync`` venv
and the checkpoints, and points jasna at them with ``--relayvsr-repo``.
"""
from __future__ import annotations

import os
from pathlib import Path

RELAYVSR_FORK_URL = "https://github.com/sh202603/RelayVSR"

_SESSION_REL = Path("refvsr_inference") / "runtime" / "session.py"

# The factor belongs to the checkpoints: each one has its own pair, and a mixed pair does not load.
_CHECKPOINTS = {
    4: ("wan.safetensors", "flash.safetensors", "kopper/RelayVSR"),
    2: ("wan_x2.safetensors", "flash_x2.safetensors", "lightx2v/RelayVSR_X2"),
}


def default_relayvsr_python(repo: Path) -> Path:
    """The interpreter ``uv sync`` creates inside the checkout."""
    if os.name == "nt":
        return Path(repo) / ".venv" / "Scripts" / "python.exe"
    return Path(repo) / ".venv" / "bin" / "python"


def default_relayvsr_weights_dir(repo: Path) -> Path:
    """Where the RelayVSR README's ``hf download --local-dir weights`` lands."""
    return Path(repo) / "weights"


def check_build_stream_api(repo: Path) -> None:
    """Fail fast if the checkout has no ``build_stream()``: the worker embeds RelayVSR through
    it, and upstream only has the command line."""
    session = Path(repo) / _SESSION_REL
    if not session.is_file() or "def build_stream" not in session.read_text(encoding="utf-8", errors="ignore"):
        raise RuntimeError(
            f"RelayVSR checkout at {repo} has no refvsr_inference.runtime.session.build_stream(). jasna "
            f"needs the fork {RELAYVSR_FORK_URL} (branch windows-support) at a revision that includes it."
        )


def relayvsr_checkpoints(weights_dir: Path, scale: int) -> tuple[Path, Path]:
    """``(wan, flash)`` of the ``scale``x pair in ``weights_dir``."""
    wan_name, flash_name, source = _CHECKPOINTS[int(scale)]
    wan, flash = Path(weights_dir) / wan_name, Path(weights_dir) / flash_name
    missing = [path.name for path in (wan, flash) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"RelayVSR {scale}x checkpoint not found in {weights_dir}: {', '.join(missing)} (download the "
            f"pair with 'uvx hf download {source} {wan_name} {flash_name} --local-dir <dir>')"
        )
    return wan, flash


def resolve_relayvsr_paths(repo: str, python: str, weights_dir: str) -> tuple[Path, Path, Path]:
    """Validate the ``--relayvsr-*`` inputs; return ``(repo, python, weights_dir)``. Which pair
    of checkpoints a video needs is only known from its size, so the pairs are checked per video."""
    if not str(repo or "").strip():
        raise ValueError("--relayvsr-repo is required for --upscale relayvsr")
    repo_path = Path(str(repo)).expanduser()
    if not repo_path.is_dir():
        raise FileNotFoundError(f"--relayvsr-repo not found: {repo_path}")
    check_build_stream_api(repo_path)

    py_arg = str(python or "").strip()
    py = Path(py_arg).expanduser() if py_arg else default_relayvsr_python(repo_path)
    if not py.exists():
        raise FileNotFoundError(
            f"RelayVSR Python not found: {py}. Create the venv with 'uv sync --locked --extra cu128' in "
            "the checkout or pass --relayvsr-python."
        )

    weights_arg = str(weights_dir or "").strip()
    weights = Path(weights_arg).expanduser() if weights_arg else default_relayvsr_weights_dir(repo_path)
    if not weights.is_dir():
        raise FileNotFoundError(f"--relayvsr-weights-dir not found: {weights}")
    return repo_path, py, weights
