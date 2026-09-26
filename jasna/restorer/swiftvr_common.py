"""Import-light helpers shared by the SwiftVR secondary restoration paths.

Argument registration (``--swiftvr-*``) and path resolution for
``--secondary-restoration swiftvr-inline``. Deliberately free of torch so
``jasna.main.build_parser`` and the startup checks stay fast; the restorer
itself lives in ``swiftvr_inline_secondary_restorer.py``.

SwiftVR (one-step streaming diffusion VSR, Wan2.2-TI2V-5B backbone) is not
bundled: the user supplies a checkout of the fork ``sh202603/SwiftVR`` (branch
``modi``, which adds ``SwiftVRPipeline.restore_clip()``), its ~20 GB checkpoint
and a ``uv sync`` venv, and points jasna at them with ``--swiftvr-repo``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import argparse

SWIFTVR_FORK_URL = "https://github.com/sh202603/SwiftVR"

# Files ``SwiftVRPipeline.from_pretrained`` reads from the checkpoint dir.
_CHECKPOINT_FILES = (
    "reae.safetensors",
    "prompt_embedding.safetensors",
    os.path.join("transformer", "config.json"),
)
_PIPELINE_REL = Path("swiftvr") / "pipeline.py"


def default_swiftvr_python(repo: Path) -> Path:
    """The interpreter ``uv sync`` creates inside the checkout."""
    if os.name == "nt":
        return Path(repo) / ".venv" / "Scripts" / "python.exe"
    return Path(repo) / ".venv" / "bin" / "python"


def default_swiftvr_model_dir(repo: Path) -> Path:
    """Where the SwiftVR README's ``hf download --local-dir checkpoints/`` lands."""
    return Path(repo) / "checkpoints"


def check_restore_clip_api(repo: Path) -> None:
    """Fail fast if the checkout predates ``SwiftVRPipeline.restore_clip()``.

    The worker hands whole in-memory clips to that method; an upstream or older
    checkout only has the file-based ``restore_video`` and the frame-count-
    changing ``StreamSession``, neither of which satisfies the secondary
    restorer's exact-frame-count contract.
    """
    pipeline = Path(repo) / _PIPELINE_REL
    if not pipeline.is_file():
        raise FileNotFoundError(
            f"SwiftVR pipeline not found: {pipeline} (is --swiftvr-repo a SwiftVR checkout?)"
        )
    text = pipeline.read_text(encoding="utf-8", errors="ignore")
    if "def restore_clip" not in text:
        raise RuntimeError(
            f"SwiftVR checkout at {repo} has no SwiftVRPipeline.restore_clip(). jasna needs "
            f"the fork {SWIFTVR_FORK_URL} (branch modi) at a revision that includes it."
        )


def resolve_swiftvr_paths(
    repo: str, python: str, model_dir: str, *, mode: str = "swiftvr-inline"
) -> tuple[Path, Path, Path]:
    """Validate the ``--swiftvr-*`` inputs; return ``(repo, python, model_dir)``.

    Used by the session factory (inline) so the checks and their messages live
    in one place.
    """
    flag = f"--secondary-restoration {mode}"
    if not str(repo or "").strip():
        raise ValueError(f"--swiftvr-repo is required for {flag}")
    repo_path = Path(str(repo)).expanduser()
    if not repo_path.is_dir():
        raise FileNotFoundError(f"--swiftvr-repo not found: {repo_path}")
    check_restore_clip_api(repo_path)

    py_arg = str(python or "").strip()
    py = Path(py_arg).expanduser() if py_arg else default_swiftvr_python(repo_path)
    if not py.exists():
        raise FileNotFoundError(
            f"SwiftVR Python not found: {py}. Create the venv with 'uv sync' in the checkout "
            "or pass --swiftvr-python. On Linux its base Python must ship the dev headers "
            "(python3.X-dev) that Triton needs for --swiftvr-accel."
        )

    md_arg = str(model_dir or "").strip()
    md = Path(md_arg).expanduser() if md_arg else default_swiftvr_model_dir(repo_path)
    if not md.is_dir():
        raise FileNotFoundError(
            f"--swiftvr-model-dir not found: {md} (download the checkpoint with "
            "'uv run hf download H-oliday/SwiftVR --local-dir <dir>')"
        )
    missing = [f for f in _CHECKPOINT_FILES if not (md / f).is_file()]
    if missing:
        raise FileNotFoundError(f"--swiftvr-model-dir {md} is missing {', '.join(missing)}")
    return repo_path, py, md


def add_swiftvr_arguments(group: "argparse._ArgumentGroup") -> None:
    """Register the ``--swiftvr-*`` flags (names mirror the ``--flashvsr-*`` set)."""
    import argparse

    group.add_argument(
        "--swiftvr-repo",
        type=str,
        default="",
        help="Path to the SwiftVR checkout (required for --secondary-restoration swiftvr-inline). "
             f"Use the fork {SWIFTVR_FORK_URL} (branch modi): jasna needs its "
             "SwiftVRPipeline.restore_clip().",
    )
    group.add_argument(
        "--swiftvr-python",
        type=str,
        default="",
        help="Python of the SwiftVR venv (default: <repo>/.venv/bin/python; on Windows "
             "<repo>/.venv/Scripts/python.exe), created with 'uv sync' in the checkout. On "
             "Linux its base Python must ship the dev headers (python3.X-dev) that Triton "
             "needs for --swiftvr-accel.",
    )
    group.add_argument(
        "--swiftvr-model-dir",
        type=str,
        default="",
        help="SwiftVR checkpoint directory holding reae.safetensors, "
             "prompt_embedding.safetensors and transformer/ (default: <repo>/checkpoints).",
    )
    group.add_argument(
        "--swiftvr-scale",
        type=int,
        default=4,
        choices=[2, 4],
        help="Processing scale (default: %(default)s). 4 processes the 256px crops at 1024px "
             "(model-native); 2 processes at 512px, faster with less VRAM (the model is "
             "4x-trained, so 2 is an opt-in). The output video resolution is unchanged either "
             "way: the blend shrink-composites the crops back onto the frame.",
    )
    group.add_argument(
        "--swiftvr-accel",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run the SwiftVR DiT with FP8 linears and torch.compile (default: %(default)s). "
             "Needs an RTX 40 series or newer GPU and a working Triton; the worker checks "
             "both before loading the model and falls back to plain bf16 for whatever is "
             "unavailable (with a warning). bf16 peaks ~12 GiB and does not co-reside with "
             "the primary pipeline on a 16 GB card. The output differs slightly from bf16 "
             "(about 47 dB PSNR).",
    )
