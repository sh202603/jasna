"""Import-light helpers shared by the SwiftVR secondary restoration paths.

Argument registration (``--swiftvr-*``), path resolution and the offline
Phase 2 command for ``--secondary-restoration swiftvr`` (offline 3-phase, run
by ``flashvsr_offline.run_flashvsr_offline`` with the SwiftVR engine) and
``swiftvr-inline``. Deliberately free of torch so ``jasna.main.build_parser``
and the startup checks stay fast; the inline restorer itself lives in
``swiftvr_inline_secondary_restorer.py``.

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

    Used by the session factory (inline) and the offline orchestrator so the
    checks and their messages live in one place; ``mode`` names the
    ``--secondary-restoration`` value in them.
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


def swiftvr_phase2_command(
    args: "argparse.Namespace", bundle_dir: Path, repo: Path, sv_python: Path, model_dir: Path
) -> tuple[list[str], dict[str, str]]:
    """The offline Phase 2 command and environment.

    Runs ``swiftvr_phase2_driver.py`` under the SwiftVR venv's Python with the
    same contract as the inline restorer's worker command: ``--swiftvr-accel``
    passes both acceleration parts (the driver drops whichever its GPU or
    Triton cannot run), color correction is always on (driver default: wavelet)
    with ``JASNA_SWIFTVR_COLOR_FIX`` as the verification-only override, and the
    venv imports its own package rather than jasna's ``PYTHONPATH``.
    """
    from jasna.restorer import bundled_script_path

    driver = bundled_script_path("swiftvr_phase2_driver.py")
    cmd = [
        str(sv_python), str(driver),
        "--bundle-dir", str(bundle_dir),
        "--repo", str(repo),
        "--model-dir", str(model_dir),
        "--device", str(args.device),
        "--scale", str(int(getattr(args, "swiftvr_scale", 4))),
    ]
    if bool(getattr(args, "swiftvr_accel", True)):
        cmd += ["--fp8-dit", "--torch-compile"]
    color_fix = os.environ.get("JASNA_SWIFTVR_COLOR_FIX")
    if color_fix:
        if color_fix not in ("adain", "wavelet", "none"):
            raise ValueError(
                "[swiftvr] JASNA_SWIFTVR_COLOR_FIX must be adain|wavelet|none, "
                f"got {color_fix!r}"
            )
        cmd += ["--color-fix-method", color_fix]

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.setdefault("TQDM_DISABLE", "1")
    if os.name == "nt":
        # expandable_segments is not supported on Windows; force UTF-8 so a
        # non-ASCII log line cannot raise on the cp932 text layer.
        env["PYTHONUTF8"] = "1"
    else:
        # Same allocator discipline as the inline worker: with the fork's
        # block-wise FP8 load it changes the peak by ~0.2 GB only, but it keeps
        # the bf16 path's 12.4 GiB inside a 16 GB card.
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return cmd, env


def add_swiftvr_arguments(group: "argparse._ArgumentGroup") -> None:
    """Register the ``--swiftvr-*`` flags (names mirror the ``--flashvsr-*`` set)."""
    import argparse

    group.add_argument(
        "--swiftvr-repo",
        type=str,
        default="",
        help="Path to the SwiftVR checkout (required for --secondary-restoration swiftvr and "
             f"swiftvr-inline). Use the fork {SWIFTVR_FORK_URL} (branch modi): jasna needs its "
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
        "--swiftvr-view-window",
        type=int,
        default=15,
        metavar="N",
        help="Smooth the placement of the crops SwiftVR sees over N frames (default: "
             "%(default)s; 0 disables). The primary crops follow the detection box and "
             "shift by a few px every frame; SwiftVR redraws its detail on such shifts, "
             "which is the frame-to-frame flicker of scale 2. The crops are re-viewed "
             "through a moving-average placement before SwiftVR and composited back from "
             "that view; the primary restoration and the blend mask are unchanged.",
    )
    group.add_argument(
        "--swiftvr-accel",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run the SwiftVR DiT with FP8 linears and torch.compile (default: %(default)s). "
             "Needs an RTX 40 series or newer GPU and a working Triton; the worker checks "
             "both before loading the model and falls back to plain bf16 for whatever is "
             "unavailable (with a warning). bf16 peaks ~12 GiB: it does not co-reside with "
             "the primary pipeline on a 16 GB card (swiftvr-inline), but fits alone in the "
             "offline swiftvr mode. The output differs slightly from bf16 (about 47 dB PSNR).",
    )
    group.add_argument(
        "--swiftvr-bundle-dir",
        type=str,
        default="",
        help="swiftvr (offline) only: persist the intermediate bundle here (default: a temp "
             "dir removed on completion). A persisted bundle lets a failed run resume from "
             "the phase that failed.",
    )
    group.add_argument(
        "--swiftvr-keep-bundle",
        action="store_true",
        help="swiftvr (offline) only: keep the bundle dir after completion (implied by "
             "--swiftvr-bundle-dir).",
    )
