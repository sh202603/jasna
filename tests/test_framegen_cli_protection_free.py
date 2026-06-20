"""Guard: the frame-gen CLI must never depend on the protection / license code.

The whole point of the standalone CLI is that frame generation is independent of
the proprietary protection submodule. These tests fail loudly if someone wires an
import to jasna.protection / license_store (directly or transitively via
jasna.pipeline) into the CLI's import chain.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import jasna.framegen_cli as cli


def test_source_has_no_protection_import():
    src = Path(cli.__file__).read_text(encoding="utf-8")
    # No import of the protection subpackage (the docstring mentions the name to
    # explain the boundary, so match import statements, not the bare word).
    assert not re.search(r"^\s*(from|import)\s+jasna\.protection", src, re.MULTILINE)
    assert "license_store" not in src


def test_import_pulls_no_protection_or_pipeline():
    # Importing the CLI module must not transitively import protection or the heavy
    # pipeline (restorer/detection/tracking). Run in a clean subprocess so the
    # check is unaffected by modules other tests already imported.
    code = (
        "import jasna.framegen_cli, sys; "
        "bad = [m for m in ('jasna.protection', 'jasna.pipeline') if m in sys.modules]; "
        "assert not bad, bad; "
        "print('ok')"
    )
    # Drop JASNA_MAIN_PID so the multiprocessing fork-guard at the CLI module top
    # (inherited from this pytest process) does not sys.exit(0) the subprocess.
    env = {k: v for k, v in os.environ.items() if k != "JASNA_MAIN_PID"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(cli.__file__).resolve().parents[2]),
        env=env,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "ok" in result.stdout
