"""``--ltx-crop-vsr swiftvr-distill`` end to end on the probe clip: the output has the LTX
output's frame count and timing, and every window of the release's plan is restored."""

from __future__ import annotations

import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from jasna.ltx.refine import check_alignment, read_stream_times
from jasna.model_weights_resolver import resolve_model_weights_file

TESTDATA = Path.home() / "testdata" / "ltx_sharpen"
LTX_OUTPUT = TESTDATA / "SHI-017.restored-ltx.mp4"
SOURCE = TESTDATA / "SHI-017.mp4"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not LTX_OUTPUT.is_file()
    or not SOURCE.is_file()
    or not resolve_model_weights_file("swiftvr-distill.pt").is_file()
    or not resolve_model_weights_file("rfdetr-v6.onnx").is_file(),
    reason="needs CUDA, the ltx_sharpen clips, swiftvr-distill.pt and rfdetr-v6",
)


def test_distill_refine_keeps_the_ltx_output_frames_and_timing(tmp_path):
    output = tmp_path / "refined.mkv"
    # A test that imported jasna.__main__ in this process leaves JASNA_MAIN_PID behind, and
    # the CLI's fork guard then exits the child silently with 0.
    environment = {k: v for k, v in os.environ.items() if k != "JASNA_MAIN_PID"}
    result = subprocess.run(
        [
            sys.executable, "-m", "jasna",
            "--input", str(LTX_OUTPUT), "--output", str(output),
            "--ltx-crop-vsr", "swiftvr-distill", "--ltx-crop-vsr-source", str(SOURCE),
            "--cq", "20", "--log-level", "info", "--no-progress",
        ],
        capture_output=True, text=True, timeout=900, env=environment,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    log = result.stderr + result.stdout
    assert "LTX plan: 890 frames, 1 scene cuts, 2 tracks, 11 windows" in log
    assert "11 windows (11 restored, 0 left as is)" in log
    ltx, refined = read_stream_times(LTX_OUTPUT), read_stream_times(output)
    assert len(refined.pts) == len(ltx.pts) == 890
    check_alignment(refined, ltx, fps=Fraction(30000, 1001), ltx_name="refined", source_name="ltx")
