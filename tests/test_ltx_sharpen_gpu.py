"""``--secondary-restoration rtx-super-res`` with an LTX model end to end (modi), on the trial
weights so no LTX model files are needed: the compose pass loads the RTX upscaler of the
512 px canvas, the output keeps the input's frame count, and the run's speed profile is
keyed apart from runs without super resolution. The trial transformer is a placeholder of the
real size, so this takes about as long as a real LTX run of the clip."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch

from jasna.ltx.refine import read_stream_times
from jasna.model_weights_resolver import resolve_model_weights_file

CLIP = Path(__file__).resolve().parents[1] / "assets" / "test_clip1_1080p.mp4"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not CLIP.is_file()
    or find_spec("nvvfx") is None
    or not resolve_model_weights_file("rfdetr-v6.onnx").is_file(),
    reason="needs CUDA, the test clip, nvvfx (RTX Video Effects) and rfdetr-v6",
)


def test_an_ltx_trial_run_sharpens_its_windows_with_rtx_super_res(tmp_path):
    output = tmp_path / "out.mkv"
    environment = {k: v for k, v in os.environ.items() if k != "JASNA_MAIN_PID"}
    result = subprocess.run(
        [
            sys.executable, "-m", "jasna",
            "--input", str(CLIP), "--output", str(output),
            "--restoration-model-name", "ltx-8step", "--ltx-trial",
            "--secondary-restoration", "rtx-super-res", "--rtx-strength", "0.75",
            "--log-level", "info",
        ],
        capture_output=True, text=True, timeout=900, env=environment,
    )
    log = result.stdout + result.stderr
    assert result.returncode == 0, log[-4000:]
    assert "RtxSuperresSecondaryRestorer: scale=2x" in log and "deblur=low" in log and "strength=0.75" in log
    assert "(512x512 -> 1024x1024)" in log  # upstream sizes each upscaler for its canvas
    assert "decode free VRAM+rtx" in log  # the speed profile key of a run that sharpens
    assert len(read_stream_times(output).pts) == len(read_stream_times(CLIP).pts)
