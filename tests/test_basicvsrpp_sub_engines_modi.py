"""modi-only BasicVSR++ sub-engine tests (the FP8 upsample skips the TRT upsample engine).

modi 独自。上流の tests/test_basicvsrpp_sub_engines.py には追記しない。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from jasna.engine_paths import get_basicvsrpp_sub_engine_paths as get_sub_engine_paths


def test_load_sub_engines_skip_upsample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jasna.restorer.basicvsrpp_sub_engines as se

    model_path = str(tmp_path / "model.pth")
    paths = get_sub_engine_paths(model_path, fp16=True)
    for p in paths.values():
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text("x", encoding="utf-8")

    loaded: list[str] = []

    def _fake_load(checkpoint_path: str, device: torch.device) -> nn.Module:
        loaded.append(checkpoint_path)
        return nn.Module()

    monkeypatch.setattr(se, "load_torchtrt_export", _fake_load)
    result = se.load_sub_engines(
        model_path, torch.device("cpu"), fp16=True, load_upsample=False,
    )
    assert result is not None
    loop_body_engines, preprocess_engine, upsample_engine = result
    assert upsample_engine is None
    assert len(loaded) == 5  # 4 loop bodies + preprocess, no upsample
    assert not any("upsample" in p for p in loaded)
