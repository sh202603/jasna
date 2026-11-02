"""modi-only session factory tests (the SwiftVR distill secondary).

modi 独自。上流の tests/test_session_factory.py の ``_build_session`` と factories.py の
``session_config``、conftest の ``nvidia_vendor`` fixture に依存する。上流のファイルには追記しない。
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from factories import session_config
from test_session_factory import _build_session

pytestmark = pytest.mark.usefixtures("nvidia_vendor")


def test_session_selects_swiftvr_distill_secondary(tmp_path: Path) -> None:
    model_path = tmp_path / "distill.pt"
    model_path.write_bytes(b"x")
    with patch(
        "jasna.restorer.swiftvr_distill_secondary_restorer.SwiftvrDistillSecondaryRestorer"
    ) as distill_cls:
        _, compiled, _, pipeline_cls, *_ = _build_session(
            session_config(
                secondary_restoration="swiftvr-distill",
                swiftvr_distill_model=str(model_path),
                swiftvr_distill_view_window=15,
                swiftvr_distill_strength=0.75,
                swiftvr_distill_stabilize=2,
            )
        )

    assert pipeline_cls.call_args.kwargs["secondary_restorer"] is distill_cls.return_value
    assert compiled.call_args.args[0].unet4x is False
    kwargs = distill_cls.call_args.kwargs
    assert kwargs["model_path"] == model_path
    assert kwargs["view_window"] == 15
    assert kwargs["strength"] == 0.75
    assert kwargs["stabilize_radius"] == 2


def test_swiftvr_distill_requires_model_path(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "model_weights").mkdir()  # empty: the default file is missing
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="SwiftVR distill checkpoint not found"):
        _build_session(session_config(secondary_restoration="swiftvr-distill"))
    with pytest.raises(FileNotFoundError, match="--swiftvr-distill-model not found"):
        _build_session(
            session_config(
                secondary_restoration="swiftvr-distill",
                swiftvr_distill_model=str(tmp_path / "missing.pt"),
            )
        )
