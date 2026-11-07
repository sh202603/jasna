"""modi-only video session tests: the player's restoration checkpoint and the SeedVR2 model.

modi 独自。上流の tests/test_video_session.py には追記しない。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from jasna.gui.models import AppSettings
from jasna.gui.video_session import video_session_config, video_session_key


def _registry_patches():
    return (
        patch("jasna.engine_paths.model_weights_dir"),
        patch("jasna.mosaic.detection_registry.coerce_detection_model_name", side_effect=lambda name: name),
        patch("jasna.mosaic.detection_registry.require_detection_model_weights", return_value=Path("det.engine")),
    )


def test_video_session_key_changes_on_restoration_checkpoint() -> None:
    base = video_session_key(AppSettings())
    assert video_session_key(replace(AppSettings(), restoration_checkpoint="finetune_v2")) != base


def test_video_session_config_resolves_restoration_checkpoint_from_settings() -> None:
    settings = replace(AppSettings(), restoration_checkpoint="finetune_v2")
    a, b, c = _registry_patches()
    with (
        a, b, c,
        patch("jasna.restorer.checkpoint_info.resolve_restoration_checkpoint", return_value=Path("finetune_v2.pth")) as resolver,
    ):
        config = video_session_config(settings, codec="hevc", encoder_settings={})

    resolver.assert_called_once_with("finetune_v2")
    assert config.restoration_model_path == Path("finetune_v2.pth")


def test_video_session_config_explicit_path_bypasses_resolver() -> None:
    settings = replace(AppSettings(), restoration_checkpoint="finetune_v2")
    a, b, c = _registry_patches()
    with a, b, c, patch("jasna.restorer.checkpoint_info.resolve_restoration_checkpoint") as resolver:
        config = video_session_config(
            settings, codec="hevc", encoder_settings={}, restoration_model_path=Path("explicit.pth")
        )

    resolver.assert_not_called()
    assert config.restoration_model_path == Path("explicit.pth")


def test_video_session_config_forwards_seedvr2_model() -> None:
    a, b, c = _registry_patches()
    with a, b, c:
        config = video_session_config(
            AppSettings(),
            codec="hevc",
            encoder_settings={},
            restoration_model_path=Path("lada_seedvr2_lora_v2.pt"),
            restoration_model_name="seedvr2",
            seedvr2_repo="/opt/seedvr2_videoupscaler",
        )

    assert config.restoration_model_name == "seedvr2"
    assert config.seedvr2_repo == "/opt/seedvr2_videoupscaler"
    # The LoRA rides restoration_model_path (same convention as the CLI).
    assert config.restoration_model_path == Path("lada_seedvr2_lora_v2.pt")


def test_video_session_config_defaults_to_basicvsrpp() -> None:
    a, b, c = _registry_patches()
    with a, b, c, patch("jasna.engine_paths.default_restoration_model_path", return_value=Path("gen.pth")):
        config = video_session_config(AppSettings(), codec="hevc", encoder_settings={})

    assert config.restoration_model_name == "basicvsrpp"
    assert config.seedvr2_repo == ""
