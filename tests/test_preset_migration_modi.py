"""modi-only preset migrations (the player's checkpoint stem, the old frame-gen keys).

modi 独自。上流の tests/test_preset_migration.py には追記しない。
"""
from __future__ import annotations

import pytest

from jasna.gui.models import AppSettings, _migrate_preset_dict


@pytest.mark.parametrize(
    ("saved", "model", "checkpoint"),
    [("", "basicvsrpp", ""), ("finetune_v2", "basicvsrpp", "finetune_v2"), ("ltx", "ltx", ""), ("basicvsrpp", "basicvsrpp", "")],
)
def test_modi_checkpoint_stem_moves_out_of_restoration_model(saved, model, checkpoint) -> None:
    # Before v0.11, modi presets kept the player's checkpoint stem in "restoration_model".
    settings = AppSettings(**_migrate_preset_dict({"restoration_model": saved}))

    assert (settings.restoration_model, settings.restoration_checkpoint) == (model, checkpoint)


@pytest.mark.parametrize(
    ("old", "frame_generation", "backend", "mode", "path", "video_backend"),
    [
        ({"frame_gen": "Off", "frame_gen_backend": "RIFE", "frame_gen_rtx_mode": "High", "video_backend": "TorchCodec"},
         "none", "rife", "high", "", "torchcodec"),
        ({"frame_gen": "4x", "frame_gen_model_path": "/w/rife.pth"}, "4x", "rtx", "medium", "/w/rife.pth", "native"),
        ({"frame_generation": "2x", "frame_gen": "4x"}, "2x", "rtx", "medium", "", "native"),
    ],
)
def test_modi_frame_gen_keys_move_to_frame_generation(old, frame_generation, backend, mode, path, video_backend) -> None:
    # Presets saved before the v0.12.2 base kept frame generation in modi's own keys.
    settings = AppSettings(**_migrate_preset_dict(old))

    assert settings.frame_generation == frame_generation
    assert settings.frame_generation_backend == backend
    assert settings.frame_generation_rtx_mode == mode
    assert settings.frame_generation_model_path == path
    assert settings.video_backend == video_backend
