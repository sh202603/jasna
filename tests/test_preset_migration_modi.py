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
