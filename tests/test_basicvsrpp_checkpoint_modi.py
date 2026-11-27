"""+modi: BasicVSR++ checkpoints saved by the trainer (weights plus non-tensor objects) still load."""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import torch

from jasna.models.basicvsrpp import inference


class _TrainerLogBuffer:
    """Stands in for mmengine's HistoryBuffer: a class the weights-only loader does not allow."""


def test_tensor_only_checkpoint_loads_without_a_warning(tmp_path: Path, caplog) -> None:
    path = tmp_path / "clean.pth"
    torch.save({"state_dict": OrderedDict(a=torch.ones(2))}, path)

    with caplog.at_level(logging.WARNING, logger=inference.logger.name):
        checkpoint = inference._load_checkpoint(str(path))

    assert torch.equal(checkpoint["state_dict"]["a"], torch.ones(2))
    assert not caplog.records


def test_trainer_checkpoint_loads_with_a_warning(tmp_path: Path, caplog) -> None:
    path = tmp_path / "trainer.pth"
    torch.save({"state_dict": OrderedDict(a=torch.ones(2)), "message_hub": _TrainerLogBuffer()}, path)

    with caplog.at_level(logging.WARNING, logger=inference.logger.name):
        checkpoint = inference._load_checkpoint(str(path))

    assert torch.equal(checkpoint["state_dict"]["a"], torch.ones(2))
    assert "without the weights-only check" in caplog.text
