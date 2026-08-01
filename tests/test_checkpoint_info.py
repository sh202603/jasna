from __future__ import annotations

import torch

from jasna.restorer.checkpoint_info import (
    checkpoint_has_ema_weights,
    discover_restoration_checkpoints,
)


def test_discover_lists_pth_files_sorted_and_filters_non_restoration(tmp_path) -> None:
    (tmp_path / "b_model.pth").write_bytes(b"x")
    (tmp_path / "a_model.pth").write_bytes(b"x")
    (tmp_path / "rife.pth").write_bytes(b"x")
    (tmp_path / "detection.pt").write_bytes(b"x")
    (tmp_path / "rfdetr-v6.onnx").write_bytes(b"x")
    (tmp_path / "subdir").mkdir()

    result = discover_restoration_checkpoints(tmp_path)

    assert [p.name for p in result] == ["a_model.pth", "b_model.pth"]


def test_discover_missing_directory_returns_empty(tmp_path) -> None:
    assert discover_restoration_checkpoints(tmp_path / "missing") == []


def test_checkpoint_with_ema_keys_is_accepted(tmp_path) -> None:
    path = tmp_path / "ema.pth"
    torch.save(
        {
            "state_dict": {
                "generator.conv.weight": torch.zeros(1),
                "generator_ema.conv.weight": torch.zeros(1),
            }
        },
        path,
    )

    assert checkpoint_has_ema_weights(path) is True


def test_checkpoint_without_ema_keys_is_rejected(tmp_path) -> None:
    path = tmp_path / "no_ema.pth"
    torch.save({"state_dict": {"generator.conv.weight": torch.zeros(1)}}, path)

    assert checkpoint_has_ema_weights(path) is False


def test_bare_state_dict_without_wrapper_is_scanned_directly(tmp_path) -> None:
    path = tmp_path / "bare.pth"
    torch.save({"generator_ema.conv.weight": torch.zeros(1)}, path)

    assert checkpoint_has_ema_weights(path) is True


def test_result_is_cached_by_path_and_mtime(tmp_path, monkeypatch) -> None:
    path = tmp_path / "cached.pth"
    torch.save({"state_dict": {"generator_ema.x": torch.zeros(1)}}, path)
    assert checkpoint_has_ema_weights(path) is True

    loads = []
    real_load = torch.load
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: loads.append(1) or real_load(*args, **kwargs)
    )

    assert checkpoint_has_ema_weights(path) is True
    assert loads == []
