from __future__ import annotations

import torch

from jasna.restorer.checkpoint_info import (
    checkpoint_has_ema_weights,
    discover_restoration_checkpoints,
    resolve_restoration_checkpoint,
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


def test_resolve_restoration_checkpoint_matches_stem(tmp_path) -> None:
    target = tmp_path / "finetune_v2.pth"
    target.write_bytes(b"x")
    (tmp_path / "other_model.pth").write_bytes(b"x")

    assert resolve_restoration_checkpoint("finetune_v2", tmp_path) == target


def test_resolve_restoration_checkpoint_falls_back_to_default(tmp_path, monkeypatch) -> None:
    import jasna.engine_paths as engine_paths

    default = tmp_path / "default_model.pth"
    monkeypatch.setattr(
        engine_paths, "default_restoration_model_path", lambda: default
    )
    (tmp_path / "existing.pth").write_bytes(b"x")

    assert resolve_restoration_checkpoint("", tmp_path) == default
    assert resolve_restoration_checkpoint("  ", tmp_path) == default
    assert resolve_restoration_checkpoint("no_such_stem", tmp_path) == default


def test_discover_seedvr2_env_repo_and_lora(tmp_path, monkeypatch) -> None:
    from jasna.restorer.checkpoint_info import discover_seedvr2

    repo = tmp_path / "repo"
    (repo / "src" / "core").mkdir(parents=True)
    (repo / "src" / "core" / "generation_utils.py").touch()
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "lada_seedvr2_lora_v2.pt").touch()
    monkeypatch.setenv("JASNA_SEEDVR2_REPO", str(repo))
    monkeypatch.setattr("jasna.engine_paths.model_weights_dir", lambda: weights)

    result = discover_seedvr2()
    assert result is not None
    found_repo, found_lora = result
    assert found_repo == repo
    assert found_lora == weights / "lada_seedvr2_lora_v2.pt"


def test_discover_seedvr2_missing_marker_returns_none(tmp_path, monkeypatch) -> None:
    from jasna.restorer.checkpoint_info import discover_seedvr2

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("JASNA_SEEDVR2_REPO", str(repo))

    assert discover_seedvr2() is None


def test_discover_seedvr2_missing_lora_returns_none(tmp_path, monkeypatch) -> None:
    from jasna.restorer.checkpoint_info import discover_seedvr2

    repo = tmp_path / "repo"
    (repo / "src" / "core").mkdir(parents=True)
    (repo / "src" / "core" / "generation_utils.py").touch()
    weights = tmp_path / "weights"
    weights.mkdir()
    monkeypatch.setenv("JASNA_SEEDVR2_REPO", str(repo))
    monkeypatch.setattr("jasna.engine_paths.model_weights_dir", lambda: weights)

    assert discover_seedvr2() is None


def test_discover_seedvr2_prefers_v5_over_v3(tmp_path, monkeypatch) -> None:
    from jasna.restorer.checkpoint_info import discover_seedvr2

    repo = tmp_path / "repo"
    (repo / "src" / "core").mkdir(parents=True)
    (repo / "src" / "core" / "generation_utils.py").touch()
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "lada_seedvr2_lora_v3.pt").touch()
    (weights / "lada_seedvr2_lora_v5.pt").touch()
    monkeypatch.setenv("JASNA_SEEDVR2_REPO", str(repo))
    monkeypatch.setattr("jasna.engine_paths.model_weights_dir", lambda: weights)

    result = discover_seedvr2()
    assert result is not None
    assert result[1] == weights / "lada_seedvr2_lora_v5.pt"


def test_discover_seedvr2_prefers_v3_over_v2(tmp_path, monkeypatch) -> None:
    from jasna.restorer.checkpoint_info import discover_seedvr2

    repo = tmp_path / "repo"
    (repo / "src" / "core").mkdir(parents=True)
    (repo / "src" / "core" / "generation_utils.py").touch()
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "lada_seedvr2_lora_v2.pt").touch()
    (weights / "lada_seedvr2_lora_v3.pt").touch()
    monkeypatch.setenv("JASNA_SEEDVR2_REPO", str(repo))
    monkeypatch.setattr("jasna.engine_paths.model_weights_dir", lambda: weights)

    result = discover_seedvr2()
    assert result is not None
    assert result[1] == weights / "lada_seedvr2_lora_v3.pt"
