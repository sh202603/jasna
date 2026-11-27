"""+modi: the settings tag names what a converted LTX transformer was made from."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from factories import session_config
from jasna.ltx.model_files import LTX_MODEL_BRANCH, LtxModelFiles
from jasna.settings_tag import job_settings
from jasna.vr180 import VrModeResolution

FLAT = VrModeResolution("auto", "off", "", 16 / 9, "raw")
WEIGHTS = "ltx-restore-alpha1-distill2-int8.safetensors"
SOURCE = "kukkiri kiyo-ltx-v1-transformer.safetensors (plain int8 -> ConvRot int8)"


def _ltx(source: str | None) -> dict[str, object]:
    return job_settings(
        session_config(restoration_model_name="ltx"),
        models=frozenset({"ltx"}),
        segments=None,
        vr=FLAT,
        ltx_weights=WEIGHTS,
        ltx_large_canvas=False,
        ltx_weights_source=source,
    )["ltx"]


def test_a_converted_transformer_is_named_by_its_source_not_upstreams_release() -> None:
    ltx = _ltx(SOURCE)

    assert ltx["weights_source"] == SOURCE
    assert "weights_release" not in ltx


def test_a_published_transformer_keeps_upstreams_tag() -> None:
    ltx = _ltx(None)

    assert ltx["weights_release"] == LTX_MODEL_BRANCH
    assert "weights_source" not in ltx


def _files(path: Path, metadata: dict[str, str] | None) -> LtxModelFiles:
    save_file({"x": torch.zeros(1)}, str(path), metadata=metadata)
    return LtxModelFiles(path, path, path)


def test_the_source_is_read_from_the_transformers_metadata(tmp_path: Path) -> None:
    assert _files(tmp_path / WEIGHTS, {"converted_from": SOURCE}).transformer_source == SOURCE


def test_a_file_without_the_key_or_without_metadata_has_no_source(tmp_path: Path) -> None:
    assert _files(tmp_path / "a.safetensors", {"sigmas": "[1.0, 0.0]"}).transformer_source is None
    assert _files(tmp_path / "b.safetensors", None).transformer_source is None


def test_an_encrypted_file_and_the_trial_have_no_source(tmp_path: Path) -> None:
    encrypted = tmp_path / (WEIGHTS + ".enc")
    encrypted.write_bytes(b"not safetensors")

    assert LtxModelFiles(encrypted, encrypted, encrypted).transformer_source is None
    assert LtxModelFiles.placeholder("distilled2", fast=False).transformer_source is None
