"""TRT flavor detection and flavor-tagged engine naming (no tensorrt needed)."""
import os
import re
from pathlib import Path

import pytest

import jasna
from jasna import engine_paths
from jasna.engine_paths import (
    engine_system_suffix,
    get_basicvsrpp_sub_engine_paths,
    get_onnx_tensorrt_engine_path,
    get_unet4x_encrypted_engine_path,
    trt_flavor,
)

_OS_TAG = ".win" if os.name == "nt" else ".linux"


@pytest.fixture
def _fresh_flavor(monkeypatch):
    """Reset the process-wide flavor cache; monkeypatch restores it afterwards."""
    monkeypatch.setattr(engine_paths, "_trt_flavor_cache", None)
    monkeypatch.delenv("JASNA_TRT_FLAVOR", raising=False)
    return monkeypatch


def _force(monkeypatch, flavor: str) -> None:
    monkeypatch.setenv("JASNA_TRT_FLAVOR", flavor)
    monkeypatch.setattr(engine_paths, "_trt_flavor_cache", None)


class TestTrtFlavor:
    def test_env_override_rtx(self, _fresh_flavor):
        _force(_fresh_flavor, "rtx")
        assert trt_flavor() == "rtx"

    def test_env_override_standard(self, _fresh_flavor):
        _force(_fresh_flavor, "standard")
        assert trt_flavor() == "standard"

    def test_invalid_override_falls_back_to_probe(self, _fresh_flavor):
        from importlib.util import find_spec

        _fresh_flavor.setenv("JASNA_TRT_FLAVOR", "bogus")
        _fresh_flavor.setattr(engine_paths, "_trt_flavor_cache", None)
        expected = "rtx" if find_spec("tensorrt_rtx") is not None else "standard"
        assert trt_flavor() == expected

    def test_default_matches_installed_dist(self, _fresh_flavor):
        from importlib.util import find_spec

        expected = "rtx" if find_spec("tensorrt_rtx") is not None else "standard"
        assert trt_flavor() == expected

    def test_result_is_cached(self, _fresh_flavor):
        _force(_fresh_flavor, "rtx")
        assert trt_flavor() == "rtx"
        # A later env change must not flip paths mid-process.
        _fresh_flavor.setenv("JASNA_TRT_FLAVOR", "standard")
        assert trt_flavor() == "rtx"


class TestFlavorTaggedNaming:
    def test_standard_names_are_byte_identical_to_historical(self, _fresh_flavor, tmp_path):
        _force(_fresh_flavor, "standard")
        assert engine_system_suffix() == _OS_TAG
        p = get_onnx_tensorrt_engine_path(tmp_path / "model.onnx", batch_size=4, fp16=True)
        assert p.name == f"model.bs4.fp16{_OS_TAG}.engine"

    def test_rtx_tag_in_system_suffix(self, _fresh_flavor):
        _force(_fresh_flavor, "rtx")
        assert engine_system_suffix() == f".rtx{_OS_TAG}"

    def test_rtx_tag_onnx_engine_paths(self, _fresh_flavor, tmp_path):
        _force(_fresh_flavor, "rtx")
        fixed = get_onnx_tensorrt_engine_path(tmp_path / "model.onnx", batch_size=4, fp16=True)
        assert fixed.name == f"model.bs4.fp16.rtx{_OS_TAG}.engine"
        dyn = get_onnx_tensorrt_engine_path(
            tmp_path / "model.onnx", batch_size=8, fp16=True, dynamic_batch=True
        )
        assert dyn.name == f"model.bs1-8.fp16.rtx{_OS_TAG}.engine"

    def test_rtx_tag_basicvsrpp_sub_engines(self, _fresh_flavor, tmp_path):
        _force(_fresh_flavor, "rtx")
        paths = get_basicvsrpp_sub_engine_paths(str(tmp_path / "weights.pth"), fp16=True)
        assert len(paths) == 6
        for p in paths.values():
            assert p.endswith(f".rtx{_OS_TAG}.engine"), p

    def test_rtx_tag_unet4x_encrypted(self, _fresh_flavor):
        _force(_fresh_flavor, "rtx")
        enc = get_unet4x_encrypted_engine_path(fp16=True)
        assert enc.name == f"unet-4x.fp16.rtx{_OS_TAG}.engine.enc"


def test_no_bare_tensorrt_imports_outside_backend():
    """`import tensorrt` outside the shim is an import-order landmine: in an RTX
    venv torch-tensorrt-rtx aliases `tensorrt` into sys.modules, so a bare import
    works or fails depending on what was imported first. Everything must go
    through jasna.trt._backend (or the find_spec-guarded _suppress_noise)."""
    allowed = {"trt/_backend.py", "_suppress_noise.py"}
    pattern = re.compile(r"^\s*(import tensorrt(\s+as\s+\w+)?\s*$|from tensorrt[.\s])", re.M)
    pkg_root = Path(jasna.__file__).parent
    offenders = []
    for py in pkg_root.rglob("*.py"):
        rel = py.relative_to(pkg_root).as_posix()
        if rel in allowed:
            continue
        if pattern.search(py.read_text(encoding="utf-8", errors="replace")):
            offenders.append(rel)
    assert offenders == []
