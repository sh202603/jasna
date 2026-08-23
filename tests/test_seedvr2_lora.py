"""Tests for the SeedVR2+LoRA primary restorer (no GPU, no SeedVR2 checkout).

The real worker is replaced by a tiny stub speaking the same JSON-header +
raw-uint8-BGR protocol, so the parent's wire handling (handshake, BGR flip,
quantization, respawn+retry, contract errors) runs on a CPU box. The real
worker file itself is a verbatim copy of the lada-ex one and is only checked
for staying import-light at the top level.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

import jasna.restorer as restorer_pkg
from jasna.restorer.seedvr2_lora_restorer import (
    Seedvr2ContractError,
    Seedvr2LoraRestorer,
    default_seedvr2_python,
)

# Stub worker: ready handshake, then per clip either an inverted echo (ok), a
# constant-tagged BGR pattern (tag), a per-clip error (error), a first-process
# failure with later-process success (flaky, via a state file), or a frame
# count lie (shortcount).
_STUB_WORKER = textwrap.dedent(
    """
    import os, sys, json
    import numpy as np

    MODE = os.environ.get("STUB_MODE", "ok")
    STATE = os.environ.get("STUB_STATE_FILE", "")

    def rh(f):
        line = bytearray()
        while True:
            b = f.read(1)
            if not b:
                return None
            if b == b"\\n":
                break
            line += b
        return json.loads(line.decode())

    def rx(f, n):
        buf = bytearray()
        while len(buf) < n:
            c = f.read(n - len(buf))
            if not c:
                raise EOFError()
            buf += c
        return bytes(buf)

    out = sys.stdout.buffer
    stdin = sys.stdin.buffer
    out.write((json.dumps({"status": "ready"}) + "\\n").encode()); out.flush()
    while True:
        h = rh(stdin)
        if h is None:
            break
        n, ih, iw = h["n"], h["h"], h["w"]
        crops = np.frombuffer(rx(stdin, n * ih * iw * 3), np.uint8).reshape(n, ih, iw, 3)
        if MODE == "flaky" and STATE and not os.path.exists(STATE):
            open(STATE, "w").write("died")
            sys.exit(1)
        if MODE == "error":
            out.write((json.dumps({"seq": h.get("seq", 0), "error": "boom"}) + "\\n").encode()); out.flush()
            continue
        rn = n - 1 if MODE == "shortcount" else n
        if MODE == "tag":
            arr = np.zeros((rn, ih, iw, 3), np.uint8)
            arr[..., 0] = 200  # channel 0 on the wire = B
        else:
            arr = (255 - crops[:rn]).astype(np.uint8)
        out.write((json.dumps({"seq": h.get("seq", 0), "n": rn, "h": ih, "w": iw}) + "\\n").encode())
        out.write(arr.tobytes()); out.flush()
    """
)


@pytest.fixture
def stub_env(tmp_path, monkeypatch):
    """A fake seedvr2 checkout + LoRA + stub worker, wired into the restorer."""
    repo = tmp_path / "repo"
    (repo / "src" / "core").mkdir(parents=True)
    (repo / "src" / "core" / "generation_utils.py").touch()
    (repo / "models" / "SEEDVR2").mkdir(parents=True)
    lora = tmp_path / "lora.pt"
    lora.touch()
    worker = tmp_path / "stub_worker.py"
    worker.write_text(_STUB_WORKER)
    monkeypatch.setattr(restorer_pkg, "bundled_script_path", lambda name: worker)
    return {"repo": repo, "lora": lora, "worker": worker}


def _make_restorer(stub_env):
    return Seedvr2LoraRestorer(
        repo_path=str(stub_env["repo"]),
        lora_path=str(stub_env["lora"]),
        device="cpu",
        python_path=sys.executable,
        startup_timeout_s=30.0,
    )


def _frames(t=3, value_step=17):
    """(C,256,256) float32 RGB frames with a distinct per-frame per-channel value."""
    frames = []
    for i in range(t):
        f = torch.zeros(3, 256, 256, dtype=torch.float32)
        for c in range(3):
            f[c] = float((i * 3 + c) * value_step % 256)
        frames.append(f)
    return frames


class TestConstructorValidation:
    def test_rejects_non_4n1_window(self, stub_env):
        with pytest.raises(ValueError, match="4n"):
            Seedvr2LoraRestorer(
                repo_path=str(stub_env["repo"]), lora_path=str(stub_env["lora"]),
                device="cpu", python_path=sys.executable, window=32,
            )

    def test_rejects_bad_overlap(self, stub_env):
        with pytest.raises(ValueError, match="overlap"):
            Seedvr2LoraRestorer(
                repo_path=str(stub_env["repo"]), lora_path=str(stub_env["lora"]),
                device="cpu", python_path=sys.executable, window=33, overlap=33,
            )

    def test_rejects_missing_repo_marker(self, stub_env, tmp_path):
        bad_repo = tmp_path / "notrepo"
        bad_repo.mkdir()
        with pytest.raises(RuntimeError, match="checkout"):
            Seedvr2LoraRestorer(
                repo_path=str(bad_repo), lora_path=str(stub_env["lora"]),
                device="cpu", python_path=sys.executable,
            )

    def test_rejects_missing_lora(self, stub_env):
        with pytest.raises(RuntimeError, match="LoRA"):
            Seedvr2LoraRestorer(
                repo_path=str(stub_env["repo"]), lora_path=str(stub_env["repo"] / "nope.pt"),
                device="cpu", python_path=sys.executable,
            )

    def test_default_python_layout(self, tmp_path):
        assert default_seedvr2_python(tmp_path).name in ("python", "python.exe")

    def test_primary_contract_attributes(self, stub_env, monkeypatch):
        monkeypatch.setenv("STUB_MODE", "ok")
        r = _make_restorer(stub_env)
        try:
            assert r.pad_mode == "zero"
            assert r.tensorrt_active is False
            assert r.input_dtype == torch.float16
            assert r.dtype == torch.float16
        finally:
            r.close()


class TestRawProcessWire:
    def test_roundtrip_inverted_echo(self, stub_env, monkeypatch):
        """The stub inverts bytes; inversion commutes with the BGR flip, so the
        result must be (255 - input)/255 with shape (T,C,256,256)."""
        monkeypatch.setenv("STUB_MODE", "ok")
        r = _make_restorer(stub_env)
        try:
            frames = _frames(t=3)
            out = r.raw_process(frames)
            assert out.shape == (3, 3, 256, 256)
            assert out.dtype == torch.float16
            expected = (255.0 - torch.stack(frames)) / 255.0
            assert torch.allclose(out.float(), expected, atol=1e-3)
        finally:
            r.close()

    def test_wire_is_bgr(self, stub_env, monkeypatch):
        """The stub writes 200 into wire channel 0 (B). After the parent's
        BGR->RGB flip that value must land in RGB channel 2 (blue)."""
        monkeypatch.setenv("STUB_MODE", "tag")
        r = _make_restorer(stub_env)
        try:
            out = r.raw_process(_frames(t=2))
            blue = out[:, 2].float()
            assert torch.allclose(blue, torch.full_like(blue, 200.0 / 255.0), atol=1e-3)
            assert torch.count_nonzero(out[:, 0]) == 0
            assert torch.count_nonzero(out[:, 1]) == 0
        finally:
            r.close()

    def test_quantization_matches_round(self, stub_env, monkeypatch):
        """Fractional inputs are round()-quantized before hitting the wire."""
        monkeypatch.setenv("STUB_MODE", "ok")
        r = _make_restorer(stub_env)
        try:
            f = torch.full((3, 256, 256), 100.6, dtype=torch.float32)
            out = r.raw_process([f])
            expected = (255.0 - 101.0) / 255.0
            assert torch.allclose(out.float(), torch.full_like(out.float(), expected), atol=1e-3)
        finally:
            r.close()


class TestErrorPaths:
    def test_worker_error_raises_after_one_retry(self, stub_env, monkeypatch):
        monkeypatch.setenv("STUB_MODE", "error")
        r = _make_restorer(stub_env)
        try:
            with pytest.raises(RuntimeError, match="worker error"):
                r.raw_process(_frames(t=2))
        finally:
            r.close()

    def test_dead_worker_respawns_and_retries_once(self, stub_env, monkeypatch, tmp_path):
        """First process dies mid-clip; the respawned process (state file now
        present) succeeds, so the clip completes without an exception."""
        state = tmp_path / "died_once"
        monkeypatch.setenv("STUB_MODE", "flaky")
        monkeypatch.setenv("STUB_STATE_FILE", str(state))
        r = _make_restorer(stub_env)
        try:
            frames = _frames(t=2)
            out = r.raw_process(frames)
            assert state.exists()
            expected = (255.0 - torch.stack(frames)) / 255.0
            assert torch.allclose(out.float(), expected, atol=1e-3)
        finally:
            r.close()

    def test_shape_mismatch_is_contract_error(self, stub_env, monkeypatch):
        monkeypatch.setenv("STUB_MODE", "shortcount")
        r = _make_restorer(stub_env)
        try:
            with pytest.raises(Seedvr2ContractError, match="mismatch"):
                r.raw_process(_frames(t=3))
        finally:
            r.close()

    def test_close_is_idempotent(self, stub_env, monkeypatch):
        monkeypatch.setenv("STUB_MODE", "ok")
        r = _make_restorer(stub_env)
        r.close()
        r.close()


class TestPadModePlumbing:
    def test_zero_pad_reaches_crops(self):
        from jasna.crop_buffer import RawCrop, prepare_crops_for_restoration

        crop = torch.full((3, 64, 128), 200.0)
        raw = RawCrop(crop=crop, enlarged_bbox=(0, 0, 128, 64), crop_shape=(64, 128))
        crops, pad_offsets, _ = prepare_crops_for_restoration(
            [raw], torch.device("cpu"), torch.float32, pad_mode="zero"
        )
        padded = crops[0]
        pl, pt = pad_offsets[0]
        assert pt > 0
        # Padding rows must be exactly zero (reflect would mirror the 200s).
        assert torch.count_nonzero(padded[:, :pt, :]) == 0
        assert torch.count_nonzero(padded[:, -pt:, :]) == 0

    def test_default_stays_reflect(self):
        from jasna.crop_buffer import RawCrop, prepare_crops_for_restoration

        crop = torch.full((3, 64, 128), 200.0)
        raw = RawCrop(crop=crop, enlarged_bbox=(0, 0, 128, 64), crop_shape=(64, 128))
        crops, pad_offsets, _ = prepare_crops_for_restoration(
            [raw], torch.device("cpu"), torch.float32
        )
        pl, pt = pad_offsets[0]
        assert torch.count_nonzero(crops[0][:, :pt, :]) > 0

    def test_restoration_pipeline_passes_restorer_pad_mode(self, monkeypatch):
        from jasna import crop_buffer as cb
        from jasna.restorer.restoration_pipeline import RestorationPipeline

        seen = {}
        orig = cb.prepare_crops_for_restoration

        def spy(raw_crops, device, dtype, restoration_size=cb.RESTORATION_SIZE, pad_mode="reflect"):
            seen["pad_mode"] = pad_mode
            return orig(raw_crops, device, dtype, restoration_size, pad_mode)

        import jasna.restorer.restoration_pipeline as rp_mod
        monkeypatch.setattr(rp_mod, "prepare_crops_for_restoration", spy)

        class _ZeroPadRestorer:
            device = torch.device("cpu")
            input_dtype = torch.float32
            pad_mode = "zero"

        rp = RestorationPipeline.__new__(RestorationPipeline)
        rp.restorer = _ZeroPadRestorer()
        crop = torch.full((3, 64, 128), 200.0)
        raw = cb.RawCrop(crop=crop, enlarged_bbox=(0, 0, 128, 64), crop_shape=(64, 128))
        rp._prepare_from_raw_crops([raw])
        assert seen["pad_mode"] == "zero"


class TestWorkerFileHygiene:
    def test_worker_top_level_is_import_light(self):
        """The worker is a verbatim lada-ex copy executed by a foreign venv; its
        top level must not import jasna, torch, or numpy (heavy imports live
        inside functions so argparse/--help work anywhere)."""
        src = (Path(restorer_pkg.__file__).parent / "seedvr2_lora_worker.py").read_text(encoding="utf-8")
        import ast

        tree = ast.parse(src)
        top_imports = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_imports.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_imports.add(node.module.split(".")[0])
        assert "jasna" not in top_imports
        assert "lada" not in top_imports
        assert "torch" not in top_imports


class TestEdgeRevert:
    """_revert_valid_edges: the dark-line fix for zero-pad bleed (A/B finding)."""

    @staticmethod
    def _setup(pt=8, pl=0, nh=240, nw=256, k=4, t=2):
        from jasna.restorer.restoration_pipeline import _revert_valid_edges

        size = 256
        crops = []
        for _ in range(t):
            c = torch.zeros(3, size, size)
            c[:, pt:pt + nh, pl:pl + nw] = 200.0
            crops.append(c)
        raw = torch.full((t, 3, size, size), 0.5)
        pad_offsets = [(pl, pt)] * t
        resize_shapes = [(nh, nw)] * t
        out = _revert_valid_edges(raw, crops, pad_offsets, resize_shapes, k)
        return out, pt, pl, nh, nw, k

    def test_padded_sides_ramp_toward_input(self):
        out, pt, pl, nh, nw, k = self._setup()
        target = 200.0 / 255.0
        for d, w in ((0, 1.0), (1, 0.75), (2, 0.5), (3, 0.25)):
            expected = 0.5 + (target - 0.5) * w
            # top and bottom rows of the valid region (both pad-adjacent here)
            assert torch.allclose(out[0, :, pt + d, 128], torch.tensor(expected), atol=1e-4)
            assert torch.allclose(out[0, :, pt + nh - 1 - d, 128], torch.tensor(expected), atol=1e-4)
        # Row k stays untouched
        assert torch.allclose(out[0, :, pt + k, 128], torch.tensor(0.5))

    def test_unpadded_sides_untouched(self):
        out, pt, pl, nh, nw, k = self._setup()
        # No horizontal padding: left/right columns keep the raw value except
        # where the top/bottom ramps cross them.
        assert torch.allclose(out[0, :, pt + k, 0], torch.tensor(0.5))
        assert torch.allclose(out[0, :, pt + k, nw - 1], torch.tensor(0.5))

    def test_padding_region_untouched(self):
        out, pt, *_ = self._setup()
        assert torch.allclose(out[0, :, pt - 1, 128], torch.tensor(0.5))

    def test_no_padding_is_noop(self):
        out, *_ = self._setup(pt=0, pl=0, nh=256, nw=256)
        assert torch.allclose(out, torch.full_like(out, 0.5))

    def test_corner_composes_both_sides(self):
        out, pt, pl, nh, nw, k = self._setup(pt=8, pl=8, nh=240, nw=240)
        target = 200.0 / 255.0
        corner = out[0, 0, pt, pl].item()
        edge_mid = out[0, 0, pt, pl + 128].item()
        # The corner lerps twice (top then left, both w=1.0) -> at least as
        # close to the input as a single-side edge point.
        assert abs(corner - target) <= abs(edge_mid - target) + 1e-6

    def test_restorer_declares_edge_revert(self):
        assert Seedvr2LoraRestorer.edge_revert_px > 0
