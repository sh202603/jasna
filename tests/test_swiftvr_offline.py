"""Tests for the offline 3-phase SwiftVR mode (no GPU, no SwiftVR checkout).

``--secondary-restoration swiftvr`` runs the FlashVSR offline orchestrator
(``flashvsr_offline``) with the SwiftVR engine: the same Phase 1 dump, bundle,
disk checks and Phase 3 reblend, with the engine record supplying the paths,
flag names, scale, Phase 1 crop view window (no clip cap) and the Phase 2
command. Covered here: the engine-specific parts of the orchestrator, the
engine-independent validation for both engines, the version 2 bundle
(``view_placements``), the Phase 1 hook's view window override, Phase 3's
assembly and blend of a view bundle against the inline path, the Phase 2
command and environment, and the driver's argument parsing / shared worker
functions. ``test_flashvsr_offline.py`` keeps covering the shared machinery.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from jasna.blend_buffer import BlendBuffer
from jasna.pipeline_items import PrimaryRestoreResult
from jasna.restorer import flashvsr_offline as fo
from jasna.restorer import swiftvr_common as sc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(root: Path) -> Path:
    """A fake SwiftVR checkout: restore_clip() API, checkpoint files, both venv layouts."""
    repo = root / "repo"
    (repo / "swiftvr").mkdir(parents=True, exist_ok=True)
    (repo / "swiftvr" / "pipeline.py").write_text(
        "class SwiftVRPipeline:\n    def restore_clip(self, frames_uint8, *, upscale=4):\n        pass\n"
    )
    ck = repo / "checkpoints"
    (ck / "transformer").mkdir(parents=True, exist_ok=True)
    for name in ("reae.safetensors", "prompt_embedding.safetensors"):
        (ck / name).write_bytes(b"x")
    (ck / "transformer" / "config.json").write_text("{}")
    (repo / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
    (repo / ".venv" / "bin" / "python").touch()
    (repo / ".venv" / "Scripts").mkdir(parents=True, exist_ok=True)
    (repo / ".venv" / "Scripts" / "python.exe").touch()
    return repo


def _make_args(tmp_path, **overrides):
    repo = _make_repo(tmp_path)
    inp = tmp_path / "in.mp4"
    inp.touch()
    args = MagicMock()
    args.swiftvr_repo = str(repo)
    args.swiftvr_python = ""
    args.swiftvr_model_dir = ""
    args.swiftvr_scale = 4
    args.swiftvr_accel = True
    args.swiftvr_view_window = 15
    args.swiftvr_bundle_dir = str(tmp_path / "bundle")
    args.swiftvr_keep_bundle = True
    args.input = str(inp)
    args.output = str(tmp_path / "out.mkv")
    args.device = "cuda:0"
    args.codec = "hevc"
    args.encoder_settings = ""
    args.lut = ""
    args.batch_size = 4
    args.video_backend = "native"
    args.decode_backend = "inherit"
    args.encode_backend = "inherit"
    args.stream = False
    args.frame_gen = "none"
    args.retarget_high_fps = False
    args.segments = ""
    args.vr_mode = "off"
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _make_flashvsr_args(tmp_path, **overrides):
    """The FlashVSR counterpart, for the two-engine validation tests."""
    repo = tmp_path / "fvrepo"
    (repo / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
    (repo / ".venv" / "bin" / "python").touch()
    (repo / ".venv" / "Scripts").mkdir(parents=True, exist_ok=True)
    (repo / ".venv" / "Scripts" / "python.exe").touch()
    (repo / "models" / "FlashVSR-v1.1").mkdir(parents=True, exist_ok=True)
    inp = tmp_path / "in.mp4"
    inp.touch()
    args = MagicMock()
    args.flashvsr_repo = str(repo)
    args.flashvsr_python = ""
    args.flashvsr_model_dir = ""
    args.input = str(inp)
    args.output = str(tmp_path / "out.mkv")
    args.stream = False
    args.frame_gen = "none"
    args.retarget_high_fps = False
    args.segments = ""
    args.vr_mode = "off"
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _run_orchestrator(args, *, argv_extra=(), fake_run=None, env_capture=None):
    """Run the SwiftVR engine with subprocess + disk checks patched; return the
    spawned commands."""
    calls = []

    def _fake_run(cmd, env=None):
        calls.append(cmd)
        if env_capture is not None:
            env_capture.append(env)
        if fake_run is not None:
            return fake_run(cmd, env)
        return MagicMock(returncode=0)

    argv = ["--input", args.input, "--output", args.output,
            "--secondary-restoration", "swiftvr", "--swiftvr-repo", args.swiftvr_repo,
            *argv_extra]
    with (
        patch("jasna.restorer.flashvsr_offline.subprocess.run", side_effect=_fake_run),
        patch("jasna.restorer.flashvsr_offline._preflight_bundle_disk"),
        patch("jasna.restorer.flashvsr_offline._gate_phase2_disk"),
        patch.object(sys, "argv", ["jasna", *argv]),
    ):
        fo.run_flashvsr_offline(args, engine="swiftvr")
    return calls


def _make_primary(frame_count=3, *, keep_start=0, keep_end=None, view_placements=None) -> PrimaryRestoreResult:
    ke = keep_end if keep_end is not None else frame_count
    primary = torch.zeros(frame_count, 3, 256, 256)
    primary[:, 0], primary[:, 1], primary[:, 2] = 0.9, 0.5, 0.1
    return PrimaryRestoreResult(
        track_id=3, start_frame=5, frame_count=frame_count, frame_shape=(256, 256),
        frame_device=torch.device("cpu"),
        masks=[torch.ones(16, 16, dtype=torch.bool) for _ in range(frame_count)],
        primary_raw=primary, keep_start=keep_start, keep_end=ke, crossfade_weights=None,
        enlarged_bboxes=[(0, 0, 256, 256)] * frame_count, crop_shapes=[(256, 256)] * frame_count,
        pad_offsets=[(0, 0)] * frame_count, resize_shapes=[(256, 256)] * frame_count,
        view_placements=view_placements,
    )


def _write_manifest(bundle: Path, entries, *, version=fo.BUNDLE_VERSION) -> None:
    fo.manifest_path(bundle).write_text(json.dumps({
        "version": version, "input": "x.mp4", "fps": 30.0, "frame_count": 20, "total_frames": 20,
        "clips": entries,
    }))


def _entry(pr: PrimaryRestoreResult, key: str) -> dict:
    return {"key": key, "track_id": pr.track_id, "start_frame": pr.start_frame,
            "frame_count": pr.frame_count, "keep_start": pr.keep_start, "keep_end": pr.keep_end}


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

class TestFlags:
    def test_bundle_flags_defaults(self):
        import argparse
        p = argparse.ArgumentParser()
        sc.add_swiftvr_arguments(p.add_argument_group("SwiftVR"))
        a = p.parse_args([])
        assert a.swiftvr_bundle_dir == "" and a.swiftvr_keep_bundle is False
        a = p.parse_args(["--swiftvr-bundle-dir", "/b", "--swiftvr-keep-bundle"])
        assert a.swiftvr_bundle_dir == "/b" and a.swiftvr_keep_bundle is True

    def test_repo_help_names_both_modes(self):
        import argparse
        p = argparse.ArgumentParser()
        sc.add_swiftvr_arguments(p.add_argument_group("SwiftVR"))
        text = p.format_help()
        assert "swiftvr and" in text and "swiftvr-inline" in text


# ---------------------------------------------------------------------------
# Phase 2 command (swiftvr_common.swiftvr_phase2_command)
# ---------------------------------------------------------------------------

class TestPhase2Command:
    def _cmd(self, tmp_path, monkeypatch, env_value=None, **over):
        args = _make_args(tmp_path, **over)
        if env_value is None:
            monkeypatch.delenv("JASNA_SWIFTVR_COLOR_FIX", raising=False)
        else:
            monkeypatch.setenv("JASNA_SWIFTVR_COLOR_FIX", env_value)
        repo = Path(args.swiftvr_repo)
        return sc.swiftvr_phase2_command(args, tmp_path / "bundle", repo, repo / ".venv" / "bin" / "python",
                                         repo / "checkpoints")

    def test_driver_and_accel_on(self, tmp_path, monkeypatch):
        cmd, env = self._cmd(tmp_path, monkeypatch)
        assert cmd[0].endswith("python") and cmd[1].endswith("swiftvr_phase2_driver.py")
        assert cmd[cmd.index("--bundle-dir") + 1] == str(tmp_path / "bundle")
        assert cmd[cmd.index("--scale") + 1] == "4"
        assert cmd[cmd.index("--device") + 1] == "cuda:0"
        assert "--fp8-dit" in cmd and "--torch-compile" in cmd
        assert "--color-fix-method" not in cmd  # driver default (wavelet)
        assert "--max-clip-size" not in cmd and "--clip-len" not in cmd

    def test_accel_off_passes_neither_part(self, tmp_path, monkeypatch):
        cmd, _ = self._cmd(tmp_path, monkeypatch, swiftvr_accel=False, swiftvr_scale=2)
        assert "--fp8-dit" not in cmd and "--torch-compile" not in cmd
        assert cmd[cmd.index("--scale") + 1] == "2"

    @pytest.mark.parametrize("method", ["adain", "wavelet", "none"])
    def test_color_fix_override_forwarded(self, tmp_path, monkeypatch, method):
        cmd, _ = self._cmd(tmp_path, monkeypatch, env_value=method)
        assert cmd[cmd.index("--color-fix-method") + 1] == method

    def test_invalid_color_fix_override_rejected(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="JASNA_SWIFTVR_COLOR_FIX"):
            self._cmd(tmp_path, monkeypatch, env_value="lab")

    def test_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PYTHONPATH", "/should/not/leak")
        _, env = self._cmd(tmp_path, monkeypatch)
        assert "PYTHONPATH" not in env
        assert env["TQDM_DISABLE"] == "1"
        if sys.platform == "win32":
            assert env["PYTHONUTF8"] == "1"
        else:
            assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


# ---------------------------------------------------------------------------
# Orchestrator with the SwiftVR engine
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def test_runs_three_phases_in_order(self, tmp_path, capsys):
        args = _make_args(tmp_path)
        calls = _run_orchestrator(args)
        assert len(calls) == 3
        assert calls[0][1:5] == ["-m", "jasna", "--flashvsr-phase", "dump"]  # internal hook name
        assert calls[1][0] == str(Path(args.swiftvr_repo) / ".venv" / "bin" / "python") \
            or calls[1][0].endswith("python.exe")
        assert calls[1][1].endswith("swiftvr_phase2_driver.py")
        assert calls[2][1:5] == ["-m", "jasna", "--flashvsr-phase", "reblend"]
        assert "SwiftVR offline restoration complete" in capsys.readouterr().out

    def test_dump_cfg_has_view_window_and_no_clip_cap(self, tmp_path):
        args = _make_args(tmp_path)
        captured = {}

        def fake_run(cmd, env):
            if "dump" in cmd:
                captured["dump"] = json.loads(Path(cmd[-1]).read_text())
            return MagicMock(returncode=0)

        _run_orchestrator(args, fake_run=fake_run, argv_extra=["--max-clip-size", "180"])
        dump = captured["dump"]
        assert dump["view_window"] == 15  # the default --swiftvr-view-window reaches Phase 1
        argv = dump["argv"]
        assert argv[argv.index("--secondary-restoration") + 1] == "none"
        assert argv[argv.index("--max-clip-size") + 1] == "180"  # the user's value, uncapped
        assert argv.count("--max-clip-size") == 1

    def test_dump_cfg_view_window_zero(self, tmp_path):
        args = _make_args(tmp_path, swiftvr_view_window=0)
        captured = {}

        def fake_run(cmd, env):
            if "dump" in cmd:
                captured["dump"] = json.loads(Path(cmd[-1]).read_text())
            return MagicMock(returncode=0)

        _run_orchestrator(args, fake_run=fake_run)
        assert captured["dump"]["view_window"] == 0
        assert "--max-clip-size" not in captured["dump"]["argv"]  # never inserted for SwiftVR

    def test_phase2_env_reaches_subprocess(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PYTHONPATH", "/should/not/leak")
        envs = []
        _run_orchestrator(_make_args(tmp_path), env_capture=envs)
        env2 = envs[1]
        assert "PYTHONPATH" not in env2
        if sys.platform != "win32":
            assert env2["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"

    def test_phase_failure_names_engine_and_flag(self, tmp_path, capsys):
        args = _make_args(tmp_path)
        with pytest.raises(RuntimeError, match="SwiftVR Phase 1"):
            _run_orchestrator(args, fake_run=lambda cmd, env: MagicMock(returncode=1))
        out = capsys.readouterr().out
        assert "SwiftVR run failed" in out
        assert f"--swiftvr-bundle-dir {tmp_path / 'bundle'}" in out
        assert "--flashvsr-bundle-dir" not in out

    def test_disk_checks_get_engine_wording(self, tmp_path):
        args = _make_args(tmp_path, swiftvr_scale=2)
        seen = {}
        argv = ["--input", args.input, "--output", args.output]
        with (
            patch("jasna.restorer.flashvsr_offline.subprocess.run",
                  side_effect=lambda cmd, env=None: MagicMock(returncode=0)),
            patch("jasna.restorer.flashvsr_offline._preflight_bundle_disk",
                  side_effect=lambda b, i, scale, **kw: seen.setdefault("pre", (scale, kw))),
            patch("jasna.restorer.flashvsr_offline._gate_phase2_disk",
                  side_effect=lambda b, scale, **kw: seen.setdefault("gate", (scale, kw))),
            patch.object(sys, "argv", ["jasna", *argv]),
        ):
            fo.run_flashvsr_offline(args, engine="swiftvr")
        for scale, kw in seen.values():
            assert scale == 2
            assert kw == {"display": "SwiftVR", "bundle_dir_flag": "--swiftvr-bundle-dir"}

    def test_temp_bundle_prefix_and_cleanup(self, tmp_path, monkeypatch):
        args = _make_args(tmp_path, swiftvr_bundle_dir="", swiftvr_keep_bundle=False)
        made = {}

        def fake_mkdtemp(prefix):
            d = tmp_path / (prefix + "x")
            d.mkdir()
            made["dir"] = d
            return str(d)

        monkeypatch.setattr(fo.tempfile, "mkdtemp", fake_mkdtemp)
        _run_orchestrator(args)
        assert made["dir"].name.startswith("jasna_swiftvr_")
        assert not made["dir"].exists()  # removed on success without --swiftvr-keep-bundle

    def test_unknown_engine_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown offline engine"):
            fo.run_flashvsr_offline(_make_args(tmp_path), engine="bogus")

    def test_gate_message_uses_engine_flag(self, tmp_path):
        _write_manifest(tmp_path, [{"key": "clip_0_0", "track_id": 0, "start_frame": 0,
                                    "frame_count": 100, "keep_start": 0, "keep_end": 100}])
        free = MagicMock()
        free.free = 0
        with patch("jasna.restorer.flashvsr_offline.shutil.disk_usage", return_value=free):
            with pytest.raises(RuntimeError, match="SwiftVR Phase 2.*--swiftvr-bundle-dir"):
                fo._gate_phase2_disk(tmp_path, 4, display="SwiftVR", bundle_dir_flag="--swiftvr-bundle-dir")


# ---------------------------------------------------------------------------
# Validation: engine-independent constraints for both engines, SwiftVR paths
# ---------------------------------------------------------------------------

_ENGINE_ARGS = {"flashvsr": _make_flashvsr_args, "swiftvr": _make_args}


class TestValidation:
    @pytest.mark.parametrize("engine", ["flashvsr", "swiftvr"])
    @pytest.mark.parametrize("override, match", [
        ({"stream": True}, "file-output only"),
        ({"frame_gen": "2x"}, "frame-gen"),
        ({"retarget_high_fps": True}, "retarget-high-fps"),
        ({"segments": "10-25"}, "segments"),
        ({"vr_mode": "sbs"}, "VR processing"),
        ({"vr_mode": "sbs-fisheye"}, "VR processing"),
    ])
    def test_common_rejections_name_the_mode(self, tmp_path, engine, override, match):
        args = _ENGINE_ARGS[engine](tmp_path, **override)
        with pytest.raises(ValueError, match=match) as ei:
            fo._ENGINES[engine](args)
        assert f"--secondary-restoration {engine}" in str(ei.value)

    @pytest.mark.parametrize("engine", ["flashvsr", "swiftvr"])
    def test_folder_and_image_input_rejected(self, tmp_path, engine):
        d = tmp_path / "folder"
        d.mkdir()
        with pytest.raises(ValueError, match="folder input"):
            fo._ENGINES[engine](_ENGINE_ARGS[engine](tmp_path, input=str(d)))
        img = tmp_path / "pic.png"
        img.touch()
        with pytest.raises(ValueError, match="video-only"):
            fo._ENGINES[engine](_ENGINE_ARGS[engine](tmp_path, input=str(img)))

    @pytest.mark.parametrize("engine", ["flashvsr", "swiftvr"])
    def test_vr_auto_rejected_only_when_detected(self, tmp_path, engine):
        from jasna.vr180 import VrModeResolution

        args = _ENGINE_ARGS[engine](tmp_path, vr_mode="auto")
        with (
            patch("jasna.media.get_video_meta_data", return_value=MagicMock()),
            patch("jasna.vr180.resolve_vr_mode",
                  return_value=VrModeResolution("auto", "sbs", "2:1 high-res", 2.0, "fisheye190")),
        ):
            with pytest.raises(ValueError, match="detected VR content"):
                fo._ENGINES[engine](args)
        with (
            patch("jasna.media.get_video_meta_data", return_value=MagicMock()),
            patch("jasna.vr180.resolve_vr_mode",
                  return_value=VrModeResolution("auto", "off", "not VR", 1.78, "off")),
        ):
            assert fo._ENGINES[engine](args).name == engine

    def test_flag_checks_precede_path_resolution(self, tmp_path):
        # A cheap flag error is reported before any path is probed.
        with pytest.raises(ValueError, match="file-output only"):
            fo._ENGINES["swiftvr"](_make_args(tmp_path, stream=True, swiftvr_repo=""))

    def test_swiftvr_path_errors_name_swiftvr_flags(self, tmp_path):
        with pytest.raises(ValueError, match="--swiftvr-repo is required for --secondary-restoration swiftvr$"):
            fo._ENGINES["swiftvr"](_make_args(tmp_path, swiftvr_repo=""))
        with pytest.raises(FileNotFoundError, match="--swiftvr-repo not found"):
            fo._ENGINES["swiftvr"](_make_args(tmp_path, swiftvr_repo=str(tmp_path / "nope")))
        with pytest.raises(FileNotFoundError, match="SwiftVR Python not found"):
            fo._ENGINES["swiftvr"](_make_args(tmp_path, swiftvr_python=str(tmp_path / "nopy")))
        with pytest.raises(FileNotFoundError, match="--swiftvr-model-dir not found"):
            fo._ENGINES["swiftvr"](_make_args(tmp_path, swiftvr_model_dir=str(tmp_path / "nomodel")))

    def test_upstream_checkout_rejected(self, tmp_path):
        args = _make_args(tmp_path)
        (Path(args.swiftvr_repo) / "swiftvr" / "pipeline.py").write_text(
            "class SwiftVRPipeline:\n    def restore_video(self, a, b):\n        pass\n")
        with pytest.raises(RuntimeError, match="restore_clip"):
            fo._ENGINES["swiftvr"](args)

    def test_engine_record(self, tmp_path):
        eng = fo._ENGINES["swiftvr"](_make_args(tmp_path, swiftvr_scale=2, swiftvr_view_window=7))
        assert eng.name == "swiftvr" and eng.display == "SwiftVR"
        assert eng.bundle_dir_flag == "--swiftvr-bundle-dir"
        assert eng.scale == 2 and eng.view_window == 7 and eng.max_clip_frames is None
        assert eng.input_path.name == "in.mp4"
        fv = fo._ENGINES["flashvsr"](_make_flashvsr_args(tmp_path, flashvsr_scale=2,
                                                          flashvsr_max_clip_frames=32))
        assert fv.view_window == 0 and fv.max_clip_frames == 32 and fv.scale == 2


# ---------------------------------------------------------------------------
# Bundle version 2: view_placements
# ---------------------------------------------------------------------------

class TestBundleV2:
    def test_version_is_2(self):
        assert fo.BUNDLE_VERSION == 2

    def test_view_placements_round_trip(self, tmp_path):
        placements = [(1.0, 0.5, 1.0, -0.25), (1.01, 0.75, 0.99, 0.0), (1.0, 1.0, 1.0, 0.25)]
        key = fo.write_primary_clip(tmp_path, _make_primary(3, view_placements=placements))
        geom = fo.read_clip_geom(tmp_path, key)
        assert geom["view_placements"] == [list(p) for p in placements]

    def test_view_placements_null_without_view(self, tmp_path):
        key = fo.write_primary_clip(tmp_path, _make_primary(3))
        geom = fo.read_clip_geom(tmp_path, key)
        assert "view_placements" in geom and geom["view_placements"] is None

    def _write_fvsr(self, bundle, key, frame_count, size=512):
        np.savez(fo.fvsr_npz_path(bundle, key), restored_u8=np.zeros((frame_count, 3, size, size), np.uint8))

    def test_assemble_slices_view_placements_to_keep_window(self, tmp_path):
        placements = [(1.0, float(i), 1.0, -float(i)) for i in range(5)]
        pr = _make_primary(5, keep_start=1, keep_end=4, view_placements=placements)
        key = fo.write_primary_clip(tmp_path, pr)
        _write_manifest(tmp_path, [_entry(pr, key)])
        self._write_fvsr(tmp_path, key, 5)
        sr = fo._load_clip_sr(tmp_path, key, torch.device("cpu"))
        assert sr.view_placements == placements[1:4]
        assert all(isinstance(p, tuple) and len(p) == 4 for p in sr.view_placements)
        assert sr.clip_keep_offset == 1 and sr.keep_end == 3 and len(sr.restored_frames) == 3

    def test_assemble_without_view_gives_none(self, tmp_path):
        pr = _make_primary(3)
        key = fo.write_primary_clip(tmp_path, pr)
        _write_manifest(tmp_path, [_entry(pr, key)])
        self._write_fvsr(tmp_path, key, 3)
        assert fo._load_clip_sr(tmp_path, key, torch.device("cpu")).view_placements is None

    def test_version_1_bundle_reads_through_legacy_path(self, tmp_path):
        # A clip written by a version 1 jasna: no view_placements key at all.
        pr = _make_primary(3)
        key = fo.clip_key(pr.track_id, pr.start_frame)
        geom = fo._geom_from_primary(pr)
        del geom["view_placements"]
        np.savez_compressed(
            fo.clip_npz_path(tmp_path, key),
            primary_u8=np.zeros((3, 3, 256, 256), np.uint8),
            masks_packed=np.packbits(np.ones((3, 16, 16), bool)), mask_shape=np.asarray([3, 16, 16]),
            geom=np.asarray(json.dumps(geom)),
        )
        _write_manifest(tmp_path, [_entry(pr, key)], version=1)
        self._write_fvsr(tmp_path, key, 3)
        assert fo.read_manifest(tmp_path)["version"] == 1
        frame_to_tracks, loads_at = fo._plan_reblend(tmp_path)
        assert loads_at == {5: [key]}
        assert fo._load_clip_sr(tmp_path, key, torch.device("cpu")).view_placements is None

    def test_newer_version_rejected(self, tmp_path):
        _write_manifest(tmp_path, [], version=3)
        with pytest.raises(RuntimeError, match="version 3, newer"):
            fo.read_manifest(tmp_path)
        with pytest.raises(RuntimeError, match="newer"):
            fo._plan_reblend(tmp_path)
        with pytest.raises(RuntimeError, match="newer"):
            fo._gate_phase2_disk(tmp_path)


# ---------------------------------------------------------------------------
# Phase 1 hook: the crop view window override, and Phase 3 blending the view
# ---------------------------------------------------------------------------

def _jitter_fixture(monkeypatch, t=8):
    """The jittering-track clip of test_restoration_pipeline (its bbox moves,
    so the smoothed view differs from the own grid)."""
    import test_restoration_pipeline as trp

    clip, frames, raw_crops = trp._make_jitter_clip_and_frames(monkeypatch, t=t)
    return trp, clip, frames, raw_crops


def _guard_pipeline_class(monkeypatch):
    """Record the class attributes the dump hook replaces so the test restores them."""
    import jasna.restorer.restoration_pipeline as rp

    monkeypatch.setattr(rp.RestorationPipeline, "prepare_and_run_primary",
                        rp.RestorationPipeline.prepare_and_run_primary)
    monkeypatch.setattr(rp.RestorationPipeline, "view_smoothing_window",
                        rp.RestorationPipeline.__dict__["view_smoothing_window"])
    return rp


class TestPhase1Hook:
    def test_view_window_override_builds_the_view(self, tmp_path, monkeypatch):
        from jasna.tracking.crop_view import compute_view_placements

        rp = _guard_pipeline_class(monkeypatch)
        trp, clip, frames, raw_crops = _jitter_fixture(monkeypatch)
        # The own-grid reference, before the hook (its override is class-wide).
        plain = rp.RestorationPipeline(restorer=trp._IdentityRestorer())
        assert plain.view_smoothing_window == 0
        own = (plain.prepare_and_run_primary(clip, raw_crops, (120, 160), 0, 8, None).primary_raw
               .clamp(0, 1).mul(255).round().to(torch.uint8).numpy())

        # Phase 1 runs without a secondary restorer: the window comes from the config.
        fo._install_dump_hook(tmp_path, view_window=15)
        pipe = rp.RestorationPipeline(restorer=trp._IdentityRestorer())
        assert pipe.view_smoothing_window == 15
        pr = pipe.prepare_and_run_primary(clip, raw_crops, (120, 160), 0, 8, None)
        assert pr.view_placements is not None and len(pr.view_placements) == 8

        assert len(fo._DUMPED_CLIPS) == 1
        geom = fo.read_clip_geom(tmp_path, fo._DUMPED_CLIPS[0]["key"])
        _, want = compute_view_placements(pr.enlarged_bboxes, pr.pad_offsets, pr.resize_shapes, 15)
        assert np.allclose(np.asarray(geom["view_placements"]), want)
        # the dumped crops are the re-viewed ones, not the own grid
        with np.load(fo.clip_npz_path(tmp_path, fo._DUMPED_CLIPS[0]["key"])) as d:
            dumped = d["primary_u8"]
        assert dumped.shape == own.shape and not np.array_equal(dumped, own)

    def test_view_window_zero_keeps_the_own_grid(self, tmp_path, monkeypatch):
        rp = _guard_pipeline_class(monkeypatch)
        trp, clip, frames, raw_crops = _jitter_fixture(monkeypatch)
        fo._install_dump_hook(tmp_path, view_window=0)
        pipe = rp.RestorationPipeline(restorer=trp._IdentityRestorer())
        assert pipe.view_smoothing_window == 0
        pr = pipe.prepare_and_run_primary(clip, raw_crops, (120, 160), 0, 8, None)
        assert pr.view_placements is None
        geom = fo.read_clip_geom(tmp_path, fo._DUMPED_CLIPS[0]["key"])
        assert geom["view_placements"] is None

    def test_single_frame_clip_has_no_view(self, tmp_path, monkeypatch):
        rp = _guard_pipeline_class(monkeypatch)
        trp, clip, frames, raw_crops = _jitter_fixture(monkeypatch, t=1)
        fo._install_dump_hook(tmp_path, view_window=15)
        pipe = rp.RestorationPipeline(restorer=trp._IdentityRestorer())
        pipe.prepare_and_run_primary(clip, raw_crops, (120, 160), 0, 1, None)
        assert fo.read_clip_geom(tmp_path, fo._DUMPED_CLIPS[0]["key"])["view_placements"] is None

    def test_override_is_gone_after_the_test(self):
        # The guard above restored the class; a plain pipeline reads its secondary again.
        import jasna.restorer.restoration_pipeline as rp
        assert isinstance(rp.RestorationPipeline.__dict__["view_smoothing_window"], property)
        assert rp.RestorationPipeline(restorer=MagicMock()).view_smoothing_window == 0


class TestPhase3ViewBlend:
    def test_bundle_blend_matches_inline_blend(self, tmp_path, monkeypatch):
        """A view bundle assembled by Phase 3 blends exactly like the same
        SecondaryRestoreResult built inline (build_secondary_result), and the
        view path is really taken (it differs from the own-grid composite)."""
        import torch.nn.functional as F

        rp = _guard_pipeline_class(monkeypatch)
        trp, clip, frames, raw_crops = _jitter_fixture(monkeypatch)
        fo._install_dump_hook(tmp_path, view_window=15)
        pipe = rp.RestorationPipeline(restorer=trp._IdentityRestorer())
        pr = pipe.prepare_and_run_primary(clip, raw_crops, (120, 160), 2, 7, None)
        key = fo._DUMPED_CLIPS[0]["key"]
        _write_manifest(tmp_path, [_entry(pr, key)])

        # A 2x "secondary" of the dumped view crops, written as Phase 2 would.
        with np.load(fo.clip_npz_path(tmp_path, key)) as d:
            dumped = torch.from_numpy(d["primary_u8"])
        up = F.interpolate(dumped.float(), scale_factor=2, mode="bilinear", align_corners=False)
        restored = up.clamp(0, 255).round().to(torch.uint8)
        np.savez(fo.fvsr_npz_path(tmp_path, key), restored_u8=restored.numpy())

        ones = lambda m, b, s: torch.ones((b[3] - b[1], b[2] - b[0]))  # noqa: E731
        ks, ke = pr.keep_start, pr.keep_end

        sr_bundle = fo._load_clip_sr(tmp_path, key, torch.device("cpu"))
        sr_inline = pipe.build_secondary_result(pr, list(restored[ks:ke].unbind(0)))
        assert sr_bundle.view_placements == sr_inline.view_placements
        assert sr_bundle.clip_keep_offset == sr_inline.clip_keep_offset == ks

        def blend(sr):
            bb = BlendBuffer(device=torch.device("cpu"), blend_mask_fn=ones)
            for i in range(ks, ke):
                bb.register_frame(i, {clip.track_id})
            bb.add_result(sr)
            return [bb.blend_frame(i, frames[i]) for i in range(ks, ke)]

        out_bundle = blend(sr_bundle)
        out_inline = blend(sr_inline)
        for a, b in zip(out_bundle, out_inline):
            assert torch.equal(a, b)
        # sanity: the view path composites differently from the legacy path
        sr_legacy = fo._load_clip_sr(tmp_path, key, torch.device("cpu"))
        sr_legacy.view_placements = None
        assert any(not torch.equal(a, b) for a, b in zip(out_bundle, blend(sr_legacy)))


# ---------------------------------------------------------------------------
# Phase 2 driver (importable without a GPU / SwiftVR checkout)
# ---------------------------------------------------------------------------

class TestDriver:
    def test_arg_parsing(self):
        from jasna.restorer import swiftvr_phase2_driver as drv
        base = ["prog", "--bundle-dir", "/b", "--repo", "/r", "--model-dir", "/m"]
        with patch.object(sys, "argv", base):
            a = drv._parse_args()
        assert (a.bundle_dir, a.repo, a.model_dir, a.device) == ("/b", "/r", "/m", "cuda:0")
        assert a.scale == 4 and a.clip_len == 24 and a.color_fix_method == "wavelet"
        assert a.fp8_dit is False and a.torch_compile is False and a.verbose is False
        with patch.object(sys, "argv", [*base, "--scale", "2", "--fp8-dit", "--torch-compile",
                                        "--color-fix-method", "none", "--clip-len", "48"]):
            a = drv._parse_args()
        assert a.scale == 2 and a.fp8_dit and a.torch_compile and a.color_fix_method == "none"
        assert a.clip_len == 48
        for bad in (["--scale", "3"], ["--color-fix-method", "lab"]):
            with patch.object(sys, "argv", [*base, *bad]):
                with pytest.raises(SystemExit):
                    drv._parse_args()

    def test_shares_the_inline_worker_functions(self):
        from jasna.restorer import swiftvr_inline_worker as w
        from jasna.restorer import swiftvr_phase2_driver as drv
        assert Path(drv._worker.__file__) == Path(w.__file__)
        for name in ("_decide_accel", "_load_pipeline", "_warmup", "_restore_checked", "_color_fix_frames_gpu"):
            assert getattr(drv._worker, name).__code__.co_filename == w.__file__
        assert drv.DEFAULT_CLIP_LEN == w.DEFAULT_CLIP_LEN == 24

    class _NearestPipe:
        """restore_clip stand-in: nearest 'upscale' of (T,H,W,3) uint8."""
        def __init__(self):
            self.calls = []

        def restore_clip(self, lq, *, upscale, clip_len):
            self.calls.append((tuple(lq.shape), upscale, clip_len))
            return lq.repeat_interleave(upscale, dim=1).repeat_interleave(upscale, dim=2)

    def test_restore_bundle_clip_layout_no_color_fix(self):
        from jasna.restorer import swiftvr_phase2_driver as drv
        primary = np.zeros((3, 3, 256, 256), np.uint8)
        primary[:, 0], primary[:, 1], primary[:, 2] = 230, 128, 26  # distinct channels: RGB kept
        pipe = self._NearestPipe()
        out = drv._restore_bundle_clip(pipe, torch, primary, "cpu", 2, 24, "none")
        assert out.shape == (3, 3, 512, 512) and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"]
        assert pipe.calls == [((3, 256, 256, 3), 2, 24)]  # HWC to the pipeline, scale + clip_len through
        assert int(out[0, 0].min()) == 230 and int(out[0, 1].min()) == 128 and int(out[0, 2].min()) == 26

    def test_restore_bundle_clip_applies_the_worker_color_fix(self):
        from jasna.restorer import swiftvr_phase2_driver as drv
        g = torch.Generator().manual_seed(0)
        primary = torch.randint(0, 256, (2, 3, 256, 256), generator=g, dtype=torch.uint8).numpy()
        pipe = self._NearestPipe()
        fixed = drv._restore_bundle_clip(pipe, torch, primary, "cpu", 2, 24, "wavelet")
        plain = drv._restore_bundle_clip(pipe, torch, primary, "cpu", 2, 24, "none")
        assert fixed.shape == plain.shape == (2, 3, 512, 512)
        assert not np.array_equal(fixed, plain)
        # same numbers as the worker's GPU color fix on the same HWC tensors
        lq = torch.from_numpy(np.ascontiguousarray(primary.transpose(0, 2, 3, 1)))
        ref = drv._worker._color_fix_frames_gpu(pipe.restore_clip(lq, upscale=2, clip_len=24), lq, "wavelet")
        assert np.array_equal(fixed, ref.permute(0, 3, 1, 2).numpy())

    def test_restore_bundle_clip_frame_count_contract(self):
        from jasna.restorer import swiftvr_phase2_driver as drv

        class _Short:
            def restore_clip(self, lq, *, upscale, clip_len):
                return lq[:-1].repeat_interleave(upscale, 1).repeat_interleave(upscale, 2)

        with pytest.raises(RuntimeError, match="returned 2 frames for a 3-frame clip"):
            drv._restore_bundle_clip(_Short(), torch, np.zeros((3, 3, 256, 256), np.uint8), "cpu", 2, 24, "none")

    def test_runs_help_without_swiftvr(self):
        from jasna.restorer import bundled_script_path
        script = bundled_script_path("swiftvr_phase2_driver.py")
        res = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
        assert res.returncode == 0 and "--fp8-dit" in res.stdout and "--bundle-dir" in res.stdout
