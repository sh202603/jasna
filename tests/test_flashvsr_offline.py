from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from jasna.blend_buffer import BlendBuffer
from jasna.pipeline_items import PrimaryRestoreResult
from jasna.restorer import flashvsr_offline as fo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_primary(
    *,
    track_id: int = 7,
    start_frame: int = 2,
    frame_count: int = 3,
    frame_shape: tuple[int, int] = (256, 256),
    keep_start: int = 0,
    keep_end: int | None = None,
    mask_hw: tuple[int, int] = (16, 16),
    rgb: tuple[float, float, float] = (0.9, 0.5, 0.1),
    crossfade_weights: dict[int, float] | None = None,
) -> PrimaryRestoreResult:
    ke = keep_end if keep_end is not None else frame_count
    fh, fw = frame_shape
    primary = torch.zeros(frame_count, 3, 256, 256)
    primary[:, 0] = rgb[0]
    primary[:, 1] = rgb[1]
    primary[:, 2] = rgb[2]
    Hm, Wm = mask_hw
    return PrimaryRestoreResult(
        track_id=track_id,
        start_frame=start_frame,
        frame_count=frame_count,
        frame_shape=frame_shape,
        frame_device=torch.device("cpu"),
        masks=[torch.ones(Hm, Wm, dtype=torch.bool) for _ in range(frame_count)],
        primary_raw=primary,
        keep_start=keep_start,
        keep_end=ke,
        crossfade_weights=crossfade_weights,
        enlarged_bboxes=[(0, 0, fw, fh)] * frame_count,
        crop_shapes=[(fh, fw)] * frame_count,
        pad_offsets=[(0, 0)] * frame_count,
        resize_shapes=[(fh, fw)] * frame_count,
    )


def _write_manifest(bundle: Path, pr: PrimaryRestoreResult, key: str, *, total_frames: int = 20) -> None:
    manifest = {
        "version": fo.BUNDLE_VERSION,
        "input": "x.mp4",
        "fps": 30.0,
        "frame_count": total_frames,
        "total_frames": total_frames,
        "clips": [{
            "key": key,
            "track_id": pr.track_id,
            "start_frame": pr.start_frame,
            "frame_count": pr.frame_count,
            "keep_start": pr.keep_start,
            "keep_end": pr.keep_end,
        }],
    }
    fo.manifest_path(bundle).write_text(json.dumps(manifest))


def _write_fvsr(bundle: Path, key: str, frame_count: int, rgb=(200, 100, 50), size: int = 1024) -> None:
    restored = np.zeros((frame_count, 3, size, size), dtype=np.uint8)
    restored[:, 0] = rgb[0]
    restored[:, 1] = rgb[1]
    restored[:, 2] = rgb[2]
    np.savez(fo.fvsr_npz_path(bundle, key), restored_u8=restored)


# ---------------------------------------------------------------------------
# Bundle round-trip
# ---------------------------------------------------------------------------

class TestBundleRoundTrip:
    def test_primary_clip_serializes_all_fields_full_length(self, tmp_path):
        pr = _make_primary(frame_count=5)
        key = fo.write_primary_clip(tmp_path, pr)
        assert key == "clip_7_2"

        geom = fo.read_clip_geom(tmp_path, key)
        assert geom["track_id"] == 7
        assert geom["start_frame"] == 2
        assert geom["frame_count"] == 5
        assert geom["frame_shape"] == [256, 256]
        assert geom["keep_start"] == 0 and geom["keep_end"] == 5
        assert len(geom["enlarged_bboxes"]) == 5
        assert len(geom["crop_shapes"]) == 5
        assert len(geom["pad_offsets"]) == 5
        assert len(geom["resize_shapes"]) == 5
        assert geom["crossfade_weights"] is None

    def test_masks_round_trip(self, tmp_path):
        pr = _make_primary(frame_count=4, mask_hw=(12, 20))
        key = fo.write_primary_clip(tmp_path, pr)
        masks = fo.read_clip_masks(tmp_path, key)
        assert masks.shape == (4, 12, 20)
        assert masks.dtype == bool
        assert masks.all()

    def test_primary_u8_preserves_rgb_channel_order(self, tmp_path):
        # Distinct per-channel values catch an RGB/BGR swap in the round-trip.
        pr = _make_primary(rgb=(0.9, 0.5, 0.1))
        key = fo.write_primary_clip(tmp_path, pr)
        with np.load(fo.clip_npz_path(tmp_path, key)) as d:
            pu8 = d["primary_u8"]
        assert pu8.shape[1:] == (3, 256, 256)
        assert abs(int(pu8[0, 0].mean()) - 230) <= 2  # R = 0.9*255
        assert abs(int(pu8[0, 1].mean()) - 128) <= 2  # G = 0.5*255
        assert abs(int(pu8[0, 2].mean()) - 26) <= 2   # B = 0.1*255

    def test_crossfade_weights_int_keys_restored(self, tmp_path):
        pr = _make_primary(crossfade_weights={0: 0.25, 2: 0.75})
        key = fo.write_primary_clip(tmp_path, pr)
        geom = fo.read_clip_geom(tmp_path, key)
        cw = geom["crossfade_weights"]
        assert cw == {0: 0.25, 2: 0.75}
        assert all(isinstance(k, int) for k in cw)


# ---------------------------------------------------------------------------
# Phase 3 assemble + reblend
# ---------------------------------------------------------------------------

class TestReblend:
    def test_full_keep_window_frame_registration(self, tmp_path):
        pr = _make_primary(start_frame=2, frame_count=3, keep_start=0, keep_end=3)
        key = fo.write_primary_clip(tmp_path, pr)
        _write_manifest(tmp_path, pr, key)
        _write_fvsr(tmp_path, key, frame_count=3)

        frame_to_tracks, loads_at = fo._plan_reblend(tmp_path)
        assert set(frame_to_tracks) == {2, 3, 4}
        assert loads_at == {2: [key]}  # loaded the moment decode reaches frame 2

        sr = fo._load_clip_sr(tmp_path, key, torch.device("cpu"))
        assert sr.keep_end == 3
        assert sr.clip_keep_offset == 0
        assert len(sr.restored_frames) == 3
        assert tuple(sr.restored_frames[0].shape) == (3, 1024, 1024)

    def test_keep_window_slice(self, tmp_path):
        # frame_count 5, keep frames [1:4]  ->  3 kept, offset 1, mapped to frames 3,4,5
        pr = _make_primary(start_frame=2, frame_count=5, keep_start=1, keep_end=4)
        key = fo.write_primary_clip(tmp_path, pr)
        _write_manifest(tmp_path, pr, key)
        _write_fvsr(tmp_path, key, frame_count=5)

        frame_to_tracks, loads_at = fo._plan_reblend(tmp_path)
        # kept frames: start_frame(2) + offset(1) .. + keep_end(3)  -> 3,4,5
        assert set(frame_to_tracks) == {3, 4, 5}
        assert loads_at == {3: [key]}

        sr = fo._load_clip_sr(tmp_path, key, torch.device("cpu"))
        assert sr.frame_count == 5
        assert sr.clip_keep_offset == 1
        assert sr.keep_end == 3            # 4 - 1
        assert len(sr.restored_frames) == 3
        assert len(sr.masks) == 3
        assert len(sr.enlarged_bboxes) == 3

    def test_reblend_preserves_rgb_end_to_end(self, tmp_path):
        pr = _make_primary(start_frame=0, frame_count=2)
        key = fo.write_primary_clip(tmp_path, pr)
        _write_manifest(tmp_path, pr, key)
        _write_fvsr(tmp_path, key, frame_count=2, rgb=(200, 100, 50))

        device = torch.device("cpu")
        frame_to_tracks, loads_at = fo._plan_reblend(tmp_path)
        bb = BlendBuffer(device=device)
        for f, tracks in frame_to_tracks.items():
            bb.register_frame(f, tracks)
        for key in loads_at[0]:
            bb.add_result(fo._load_clip_sr(tmp_path, key, device))

        original = torch.zeros(3, 256, 256, dtype=torch.uint8)
        blended = bb.blend_frame(0, original)
        center = blended[:, 128, 128].tolist()
        assert 185 <= center[0] <= 205   # R
        assert 90 <= center[1] <= 110    # G
        assert 45 <= center[2] <= 60     # B

    def test_missing_fvsr_clip_skipped(self, tmp_path):
        pr = _make_primary()
        key = fo.write_primary_clip(tmp_path, pr)
        _write_manifest(tmp_path, pr, key)
        # no *_fvsr.npz written
        frame_to_tracks, loads_at = fo._plan_reblend(tmp_path)
        assert frame_to_tracks == {}
        assert loads_at == {}

    def test_empty_keep_window_contributes_nothing(self, tmp_path):
        pr = _make_primary(frame_count=3, keep_start=2, keep_end=2)  # ke <= ks
        key = fo.write_primary_clip(tmp_path, pr)
        _write_manifest(tmp_path, pr, key)
        _write_fvsr(tmp_path, key, frame_count=3)
        frame_to_tracks, loads_at = fo._plan_reblend(tmp_path)
        assert frame_to_tracks == {}
        assert loads_at == {}

    def test_streaming_loop_two_clips_load_blend_free(self, tmp_path):
        # Two non-overlapping clips; simulate the Phase 3 streaming loop and verify
        # each frame blends the right clip AND clips are freed after their last frame.
        device = torch.device("cpu")
        pr_a = _make_primary(track_id=1, start_frame=0, frame_count=2)
        pr_b = _make_primary(track_id=2, start_frame=2, frame_count=2)
        key_a = fo.write_primary_clip(tmp_path, pr_a)
        key_b = fo.write_primary_clip(tmp_path, pr_b)
        # manifest with both clips
        manifest = {
            "version": fo.BUNDLE_VERSION, "input": "x.mp4", "fps": 30.0,
            "frame_count": 4, "total_frames": 4,
            "clips": [
                {"key": key_a, "track_id": 1, "start_frame": 0, "frame_count": 2, "keep_start": 0, "keep_end": 2},
                {"key": key_b, "track_id": 2, "start_frame": 2, "frame_count": 2, "keep_start": 0, "keep_end": 2},
            ],
        }
        fo.manifest_path(tmp_path).write_text(json.dumps(manifest))
        _write_fvsr(tmp_path, key_a, frame_count=2, rgb=(200, 0, 0))     # clip A -> red
        _write_fvsr(tmp_path, key_b, frame_count=2, rgb=(0, 0, 200))     # clip B -> blue

        frame_to_tracks, loads_at = fo._plan_reblend(tmp_path)
        assert set(frame_to_tracks) == {0, 1, 2, 3}
        assert loads_at == {0: [key_a], 2: [key_b]}

        bb = BlendBuffer(device=device)
        for f, tracks in frame_to_tracks.items():
            bb.register_frame(f, tracks)

        centers = []
        for frame_idx in range(4):
            for key in loads_at.get(frame_idx, ()):
                sr = fo._load_clip_sr(tmp_path, key, device)
                assert sr is not None
                bb.add_result(sr)
            blended = bb.blend_frame(frame_idx, torch.zeros(3, 256, 256, dtype=torch.uint8))
            centers.append(int(blended[0, 128, 128]) > int(blended[2, 128, 128]))  # True if red>blue

        assert centers == [True, True, False, False]  # frames 0,1 red; 2,3 blue
        # both clips freed after their last kept frame
        assert bb._results == {}

    def test_plan_from_manifest_matches_geom_kept_range(self, tmp_path):
        # The manifest-only plan must agree with the geom-derived kept range.
        pr = _make_primary(start_frame=10, frame_count=7, keep_start=2, keep_end=6)
        key = fo.write_primary_clip(tmp_path, pr)
        _write_manifest(tmp_path, pr, key)
        _write_fvsr(tmp_path, key, frame_count=7)
        frame_to_tracks, _ = fo._plan_reblend(tmp_path)
        sr = fo._load_clip_sr(tmp_path, key, torch.device("cpu"))
        geom_start = sr.start_frame + sr.clip_keep_offset
        geom_frames = set(range(geom_start, geom_start + sr.keep_end))
        assert set(frame_to_tracks) == geom_frames == {12, 13, 14, 15}


# ---------------------------------------------------------------------------
# argv rewrite helpers
# ---------------------------------------------------------------------------

class TestArgvRewrite:
    def test_swaps_secondary_output_framegen(self):
        argv = ["--input", "in.mp4", "--output", "out.mkv",
                "--secondary-restoration", "flashvsr", "--flashvsr-repo", "/r",
                "--frame-gen", "2x"]
        r = fo._rewrite_argv_for_dump(argv, "/tmp/tw.mkv")
        assert r[r.index("--secondary-restoration") + 1] == "none"
        assert r[r.index("--output") + 1] == "/tmp/tw.mkv"
        assert r[r.index("--frame-gen") + 1] == "none"
        assert "--flashvsr-repo" in r and r[r.index("--flashvsr-repo") + 1] == "/r"

    def test_equals_form(self):
        argv = ["--secondary-restoration=flashvsr", "--output=out.mkv", "--frame-gen=4x"]
        r = fo._rewrite_argv_for_dump(argv, "/tmp/t.mkv")
        assert "--secondary-restoration=none" in r
        assert "--output=/tmp/t.mkv" in r
        assert "--frame-gen=none" in r

    def test_cap_max_clip_size(self):
        assert fo._cap_max_clip_size(["--max-clip-size", "90"], 32) == ["--max-clip-size", "32"]
        assert fo._cap_max_clip_size(["--max-clip-size", "20"], 32) == ["--max-clip-size", "20"]
        assert fo._cap_max_clip_size(["--max-clip-size=90"], 32) == ["--max-clip-size=32"]
        assert fo._cap_max_clip_size(["--input", "x"], 32) == ["--input", "x", "--max-clip-size", "32"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def _args(self, tmp_path, **overrides):
        repo = tmp_path / "repo"
        repo.mkdir(exist_ok=True)
        # default_flashvsr_python resolves per-platform; create both layouts.
        (repo / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
        (repo / ".venv" / "bin" / "python").touch()
        (repo / ".venv" / "Scripts").mkdir(parents=True, exist_ok=True)
        (repo / ".venv" / "Scripts" / "python.exe").touch()
        (repo / "models" / "FlashVSR-v1.1").mkdir(parents=True, exist_ok=True)
        inp = tmp_path / "in.mp4"
        inp.touch()
        ns = MagicMock()
        ns.flashvsr_repo = str(repo)
        ns.flashvsr_python = ""
        ns.flashvsr_model_dir = ""
        ns.input = str(inp)
        ns.output = str(tmp_path / "out.mkv")
        ns.stream = False
        ns.frame_gen = "none"
        ns.retarget_high_fps = False
        ns.segments = ""
        ns.vr_mode = "off"
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    def test_valid_args(self, tmp_path):
        ns = self._args(tmp_path)
        repo, py, model, inp = fo._validate_flashvsr_args(ns)
        assert repo.is_dir() and py.exists() and model.is_dir() and inp.exists()

    def test_missing_repo(self, tmp_path):
        ns = self._args(tmp_path, flashvsr_repo="")
        with pytest.raises(ValueError, match="flashvsr-repo is required"):
            fo._validate_flashvsr_args(ns)

    def test_repo_not_found(self, tmp_path):
        ns = self._args(tmp_path, flashvsr_repo=str(tmp_path / "nope"))
        with pytest.raises(FileNotFoundError):
            fo._validate_flashvsr_args(ns)

    def test_stream_rejected(self, tmp_path):
        ns = self._args(tmp_path, stream=True)
        with pytest.raises(ValueError, match="file-output only"):
            fo._validate_flashvsr_args(ns)

    def test_frame_gen_rejected(self, tmp_path):
        ns = self._args(tmp_path, frame_gen="2x")
        with pytest.raises(ValueError, match="frame-gen"):
            fo._validate_flashvsr_args(ns)

    def test_folder_input_rejected(self, tmp_path):
        d = tmp_path / "folder"
        d.mkdir()
        ns = self._args(tmp_path, input=str(d))
        with pytest.raises(ValueError, match="folder input"):
            fo._validate_flashvsr_args(ns)

    def test_image_input_rejected(self, tmp_path):
        img = tmp_path / "pic.png"
        img.touch()
        ns = self._args(tmp_path, input=str(img))
        with pytest.raises(ValueError, match="video-only"):
            fo._validate_flashvsr_args(ns)

    def test_retarget_high_fps_rejected(self, tmp_path):
        ns = self._args(tmp_path, retarget_high_fps=True)
        with pytest.raises(ValueError, match="retarget-high-fps"):
            fo._validate_flashvsr_args(ns)

    def test_segments_rejected(self, tmp_path):
        ns = self._args(tmp_path, segments="10-25")
        with pytest.raises(ValueError, match="segments"):
            fo._validate_flashvsr_args(ns)

    def test_vr_mode_sbs_rejected(self, tmp_path):
        ns = self._args(tmp_path, vr_mode="sbs")
        with pytest.raises(ValueError, match="VR processing"):
            fo._validate_flashvsr_args(ns)

    def test_vr_mode_sbs_fisheye_rejected(self, tmp_path):
        ns = self._args(tmp_path, vr_mode="sbs-fisheye")
        with pytest.raises(ValueError, match="VR processing"):
            fo._validate_flashvsr_args(ns)

    def test_vr_mode_auto_rejected_when_vr_detected(self, tmp_path):
        from jasna.vr180 import VrModeResolution

        ns = self._args(tmp_path, vr_mode="auto")
        with (
            patch("jasna.media.get_video_meta_data", return_value=MagicMock()),
            patch(
                "jasna.vr180.resolve_vr_mode",
                return_value=VrModeResolution("auto", "sbs", "2:1 high-res", 2.0, "fisheye190"),
            ),
        ):
            with pytest.raises(ValueError, match="detected VR content"):
                fo._validate_flashvsr_args(ns)

    def test_vr_mode_auto_passes_when_flat(self, tmp_path):
        from jasna.vr180 import VrModeResolution

        ns = self._args(tmp_path, vr_mode="auto")
        with (
            patch("jasna.media.get_video_meta_data", return_value=MagicMock()),
            patch(
                "jasna.vr180.resolve_vr_mode",
                return_value=VrModeResolution("auto", "off", "not VR", 1.78, "off"),
            ),
        ):
            repo, py, model, inp = fo._validate_flashvsr_args(ns)
        assert inp.exists()


# ---------------------------------------------------------------------------
# Orchestrator: phase ordering
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def test_runs_three_phases_in_order(self, tmp_path):
        args = self._make_args(tmp_path)
        calls = []

        def fake_run(cmd, env=None):
            calls.append(cmd)
            return MagicMock(returncode=0)

        argv = ["--input", args.input, "--output", args.output,
                "--secondary-restoration", "flashvsr", "--flashvsr-repo", args.flashvsr_repo]
        with (
            patch("jasna.restorer.flashvsr_offline.subprocess.run", side_effect=fake_run),
            patch("jasna.restorer.flashvsr_offline._preflight_bundle_disk"),
            patch("jasna.restorer.flashvsr_offline._gate_phase2_disk"),
            patch.object(sys, "argv", ["jasna", *argv]),
        ):
            fo.run_flashvsr_offline(args)

        assert len(calls) == 3
        # Phase 1: jasna --flashvsr-phase dump
        assert calls[0][1:5] == ["-m", "jasna", "--flashvsr-phase", "dump"]
        # Phase 2: flashvsr python + driver
        assert calls[1][0] == args.flashvsr_python
        assert calls[1][1].endswith("flashvsr_phase2_driver.py")
        # Phase 3: jasna --flashvsr-phase reblend
        assert calls[2][1:5] == ["-m", "jasna", "--flashvsr-phase", "reblend"]

    def test_phase_failure_raises(self, tmp_path):
        args = self._make_args(tmp_path)

        def fake_run(cmd, env=None):
            return MagicMock(returncode=1)

        with patch("jasna.restorer.flashvsr_offline.subprocess.run", side_effect=fake_run):
            with patch.object(sys, "argv", ["jasna", "--input", args.input, "--output", args.output]):
                with pytest.raises(RuntimeError, match="Phase 1"):
                    fo.run_flashvsr_offline(args)

    def test_dump_json_has_rewritten_argv(self, tmp_path):
        args = self._make_args(tmp_path)
        captured = {}

        def fake_run(cmd, env=None):
            # Phase 1 writes phase_dump.json before invoking; read it back.
            if "dump" in cmd:
                jp = Path(cmd[-1])
                captured["dump"] = json.loads(jp.read_text())
            return MagicMock(returncode=0)

        argv = ["--input", args.input, "--output", args.output,
                "--secondary-restoration", "flashvsr", "--flashvsr-repo", args.flashvsr_repo]
        with (
            patch("jasna.restorer.flashvsr_offline.subprocess.run", side_effect=fake_run),
            patch("jasna.restorer.flashvsr_offline._preflight_bundle_disk"),
            patch("jasna.restorer.flashvsr_offline._gate_phase2_disk"),
            patch.object(sys, "argv", ["jasna", *argv]),
        ):
            fo.run_flashvsr_offline(args)

        dump = captured["dump"]
        assert dump["argv"][dump["argv"].index("--secondary-restoration") + 1] == "none"
        # max-clip-size capped (inserted since not present)
        assert "--max-clip-size" in dump["argv"]

    def _make_args(self, tmp_path):
        repo = tmp_path / "repo"
        # default_flashvsr_python resolves per-platform; create both layouts.
        (repo / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
        (repo / ".venv" / "bin" / "python").touch()
        (repo / ".venv" / "Scripts").mkdir(parents=True, exist_ok=True)
        (repo / ".venv" / "Scripts" / "python.exe").touch()
        (repo / "models" / "FlashVSR-v1.1").mkdir(parents=True, exist_ok=True)
        inp = tmp_path / "in.mp4"
        inp.touch()
        bundle = tmp_path / "bundle"
        args = MagicMock()
        args.flashvsr_repo = str(repo)
        args.flashvsr_python = str(repo / ".venv" / "bin" / "python")
        args.flashvsr_model_dir = ""
        args.flashvsr_version = "11"
        args.flashvsr_dtype = "bf16"
        args.flashvsr_max_clip_frames = 32
        args.flashvsr_unload_dit = True
        args.flashvsr_tiled_vae = True
        args.flashvsr_bundle_dir = str(bundle)
        args.flashvsr_keep_bundle = True
        args.input = str(inp)
        args.output = str(tmp_path / "out.mkv")
        args.device = "cuda:0"
        args.codec = "hevc"
        args.bit_depth = "auto"
        args.encoder_settings = ""
        args.working_directory = ""
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
        return args


# ---------------------------------------------------------------------------
# Disk-space safeguards
# ---------------------------------------------------------------------------

class TestDiskSafeguards:
    def _manifest(self, tmp_path, frame_counts, write_fvsr_for=()):
        clips = []
        for i, fc in enumerate(frame_counts):
            key = f"clip_{i}_0"
            clips.append({"key": key, "track_id": i, "start_frame": 0,
                          "frame_count": fc, "keep_start": 0, "keep_end": fc})
            if key in write_fvsr_for:
                _write_fvsr(tmp_path, key, frame_count=fc, size=8)
        fo.manifest_path(tmp_path).write_text(json.dumps({
            "version": fo.BUNDLE_VERSION, "input": "x.mp4", "fps": 30.0,
            "frame_count": sum(frame_counts), "total_frames": sum(frame_counts), "clips": clips,
        }))

    def _free(self, gib):
        m = MagicMock()
        m.free = int(gib * fo._GIB)
        return m

    def test_gate_raises_when_insufficient(self, tmp_path):
        # 2 clips x 100 frames x 3 MiB = 600 MiB needed
        self._manifest(tmp_path, [100, 100])
        with patch("jasna.restorer.flashvsr_offline.shutil.disk_usage", return_value=self._free(0.3)):
            with pytest.raises(RuntimeError, match="Not enough disk for FlashVSR Phase 2"):
                fo._gate_phase2_disk(tmp_path)

    def test_gate_passes_when_ample(self, tmp_path):
        self._manifest(tmp_path, [100, 100])
        with patch("jasna.restorer.flashvsr_offline.shutil.disk_usage", return_value=self._free(50)):
            fo._gate_phase2_disk(tmp_path)  # no raise

    def test_gate_counts_only_missing_fvsr(self, tmp_path):
        # clip 0 already has fvsr -> only clip 1 (100 frames = 300 MiB) counts
        self._manifest(tmp_path, [100, 100], write_fvsr_for={"clip_0_0"})
        needed_one_clip_gib = 100 * fo._FVSR_BYTES_PER_FRAME / fo._GIB  # ~0.29 GiB
        # free just above one clip but far below two -> passes because clip 0 is done
        with patch("jasna.restorer.flashvsr_offline.shutil.disk_usage",
                   return_value=self._free(needed_one_clip_gib * 1.1)):
            fo._gate_phase2_disk(tmp_path)  # no raise

    def _meta(self, num_frames):
        m = MagicMock()
        m.num_frames = num_frames
        return m

    def test_preflight_warns_on_tmpfs(self, tmp_path, capsys):
        with patch("jasna.media.get_video_meta_data", return_value=self._meta(1000)):
            with patch("jasna.restorer.flashvsr_offline._fstype", return_value="tmpfs"):
                with patch("jasna.restorer.flashvsr_offline.shutil.disk_usage", return_value=self._free(30)):
                    fo._preflight_bundle_disk(tmp_path, tmp_path / "in.mp4")
        assert "tmpfs" in capsys.readouterr().out

    def test_preflight_quiet_on_real_disk(self, tmp_path, capsys):
        with patch("jasna.media.get_video_meta_data", return_value=self._meta(1000)):
            with patch("jasna.restorer.flashvsr_offline._fstype", return_value="ext4"):
                with patch("jasna.restorer.flashvsr_offline.shutil.disk_usage", return_value=self._free(500)):
                    fo._preflight_bundle_disk(tmp_path, tmp_path / "in.mp4")
        out = capsys.readouterr().out
        assert "tmpfs" not in out


# ---------------------------------------------------------------------------
# Subprocess dispatch
# ---------------------------------------------------------------------------

class TestPhaseDispatch:
    def test_dump_dispatch(self, tmp_path):
        jp = tmp_path / "cfg.json"
        jp.write_text(json.dumps({"bundle_dir": str(tmp_path), "argv": [], "input": "x"}))
        with patch("jasna.restorer.flashvsr_offline._run_phase_dump") as m:
            fo.run_phase_subprocess("dump", str(jp))
        m.assert_called_once()

    def test_reblend_dispatch(self, tmp_path):
        jp = tmp_path / "cfg.json"
        jp.write_text(json.dumps({"bundle_dir": str(tmp_path)}))
        with patch("jasna.restorer.flashvsr_offline._run_phase_reblend") as m:
            fo.run_phase_subprocess("reblend", str(jp))
        m.assert_called_once()

    def test_unknown_phase(self, tmp_path):
        jp = tmp_path / "cfg.json"
        jp.write_text(json.dumps({}))
        with pytest.raises(ValueError, match="unknown --flashvsr-phase"):
            fo.run_phase_subprocess("bogus", str(jp))


# ---------------------------------------------------------------------------
# Phase 2 driver (importable without a GPU / FlashVSR checkout)
# ---------------------------------------------------------------------------

class TestPhase2DriverArgs:
    def test_driver_arg_parsing(self):
        from jasna.restorer import flashvsr_phase2_driver as drv
        argv = ["prog", "--bundle-dir", "/b", "--repo", "/r", "--model-dir", "/m",
                "--version", "11", "--dtype", "bf16", "--device", "cuda:0",
                "--unload-dit", "--tiled-vae"]
        with patch.object(sys, "argv", argv):
            a = drv._parse_args()
        assert a.bundle_dir == "/b" and a.repo == "/r" and a.model_dir == "/m"
        assert a.version == "11" and a.dtype == "bf16"
        assert a.unload_dit is True and a.tiled_vae is True
