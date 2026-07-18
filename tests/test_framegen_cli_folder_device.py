"""CLI-level test: folder batch reuses one generator and names outputs per file.

GPU-free and native-lib-free: the decoder/encoder submodules are replaced with
stub modules in sys.modules and torch.cuda.device is patched, so it runs anywhere.
Confirms the generator is built once / closed once and one encoder is created per
video with the pattern-derived output path.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch


class _NoOpDeviceContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_folder_batch_reuses_one_generator(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "a.mp4").touch()
    (in_dir / "b.mp4").touch()
    out_dir = tmp_path / "out"

    encoder_outputs: list[str] = []
    build_calls: list[str] = []
    gen = MagicMock()

    class FakeReader:
        def __init__(self, file, batch_size, device, metadata):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def frames(self):
            yield ["frame"], [0]

    class FakeEncoder:
        def __init__(self, file, device, metadata, **kwargs):
            encoder_outputs.append(file)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def encode(self, frame, pts):
            pass

    def fake_build_generator(backend, *, device, **kwargs):
        build_calls.append(backend)
        return gen

    decoder_mod = types.ModuleType("jasna.media.video_decoder")
    decoder_mod.NvidiaVideoReader = FakeReader
    encoder_mod = types.ModuleType("jasna.media.video_encoder")
    encoder_mod.NvidiaVideoEncoder = FakeEncoder

    meta = MagicMock()
    meta.num_frames = 1

    with (
        patch("jasna.framegen_cli.check_ascii_install_path", return_value=(True, "/fake")),
        patch("jasna.framegen_cli.check_supported_gpu", return_value=(True, "Fake GPU")),
        patch("jasna.framegen_cli.check_gpu_driver_version", return_value=(True, "590.18")),
        patch("jasna.framegen_cli.check_required_executables"),
        patch("jasna.framegen_cli.check_windows_nvidia_sysmem_fallback_policy", return_value=(True, "OK")),
        patch("jasna.media.get_video_meta_data", return_value=meta),
        patch("jasna.framegen.build_frame_generator", side_effect=fake_build_generator),
        patch.dict(sys.modules, {
            "jasna.media.video_decoder": decoder_mod,
            "jasna.media.video_encoder": encoder_mod,
        }),
        patch("torch.cuda.device", side_effect=lambda d: _NoOpDeviceContext()),
        patch.object(sys, "argv", [
            "jasna-framegen",
            "--input", str(in_dir),
            "--output", str(out_dir),
            "--factor", "2x",
            "--no-progress",
        ]),
    ):
        from jasna.framegen_cli import main

        main()

    # One generator built and closed once, reused across both videos.
    assert build_calls == ["rife"]
    assert gen.close.call_count == 1
    # One encoder per video, named via the default {stem}_out pattern.
    assert sorted(Path(p).name for p in encoder_outputs) == ["a_out.mp4", "b_out.mp4"]
    assert out_dir.is_dir()
