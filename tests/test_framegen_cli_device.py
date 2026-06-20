"""CLI-level test: --device and --factor are threaded into decode/encode/generator.

GPU-free and native-lib-free: the decoder/encoder submodules (which import the
native vali/PyNvVideoCodec extensions) are replaced with stub modules in
sys.modules, and torch.cuda.device is patched, so the test runs on any box.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import torch


def test_cli_threads_device_and_multiplier(tmp_path):
    input_path = tmp_path / "in.mkv"
    input_path.touch()
    output_path = tmp_path / "out.mkv"

    captured: dict = {}
    device_capture: list = []

    class NoOpDeviceContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def record_device(device):
        device_capture.append(device)
        return NoOpDeviceContext()

    class FakeReader:
        def __init__(self, file, batch_size, device, metadata):
            captured["reader_device"] = device

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def frames(self):
            # One source frame -> source_frames == 1, so main() does not error out.
            yield ["frame"], [0]

    class FakeEncoder:
        def __init__(self, file, device, metadata, **kwargs):
            captured["encoder_device"] = device
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def encode(self, frame, pts):
            captured.setdefault("encoded", []).append((frame, pts))

    def fake_build_generator(backend, *, device, **kwargs):
        captured["gen_backend"] = backend
        captured["gen_device"] = device
        return MagicMock()

    # Stub the native-backed submodules so main()'s `from jasna.media.video_* import`
    # picks them up without importing vali / PyNvVideoCodec.
    decoder_mod = types.ModuleType("jasna.media.video_decoder")
    decoder_mod.NvidiaVideoReader = FakeReader
    encoder_mod = types.ModuleType("jasna.media.video_encoder")
    encoder_mod.NvidiaVideoEncoder = FakeEncoder

    meta = MagicMock()
    meta.num_frames = 1

    with (
        patch("jasna.framegen_cli.check_ascii_install_path", return_value=(True, "/fake")),
        patch("jasna.framegen_cli.check_nvidia_gpu", return_value=(True, "Fake GPU")),
        patch("jasna.framegen_cli.check_gpu_driver_version", return_value=(True, "590.18")),
        patch("jasna.framegen_cli.check_required_executables"),
        patch("jasna.framegen_cli.check_windows_nvidia_sysmem_fallback_policy", return_value=(True, "OK")),
        patch("jasna.media.get_video_meta_data", return_value=meta),
        patch("jasna.framegen.build_frame_generator", side_effect=fake_build_generator),
        patch.dict(sys.modules, {
            "jasna.media.video_decoder": decoder_mod,
            "jasna.media.video_encoder": encoder_mod,
        }),
        patch("torch.cuda.device", side_effect=record_device),
        patch.object(sys, "argv", [
            "jasna-framegen",
            "--input", str(input_path),
            "--output", str(output_path),
            "--device", "cuda:1",
            "--factor", "4x",
            "--no-progress",
        ]),
    ):
        from jasna.framegen_cli import main

        main()

    want = torch.device("cuda:1")
    assert any(d == want for d in device_capture)
    assert captured["reader_device"] == want
    assert captured["encoder_device"] == want
    assert captured["gen_device"] == want
    assert captured["gen_backend"] == "rife"
    # output_fps_multiplier must equal the factor (4x) for correct NVENC fps/GOP.
    assert captured["output_fps_multiplier"] == 4
    assert len(captured.get("encoded", [])) == 1
