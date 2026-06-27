# torchcodec backend (experimental, with vali / PyNvVideoCodec fallback)

Design for the torchcodec-based decode/encode path selectable via `--video-backend`.

The existing native path (`python_vali` decoder + `PyNvVideoCodec` encoder) is kept
as-is. torchcodec is used only where it applies and falls back to native otherwise.
The default is `native`, which is byte-/perf-identical to before. torchcodec is an
opt-in, experimental feature.

**Runtime environment**: the same GPU-only stack as jasna itself (NVIDIA GPU, CUDA 13.x,
torch 2.12.0+cu130, an ffmpeg 8 shared build on PATH) plus the optional dependency
`torchcodec>=0.14.0` (cu130 build). The numbers in this document were measured on
Windows 11 / Python 3.13.9 / torch 2.12.0+cu130 / CUDA 13.0 / RTX 5080 / ffmpeg 8.1 full-shared.

## Background and goal

jasna depends on two vendored, build-heavy native libraries: `python_vali` for
decode and `PyNvVideoCodec` for encode. The goal is to move toward
`torchcodec>=0.14.0`, a single official PyTorch wheel.

The only case that genuinely requires native is **10-bit output** (torchcodec's GPU
encode is 8-bit nv12 only), plus a few unmappable NVENC settings, streaming, and
frame generation. AV1 works via `av1_nvenc`, color is applied by the existing remux,
and most NVENC settings map through `extra_options`. Because native is still needed
for 10-bit, this is not a full replacement: a torchcodec path is **added** and
anything it cannot satisfy **falls back to native**.

## torchcodec 0.14.0 capabilities

| Area | Supported | Detail |
|---|---|---|
| Decode (CUDA/NVDEC) | ✅ | GPU `uint8` RGB NCHW tensors; batch (`get_frames_in_range`), exact seek, `pts_seconds`, BT.601/709 + HDR conversion. Always 8-bit RGB. FFmpeg 4–8. |
| Decode: custom CUDA stream injection | ❌ | Manages its own internal stream; no external-stream injection. |
| Encode (CUDA/NVENC) | ✅ | On CUDA frames, `hevc_nvenc`/`av1_nvenc` (auto-selected if `codec` omitted). Incremental via `torchcodec.encoders.Encoder`. A bare name like `codec='hevc'` resolves to a software encoder (libx265) that rejects CUDA frames, so `*_nvenc` must be used. |
| Encode: 10-bit / P010 | ❌ | GPU encode pixel format is hardcoded to 8-bit `nv12` (`pixel_format` unsupported). The only firm gate. |
| Encode: AV1 | ✅ | `av1_nvenc`. |
| Encode: NVENC settings | ◯ (subset) | Accepted `extra_options` keys (cq/qmin/qmax/gop→g/lookahead→rc-lookahead/temporalaq→temporal-aq/aq→spatial-aq+aq-strength/nonrefp→nonref_p/maxbitrate→maxrate/vbvbufsize→bufsize; preset via the dedicated param) are mapped and applied. rc/tuning_info/bref/vbvinit/initqp/lookahead_level/tflevel are rejected by torchcodec. |
| Encode: color metadata | △ | torchcodec doesn't write color primaries/transfer/matrix, but the existing remux (`remux_with_audio_and_metadata`, HEVC VUI rewrite) applies them to the output stream. |

Decode covers every input jasna needs (8-bit RGB). The only firm encode gate is
10-bit, which falls back to native.

## Architecture

### No regression on the native path (top constraint)

The native path must stay identical in behavior and performance.

- For `backend == native` (and AUTO when torchcodec is ineligible/unavailable) the
  factory returns the existing `NvidiaVideoReader` / `NvidiaVideoEncoder` directly,
  with no wrapper.
- Selection / factory run once at construction; the per-frame hot path (`frames()` /
  `encode()`) gains no branching, delegation, or attribute access.
- Each pipeline change only swaps the construction site for a factory call; the
  decode/encode loop bodies are untouched.

### Selection layer `jasna/media/backend.py`

The single module that knows both backends. No torch/torchcodec/vali/nvc import at
module load (deferred), so the predicates are unit-testable without a GPU or native
libraries.

- `VideoBackend(str, Enum)`: `AUTO` / `NATIVE` / `TORCHCODEC`.
- `torchcodec_available()`: `find_spec` only (cheap, never raises, cached). A package
  present but failing to import is caught at construction; AUTO falls back to native.
- `select_decoder_backend(...)`: torchcodec decode covers all inputs, so the only gate
  is availability.
- `torchcodec_encoder_eligibility(...) -> (bool, reason)`: pure predicate. Eligible
  only when ALL hold:
  1. `stream_mode is False`
  2. `codec` is `hevc` or `av1`
  3. effective bit_depth `== 8` (`bit_depth if not None else (10 if metadata.is_10bit else 8)`)
  4. `--encoder-settings` are all mappable (`encoder_settings_mappable`)
  5. `color_range` is limited (JPEG/full is not supported)
  6. `output_fps_multiplier == 1`
  - Colorspace (BT.601/709/2020) is NOT gated: color is applied by the remux, so
    torchcodec + remux matches native.
- `make_video_reader(...)` / `make_video_encoder(...)`: factories. Native returns the
  existing class directly; AUTO falls back to native if torchcodec construction/init
  fails; forced `torchcodec` re-raises.

### torchcodec decoder `jasna/media/torchcodec_decoder.py`

`TorchcodecVideoReader(file, batch_size, device, metadata)` mirrors `NvidiaVideoReader`
(context manager; `frames(seek_ts=None)` yields `(uint8[B,3,H,W] CUDA, list[int] PTS)`).

- `VideoDecoder(file, device=str(device), dimension_order="NCHW", seek_mode="exact")`.
- Sequential batches via `get_frames_in_range(next, stop)`; yield `fb.data` as-is.
- PTS: `int(round(pts_seconds / float(metadata.time_base)))` (only B floats hit the
  CPU; pixels stay on GPU). Bump +1 on collision for strict monotonicity.
- torchcodec uses its own stream, so `torch.cuda.synchronize(device)` runs before each
  batch yield to hand fully-materialized frames to downstream inference (same cadence
  as native's stream sync).
- Fallback triggers: import failure, `__enter__` failure, CPU-fallback detection (from
  the first batch's `fb.data.device`; `decoder.metadata` has no device attribute).

### torchcodec encoder `jasna/media/torchcodec_encoder.py`

`TorchcodecVideoEncoder(...)` mirrors `NvidiaVideoEncoder`'s constructor + FrameWriter
contract (`__enter__` / `encode(frame, pts)` / `__exit__`). The constructor re-checks
eligibility and raises `TorchcodecEncodeUnsupported(ValueError)` if ineligible.

- Incremental encoding via `torchcodec.encoders.Encoder`. `__enter__` does
  `add_video(height, width, frame_rate, device='cuda', codec=...)` → `open_file(temp mkv)`;
  `encode` applies the LUT then `VideoStream.add_frames(frame.unsqueeze(0))`; `__exit__`
  calls `close()`. Codec maps hevc→`hevc_nvenc`, av1→`av1_nvenc`.
- `__exit__` runs the temp mkv through `remux_with_audio_and_metadata(temp, out, metadata,
  codec)`. Audio, color tags, and the HEVC VUI rewrite are applied by that shared helper,
  so colorspace is identical to native.
- Timing is CFR at `frame_rate` (`video_fps_exact`). Unlike native (raw elementary +
  mkvmerge explicit per-frame timecodes), it carries no arbitrary per-frame PTS, so VFR
  stays on native.
- NVENC settings: `_build_encode_params` builds jasna's default quality (cq=25, preset p5,
  qmin/qmax, temporal-aq, rc-lookahead 32, spatial-aq + aq-strength 8, gop, bf=4 for hevc)
  from the mappable keys and overlays `--encoder-settings`, passing
  `add_video(preset=..., extra_options=...)`, so output quality is close to native.

### Fallback

For AUTO decode, torchcodec's init (NVDEC open, CPU-fallback detection) happens at
`__enter__`, so `make_video_reader` returns a `_FallbackVideoReader` adapter. The adapter
tries torchcodec at `__enter__` and, on failure, constructs and enters native. `frames`
and `__exit__` delegate to whichever reader entered, adding nothing per frame. `native`
returns the existing reader directly.

## CLI

```
--video-backend  {native,auto,torchcodec}            # default: native
--decode-backend {inherit,native,auto,torchcodec}    # default: inherit (= --video-backend)
--encode-backend {inherit,native,auto,torchcodec}    # default: inherit
```

- `native`: always vali + PyNvVideoCodec (current behavior).
- `auto`: use torchcodec where available and eligible, else native.
- `torchcodec`: force torchcodec; requests it cannot satisfy (10-bit / streaming, ...)
  error early.
- `--decode-backend` / `--encode-backend` override each side independently; `inherit`
  follows `--video-backend`.

Added to both `jasna` (`main.py`) and `jasna-framegen` (`framegen_cli.py`). Early
validation: `torchcodec` (decode or encode) + streaming, and `encode-backend torchcodec`
+ `--bit-depth 10`, error. Because frame generation always encodes CFR (native),
`jasna-framegen` downgrades a forced `torchcodec` encode backend to `auto` and uses
torchcodec for decode only.

## Dependencies and install

Optional extra in `pyproject.toml` (native-only installs keep working):

```toml
[project.optional-dependencies]
torchcodec = ["torchcodec>=0.14.0"]   # 0.13.0 is known-buggy
```

Install it with uv from the cu130 wheel index so the build matches torch 2.12 + cu130.
Either install it together with jasna through the extra (same flags as the build guides):

```powershell
uv pip install -e .[dev,torchcodec] `
    --extra-index-url https://download.pytorch.org/whl/cu130 `
    --index-strategy unsafe-best-match `
    --prerelease=allow
```

or add it to an existing environment (`--no-deps` keeps the pinned torch untouched):

```powershell
uv pip install "torchcodec>=0.14.0" --no-deps --index-url https://download.pytorch.org/whl/cu130
```

- torchcodec dlopens FFmpeg shared libs at runtime; jasna's required **ffmpeg 8 shared
  build** (`avcodec-62`/`avformat-62`/`avutil-60`) on PATH satisfies this.
- Windows DLL search: `torchcodec` is whitelisted in `packaging/windows_dll_paths.py`
  (no-op when absent); the wrapper modules `os.add_dll_directory` before importing torchcodec.
- Not added to `__main__.py`'s `_preload_native_libs()` (it's optional).
- Note: Windows + cu130 + Python 3.13 has a history of torchcodec DLL-load failures
  (torchcodec issue #1233). It did not reproduce on the test stack (RTX 5080 / cu130 /
  ffmpeg 8.1); `torchcodec_available()` only checks the spec, so an import failure is caught
  at construction and AUTO falls back to native.

## Performance and validation

For a real 1080p clip (31524 frames, rfdetr, with restoration), native vs torchcodec (environment:
see the intro):

| Measurement | native | torchcodec | Note |
|---|---|---|---|
| Decode-only | 2463 fps | 2164 fps | native ~12% faster (content-dependent; torchcodec can be faster on synthetic clips) |
| Encode-only (hevc 8-bit, incl. mux) | 147 fps | 198 fps | torchcodec ~1.35x (native also does mkvmerge explicit timecodes, etc.) |
| Full-length total (with restoration) | ~284 s | ~275 s | on par |
| Output SSIM (native vs torchcodec) | n/a | 0.9904 | both correctly restored |

Decode and encode throughput go opposite ways, but each is only a few % of the total, so full-length
wall-clock is backend-independent (restoration and detection dominate). On a real 1080p clip with
mosaics, detect → track → restore → NVENC → remux all work correctly; the output is equivalent to
native (SSIM 0.990) and frame counts match.

## Limitations and notes

- **Decoded-frame contiguity (needs `.contiguous()`)**: torchcodec returns NCHW as a non-contiguous
  view of its internal HWC buffer (interleaved C/W strides). The detection TensorRT engine reads its
  input as contiguous NCHW from raw memory, so passing it non-contiguous makes it read garbage and
  **detect no mosaics** (→ restoration skipped). `TorchcodecVideoReader` returns `fb.data.contiguous()`.
  It only surfaces with full batches (no padding copy), so it is easy to miss; an `is_contiguous()`
  assertion in `tests/test_torchcodec_decoder.py` guards against regression.
- **10-bit**: torchcodec GPU encode is 8-bit nv12 only; 10-bit requests use native.
- **VFR**: torchcodec encodes CFR (`video_fps_exact`); it carries no arbitrary per-frame PTS,
  so VFR stays on native.
- **Unmappable NVENC settings**: rc/tuning_info/bref/vbvinit/initqp/lookahead_level/tflevel in
  `--encoder-settings` route to native.
- **Color**: vali does explicit BT.x + range conversion and dithering; torchcodec does its own
  conversion, so decoded pixels are not bit-identical (first-frame mean-abs-diff ≈ 8.5 on a
  synthetic clip). Output color metadata is applied by the remux, identical to native.
- **Decode performance**: native is ~12% faster on real 1080p (content-dependent; decode is not
  the bottleneck).
- **Frame count (does not reach output)**: the native vali reader returns a few extra frames at
  the raw-decode level (50 vs ground-truth 48; 504 vs 500, varies); torchcodec is always exact.
  The pipeline output is correct for both.

## Tests

- `tests/test_video_backend_select.py` (no GPU): decoder selection and the full encoder
  eligibility matrix (codec / bit-depth / settings-mappability / range / frame-gen / streaming).
- `tests/test_video_backend_fallback.py` (no GPU): `_FallbackVideoReader` fallback on `__enter__`
  failure, and factory dispatch.
- `tests/test_torchcodec_decoder.py` (GPU-guarded): output contract, strictly-increasing PTS,
  PTS match + close pixels + correct frame count vs native.
- `tests/test_torchcodec_encoder.py` (GPU-guarded): hevc/av1 round-trips verified with ffprobe,
  settings-mapping construction, rejection of ineligible requests.
- `tests/test_perf_regression.py` (GPU, `-m perf`): wall-clock (with `torch.cuda.synchronize()`)
  native-vs-torchcodec decode throughput.
- `tests/test_main_validation.py`: CLI guards (torchcodec + streaming, encode torchcodec + 10-bit,
  AV1 allowed).
- Existing tests: the decode/encode construction seams moved to the factory, so the mock targets
  in `test_pipeline_threads.py` / `test_pipeline_run*.py` were repointed to `make_video_reader` /
  `make_video_encoder`.
