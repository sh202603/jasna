# torchcodec backend (experimental, with vali / PyNvVideoCodec fallback)

> **⚠️ Semantics changed with the v0.8.0 rebase (2026-07-18)**: the native backend is now PyAV (NVDEC/NVENC), and torchcodec **encode** is no longer selected by `auto` (native always outputs 10-bit HEVC/AV1, so output parity with the 8-bit-only torchcodec encoder no longer holds). It runs only when forced (`--encode-backend torchcodec`), and only for 8-bit sources. `docs/CHANGES_vs_upstream_en.md` §7 is authoritative for the current semantics; the rest of this document (selection layer, NVENC mapping, fallback machinery) still applies.


Design for the torchcodec-based decode/encode path selectable via `--video-backend`.

The existing native path (`python_vali` decoder + `PyNvVideoCodec` encoder) is kept
as-is. torchcodec is used only where it applies and falls back to native otherwise.
The default is `native`, which is byte-/perf-identical to before. torchcodec is an
opt-in, experimental feature.

**Runtime environment**: the same GPU-only stack as jasna itself (NVIDIA GPU, CUDA 13.x,
torch 2.12.0+cu130, an ffmpeg 8 shared build on PATH) plus the optional dependency
`torchcodec>=0.15.0` (cu130 build). The numbers in this document were measured on
Windows 11 / Python 3.13.9 / torch 2.12.0+cu130 / CUDA 13.0 / RTX 5080 / ffmpeg 8.1 full-shared
(Appendix A is the Linux side of the same dual-booted RTX 5080; the GPU is shared).

## Background and goal

jasna depends on two vendored, build-heavy native libraries: `python_vali` for
decode and `PyNvVideoCodec` for encode. The goal is to move toward
`torchcodec>=0.15.0`, a single official PyTorch wheel.

The only case that genuinely requires native is **10-bit output** (torchcodec's GPU
encode is 8-bit nv12 only), plus a few unmappable NVENC settings, streaming, and
frame generation. AV1 works via `av1_nvenc`, color is applied by the existing remux,
and most NVENC settings map through `extra_options`. Because native is still needed
for 10-bit, this is not a full replacement: a torchcodec path is **added** and
anything it cannot satisfy **falls back to native**.

## torchcodec 0.15.0 capabilities

0.15.0 is a maintenance release over 0.14.0 (premature-EOF decode fixes, faster
forward seeks, macOS free-threaded wheels); the public encoder/decoder API and
the capabilities below are unchanged. The table was re-verified on 0.15.0 on
real hardware (RTX 5080: NVDEC decode, HEVC/AV1 encode with jasna's full NVENC
option set, rejected options still rejected).

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

- Incremental encoding via `torchcodec.encoders.Encoder`. Codec maps hevc→`hevc_nvenc`,
  av1→`av1_nvenc`.
- Encoding runs on a dedicated worker thread, like the native encoder (rationale in
  Appendix A). `__enter__` starts the worker, which does `Encoder()` →
  `add_video(height, width, frame_rate, device='cuda', codec=...)` → `open_file(temp mkv)`.
  `encode(frame, pts)` runs on the caller (BlendEncode) thread and only records a
  `torch.cuda.Event` on the current stream and pushes `(frame, event)` onto a bounded queue.
  The worker, in FIFO order, does `event.synchronize()` (torchcodec manages its own internal
  CUDA stream and cannot take an injected one, so a host-side wait is used), applies the LUT,
  and calls `add_frames`; `__exit__` sends a stop sentinel and the worker calls `close()`. The
  whole `Encoder` lifecycle lives on the worker to avoid cross-thread use of its internal
  stream. torchcodec is CFR and ignores pts, so there is no pts-reorder buffer: a single FIFO
  preserves frame order.
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
torchcodec = ["torchcodec>=0.15.0"]   # 0.13.0 is known-buggy
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
uv pip install "torchcodec>=0.15.0" --no-deps --index-url https://download.pytorch.org/whl/cu130
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

For a real 1080p clip (31524 frames, rfdetr, with restoration, threaded encode), native vs torchcodec
(environment: see the intro; RTX 5080 / Windows):

| Measurement | native | torchcodec | Note |
|---|---|---|---|
| Decode-only | 2463 fps | 2164 fps | native ~12% faster (content-dependent; torchcodec can be faster on synthetic clips) |
| Encode-only (hevc 8-bit, incl. mux) | 147 fps | 198 fps | torchcodec ~1.35x (native also does mkvmerge explicit timecodes, etc.) |
| Full-length total (with restoration) | ~265 s | ~258 s | on par (within run-to-run noise; lada-yolo shows the same: native 208 s / torchcodec 201 s) |
| Output bitrate (cq=25-equivalent settings) | ~5.93 Mbps | ~6.12 Mbps | torchcodec ~+3.3% (different encoder, so the rate point differs at the same nominal settings) |
| Output SSIM (native vs torchcodec, all 31524 frames) | n/a | All 0.954 / Y 0.937 | chroma U/V ≈ 0.99; see "Output equivalence" below |

Decode and encode throughput go opposite ways, but each is only a few % of the total, so full-length
wall-clock is backend-independent (restoration and detection dominate). On a real 1080p clip with
mosaics, detect → track → restore → NVENC → remux all work correctly, mosaics are restored under both
backends, and frame counts match. The output is, however, not bit-equivalent to native: SSIM is
All 0.954 / Y 0.937 (next section).

### Output equivalence and pipeline determinism

The native pipeline is deterministic: running the same clip twice yields a byte-identical video
bitstream (matching `ffmpeg -map 0:v -c copy -f md5` packet MD5, SSIM 1.0, identical file size). The
whole-file MKV MD5 changes even for identical video because mkvmerge writes a random segment UID, so
equivalence must be judged by packet MD5 or SSIM, not the file hash.

Since native is deterministic, the native-vs-torchcodec SSIM of 0.954 (All) / 0.937 (Y) is a
torchcodec-specific difference, **not pipeline nondeterminism**. It is detector-independent
(lada-yolo 0.9543 / rfdetr 0.9542) and roughly uniform across the frame, so the dominant cause is the
**decode pixel difference** (vali's explicit BT conversion + dither vs torchcodec's internal
conversion — see "Color"; the pipeline blends non-mosaic regions straight from the decoded source, so
the difference covers the whole frame), with the encode difference (+3.3% bitrate) secondary. No
visible breakage was observed (chroma 0.99), but the output is not bit-equivalent and is one step
lower in fidelity; acceptability is a per-use-case call.

> An earlier version listed this SSIM as 0.9904; it did not reproduce under controlled
> re-measurement (all frames, native verified deterministic) on the same RTX 5080 and clip, and was
> corrected to 0.954.

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
- **Encode threading decision (adopted)**: torchcodec encoding runs on a dedicated worker thread,
  like the native encoder. Synchronous encoding lets the encode wall-time surface on the
  blend-encode stage under a heavy detector (rfdetr), making it a co-bottleneck that loses to
  native (Appendix A). Threading brings rfdetr back to parity with native (192 s). The trade-off
  is that a light detector (lada-yolo) regresses from 100 s to 114 s, though it still beats native
  (124 s). Keeping torchcodec at or above native for both detectors was prioritized, so threaded is
  the default. Output is unchanged before/after threading (SSIM vs native stays 0.954).

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

## Appendix A: how the winner depends on the detector

Comparing native vs torchcodec with a light detector (lada-yolo) and a heavy one (rfdetr).
Measured on Linux / RTX 5080 / no secondary / 1080p, 31524 frames (the main table above is
Windows / rfdetr, so both OS and detector differ). The backend winner flips with the detector.

```
                        lada-yolo               rfdetr
                     native  torchcodec     native  torchcodec
total (wall-clock)    124s    101s           192s    197s
decode-detect total   123.9s  100.8s         191.8s  196.6s
blend-encode busy     70.1s   101.6s         150.5s  197.3s
  of which write      0.6s    60.2s          1.0s    112.6s
```

With lada-yolo torchcodec is ~18% faster; with rfdetr native is ~3% faster. For rfdetr,
`--video-backend auto` picks torchcodec (8-bit HEVC is eligible), so auto is also 198 s, below native.

### Why the per-stage split is not comparable across backends

Timing is accumulated as CPU wall-clock per category, and since GPU work is async, the split
depends on what is synchronized at each category boundary. Native synchronizes only its decode
stream at the end of a batch; torchcodec synchronizes the whole device per batch
(`torch.cuda.synchronize(device)`). So the decode-vs-detect split is not comparable across
backends; only the stage totals and the wall-clock total are.

### Why torchcodec wins with lada-yolo

With a light detector, decode-detect is the bottleneck. The decode-only bench has native ~12%
faster (2463 fps vs 2164 fps), yet the combined decode-detect total is faster on torchcodec
(123.9 s vs 100.8 s). Native winning alone but losing combined is the signature of native decode
contending with detection for SMs: vali does a two-pass color conversion after decode, and those
SM kernels compete with the detection TensorRT kernels. torchcodec's conversion is lighter, so
more SMs go to detection and the bottleneck stage shrinks by 23 s.

### Why it flips with rfdetr

With a heavy detector, that SM relief no longer applies. rfdetr nearly saturates the GPU, so
freeing SMs in decode does not help (detection refills them), and the combined decode-detect total
is even slightly higher on torchcodec (191.8 s vs 196.6 s) due to torchcodec's per-batch contiguity
copy and full-device sync, which are not hidden under saturation.

The decisive factor is encoding. torchcodec encode (before threading) ran synchronously on the
blend-encode thread and uses SMs for pre-processing. The same encode takes 60.2 s with a free GPU
(lada-yolo) but balloons to 112.6 s under saturation (rfdetr), pushing blend-encode to 197.3 s — a
co-bottleneck with decode-detect — so the total exceeds native.

### Encode threading and its effect (implemented, measured)

Following the analysis, torchcodec encoding was moved to a dedicated worker thread like native
(`encode()` records an event and enqueues; the worker `event.synchronize()`s then `add_frames`).
Output is unchanged: the encoded result is the same before/after threading, and SSIM vs native stays 0.954.

The effect split by detector (RTX 5080 / Linux, same-session A/B):

```
                  native   torchcodec inline   torchcodec threaded
lada-yolo total    124s      100s               114s
rfdetr    total    192s      197s               192s
```

For rfdetr it worked as intended: blend-encode write dropped from 112 s to ~7 s, the stage fell
below decode-detect, and the total reached 192 s (on par with native). For lada-yolo it regressed
from 100 s to 114 s: decode-detect rose from 100.3 s to 114.3 s (both decode and detect up ~7 s),
because when the detector is light and the GPU has headroom, the worker's concurrent encode
contends with detection for SMs and raises the bottleneck stage. Inline encoding had paced itself
to the blend thread's cadence, which happened to pack more efficiently.

It is a trade-off, but threaded keeps torchcodec at or above native for both detectors (parity on
rfdetr, 114 s < 124 s on lada-yolo), so threaded is the default for torchcodec.

### OS difference (same RTX 5080, dual-boot)

The main "Performance and validation" section (Windows) and this appendix (Linux) were measured on
**the same RTX 5080, dual-booted**, so the cross-OS difference is not hardware but the OS / software
stack.

Full-length total for the same clip and same code (threaded):

```
              Linux     Windows (same 5080)
lada-yolo     ~124s     ~208s   (+68%)
rfdetr        ~192s     ~265s   (+38%)
```

Windows is 34–68% slower, and the gap is concentrated in the **launch-bound TRT stages (detection and
BasicVSR++ restoration)**; the fixed-function decode/encode stages are unchanged. The lighter,
more-launch-heavy lada-yolo is hurt more. HAGS is already on (that lever is exhausted), so the
remainder points to WDDM submission overhead and the per-OS TRT engine builds (`.win.engine`).
Remedies: CUDA-graphing detection and restoration, and unifying the TRT version across OSes.

Note that on Windows (threaded), native vs torchcodec is 208 s vs 201 s for lada-yolo and 265 s vs
258 s for rfdetr — **torchcodec ≤ native for both detectors** (within run-to-run noise). So the
"a heavy detector makes auto pick torchcodec and pessimize" concern does not reproduce: after
threading it is parity even on Linux (rfdetr 192 = 192) and a slight win on Windows. It was specific
to the inline era (the first table in this appendix).

## Appendix B: why the advantage depends on the GPU

Whether torchcodec wins depends on the GPU's engine layout. The mechanism is not "torchcodec has a
stronger decoder": both backends use the same NVDEC hardware, so there is no difference at that layer.

### Two decoders and the NVDEC count

The pipeline runs two decoders concurrently: the decode-detect read and the blend-encode re-read of
original frames. The RTX 5080 has two NVDEC engines, so the two decoders land on separate engines and
do not contend. GeForce RTX 50 engine counts (SM ≈ CUDA cores / 128):

```
GPU            NVENC  NVDEC   SM
RTX 5090         3      2     170
RTX 5080 (test)  2      2      84
RTX 5070 Ti      2      1      70
RTX 5070         1      1      48
RTX 5060 Ti      1      1      36
RTX 5060         1      1      30
```

The RTX 5060 and 5060 Ti have only one NVDEC, so the two decoders share a single engine and contend.
This constraint applies to both native and torchcodec.

### Saturation causes the flip (implication of Appendix A)

rfdetr saturating the RTX 5080's SMs approximates running a light detector on a weaker GPU. As shown
in Appendix A, under saturation torchcodec's SM relief stops working and the (previously synchronous)
encode surfaces and loses to native. A weaker GPU saturates at the same load sooner, so on an RTX 5060
even a light detector may behave like rfdetr-on-5080.

### Prediction for the RTX 5060 class

Whether torchcodec wins on a weaker GPU is decided by the balance between spare SMs and encode
exposure. The SM relief only helps when detection is not saturating the GPU (measured in Appendix A).
The RTX 5060 has few SMs and saturates early, so the gain is unlikely to appear. On top of that, a
single NVDEC makes the two decoders contend, and the encode is starved of SMs. These all stack against
torchcodec. So "a weak decoder reduces the benefit" is directionally right, but the cause is not the
decoder's raw speed: it is one NVDEC for two decoders plus SM scarcity throwing off the detection/encode
balance. The final winner depends on that balance and must be measured.

### Confirming the prediction by measurement

1. Run the same conditions on RTX 5060-class hardware and compare per-stage times, especially whether
   torchcodec's encode time exceeds the decode-detect stage.
2. Capture per-engine utilization (`nvidia-smi dmon`, etc.) to tell whether NVDEC or SM saturates first.
3. See how encode threading (implemented per Appendix A) behaves on a weak GPU. On the RTX 5080 it
   improved rfdetr and regressed lada-yolo, so on the SM-scarce, early-saturating 5060 class even a
   light detector may land on the regression side.

Sources:
- [Video Encode and Decode GPU Support Matrix (NVIDIA Developer)](https://developer.nvidia.com/video-encode-decode-gpu-support-matrix)
- [NVIDIA RTX Blackwell GPU Architecture (PDF)](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
