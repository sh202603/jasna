# SwiftVR secondary restoration (`+modi`)

`--secondary-restoration swiftvr-inline` upscales each restored 256px mosaic crop
with [SwiftVR](https://github.com/H-oliday/SwiftVR) (one-step streaming diffusion
VSR on a Wan2.2-TI2V-5B backbone; jasna uses the fork
[`sh202603/SwiftVR`](https://github.com/sh202603/SwiftVR)), recovering texture the
primary BasicVSR++ leaves blurry on large mosaics, close-ups and 4K sources. It
fills the same role as the inline mode of the
[FlashVSR secondary restoration](flashvsr.md), and processes at the model-native
1024px (4x) or at 512px with `--swiftvr-scale 2`. Either way the blend
shrink-composites the crops back onto the frame, so the output resolution is
unchanged. The restored crops are always color-corrected against the primary
output they were built from ("[Color correction](#color-correction)").

The difference from FlashVSR inline is speed and VRAM. SwiftVR runs with FP8 and
torch.compile by default ("[Acceleration](#acceleration---swiftvr-accel)"); on an
RTX 5080 one 90-frame crop clip takes about 2 s at scale 4 and 0.6 s at scale 2,
about 4 to 6 times faster per clip than FlashVSR tiny-long in the same
configuration (2 strips at scale 4, no strips at scale 2), and 2 to 4 times
faster over a whole run ("[VRAM and speed](#vram-and-speed)"). Its VRAM lets it
co-reside with the primary pipeline without strip tiling even at scale 4.

SwiftVR currently has the inline mode only. An offline 3-phase mode (the
counterpart of FlashVSR's `flashvsr`) is planned after the inline verification;
until then, the SeedVR2 primary and 12 GB-class GPUs are served by FlashVSR's
offline mode.

## How it works

`swiftvr-inline` runs SwiftVR as the secondary restoration stage inside jasna's
normal streaming pipeline. The restorer spawns a resident worker
(`jasna/restorer/swiftvr_inline_worker.py`) under the SwiftVR virtualenv's Python
and, per clip, sends the 256px crops over stdin and reads the 256*scale px
restored crops back over stdout (a JSON header plus raw uint8, the same protocol
as the FlashVSR worker). No intermediate files are written.

The worker calls the fork's `SwiftVRPipeline.restore_clip()`. That API takes a
clip held in memory and returns exactly as many frames as it was given
(upstream's `restore_video()` works on files and truncates the frame count to
4k+1, and `StreamSession` drops leading frames, so neither meets the secondary
restorer's contract). SwiftVR restores a clip in fixed causal chunks (28 frames,
then 24 at a time), so its VRAM does not depend on the clip length.

## Requirements

SwiftVR is **not bundled**. You supply the checkout, the checkpoint (~20 GB) and
its own virtualenv, and point jasna at them with `--swiftvr-repo`.

Use the fork [`sh202603/SwiftVR`](https://github.com/sh202603/SwiftVR) (default
branch `modi`). It adds `restore_clip()`, uv packaging, the FP8 DiT,
torch.compile, cuDNN attention and a lower-memory ReAE to upstream; the model
and checkpoint are upstream's. An upstream checkout has no `restore_clip()`, and
jasna checks for it at startup and stops with an explicit error.

### Setting up the SwiftVR checkout (one-time)

```bash
# 1. Clone the fork (default branch modi).
git clone https://github.com/sh202603/SwiftVR.git ~/SwiftVR
cd ~/SwiftVR

# 2. Create .venv and install the dependencies. torch comes from the PyTorch cu132
#    index, as configured in pyproject.toml (Python 3.12+). On Linux the base
#    Python must ship the dev headers (Python.h): Triton, used by the FP8
#    quantization kernels and by torch.compile, builds a C launcher on first use
#    (install python3.X-dev for a system Python; Windows needs nothing, since
#    triton-windows bundles it). Without the headers jasna still runs, but the
#    worker drops the acceleration at startup and falls back to bf16 (~12 GiB).
uv sync

# 3. The checkpoint (~20 GB). The default location is <repo>/checkpoints; pass
#    --swiftvr-model-dir for any other.
uv run hf download H-oliday/SwiftVR --local-dir checkpoints/

# 4. (Recommended) Smoke-test SwiftVR on its own before wiring jasna in, with the
#    FP8 + torch.compile path inline uses.
uv run swiftvr --input some.mp4 --output out.mp4 --checkpoint checkpoints/ \
    --upscale 4 --fp8-dit --torch_compile
```

### Pointing jasna at it

- `--swiftvr-repo <path>` (required): the checkout from above.
- `--swiftvr-python <path>` (default `<repo>/.venv/bin/python`, on Windows
  `<repo>/.venv/Scripts/python.exe`): the Python of the venv from step 2.
- `--swiftvr-model-dir <path>` (default `<repo>/checkpoints`): the checkpoint.

## Usage

```bash
jasna --input in.mp4 --output out.mkv \
      --secondary-restoration swiftvr-inline \
      --swiftvr-repo ~/SwiftVR \
      --log-level info
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--swiftvr-repo` | (required) | Path to the SwiftVR checkout (the fork, with `restore_clip()`). |
| `--swiftvr-python` | `<repo>/.venv/bin/python` | Python of the SwiftVR env (the venv `uv sync` creates). |
| `--swiftvr-model-dir` | `<repo>/checkpoints` | Checkpoint directory (`reae.safetensors`, `prompt_embedding.safetensors`, `transformer/`). |
| `--swiftvr-scale` | `4` | Processing scale. `4` = model-native 1024px, `2` = 512px (faster, lower VRAM). See "[Processing scale](#processing-scale---swiftvr-scale)". |
| `--swiftvr-accel` / `--no-swiftvr-accel` | **on** | FP8 DiT and torch.compile. Needs an RTX 40 series or newer GPU and a working Triton; the worker drops unavailable parts at startup with a warning. See "[Acceleration](#acceleration---swiftvr-accel)". |

There is no counterpart of FlashVSR's `--flashvsr-version`, `--flashvsr-dtype`,
`--flashvsr-tiles` or `--flashvsr-lora`: there is one model, bf16 only (FP8
requires it), no strip tiling is needed, and there is no LoRA. Color correction
has no flag either (always on, see below).

### Processing scale (`--swiftvr-scale`)

SwiftVR is a 4x model: the 256px crops are pre-upscaled bilinearly by the scale
and the DiT processes the result. `4` processes at 1024px as in training; `2`
processes at 512px. At 512px the DiT's token grid is 16x16, one window, so the
window shift has no effect and the attention structure differs from training. It
works, but its quality is checked separately with the same gates as scale 4
("[Verification](#verification)"). The output video resolution is the same for
both.

### Acceleration (`--swiftvr-accel`)

On by default. It uses the fork's two acceleration parts.

- **FP8 DiT:** the DiT blocks' linear layers run as FP8 GEMMs. RTX 40 series or
  newer only (compute capability 8.9+). The DiT weights shrink from 9.4 GiB to
  4.8 GiB and the peak VRAM from about 12 GiB (bf16) to about 8 GiB.
- **torch.compile:** fuses the DiT's elementwise ops. The first warmup compiles
  four graphs (two chunk kinds times with/without the window shift), which adds
  a few seconds to the startup; since the crops are always 256px, nothing
  recompiles afterwards.

In the fork's measurements (RTX 5060 Ti, 640x480 to 1280x960) the two together
raise the GPU throughput from 9.1 to 23.3 fps and lower the peak from 12.0 to
8.0 GiB. The output differs slightly from bf16 (the DiT amplifies the FP8
rounding to about 47 dB against bf16; the fork's visual A/B saw no difference).

Both parts use Triton. Before loading the model, the worker checks the GPU
generation and that Triton works (it runs a small kernel once), drops whatever
is unavailable and reports why in jasna's log. Without FP8 the DiT runs in bf16
(~12 GiB), which **does not co-reside with the primary on a 16 GB card**: jasna
warns and continues, and if VRAM runs out a clip fails with an out-of-memory
error (the worker retries it once, then stops). With 24 GB or more,
`--no-swiftvr-accel` also runs.

### Color correction

SwiftVR's generated crops can also drift in tone from the primary restoration
they were built from, which after the blend reads as a color mismatch between
the restored region and its surroundings. Every output crop is therefore
corrected against the **bicubic-upscaled input crop (the primary output)**. The
method is FlashVSR's wavelet reconstruction (SwiftVR's high frequencies on the
input's low frequencies), and the function is shared with the FlashVSR worker
by loading it by path. Since SwiftVR's output lives on the GPU, the correction
is applied there frame by frame (numerically the same as the FlashVSR worker's
host-round-trip version).

There is no CLI flag. For A/B verification the environment variable
`JASNA_SWIFTVR_COLOR_FIX=adain|wavelet|none` overrides the method. A value left
in the shell overrides every later run, so pass it per command, as in
`JASNA_SWIFTVR_COLOR_FIX=none jasna ...`.

### Clip length and short clips

The clip length is set by `--max-clip-size` (default 90) as usual, with no cap.
SwiftVR restores a clip in fixed chunks (28 frames, then 24 at a time), so VRAM
is flat in the clip length (four DiT passes for 90 frames).

Short clips are padded inside `restore_clip()` by repeating the last frame, both
to the chunk protocol's 4k+1 and to **at least 25 frames**. A clip of up to 28
frames is a single LAST chunk whose DiT input always holds 7 latents, so the DiT
cost is the same for 2 frames and for 25. What differs is the padding: at 25
frames all 7 latents come from (repeated) real frames, while a shorter clip has
the missing latents filled with zeros. The 25-frame floor is a design choice
that avoids zero latents; its extra cost is a few autoencoder frames.

Note that however it is padded, a short clip's output differs from the same
frames restored inside a longer clip (the DiT sees every frame of its chunk, so
whether the following frames are copies or zeros changes the result). Measured
on an RTX 5080, clips of 1 to 21 frames restored on their own sit 31 to 37 dB
from the same frames inside an 89-frame clip, and which padding comes closer
varies with the clip length (with 25 real frames the difference is 47 dB).
Since the numbers do not separate the two, the padding that produces no zero
latents is used. FlashVSR pads short clips the same way (to 21 frames with
copies), and how short clips look is checked visually.

## Behavior and constraints

- **frame-gen forced off** (same reason as FlashVSR inline).
- **fp8-recon auto-enabled** (when not given). It lowers the primary's peak by
  ~0.9 to 1.7 GB to fit the co-residence budget; a GPU without fp8 falls back
  to TRT.
- **Not combinable with the SeedVR2 primary** (two resident workers exceed 16 GB;
  rejected at startup).
- Synchronous. SwiftVR is the rate limiter, so mosaic-dense stretches run at its
  speed (frames without mosaic are primary-only and fast).
- VR modes and `--stream` are not rejected, as with FlashVSR inline.
- Starting the worker takes the checkpoint read (~20 GB; a few seconds when it
  is in the page cache) plus the warmup (a few seconds including
  torch.compile). The handshake wait gives up after 600 s.
- **Not bundled, unrelated to the supporter models.** SwiftVR is an Apache-2.0
  third-party model; the checkout, checkpoint and venv are the user's. Not
  exposed in the GUI.

## Verification

### SwiftVR side (`restore_clip()`)

Checked on an RTX 5080 (16 GB), Linux, with the fork on an 89-frame 256px clip.

- Extracting the chunk processing out of `runner.py` left `restore_video()`'s
  PNG output bit-identical in all three configurations: bf16, FP8 +
  torch.compile at 4x, FP8 + torch.compile at 2x.
- The same 89 frames through `restore_clip()` are bit-identical to
  `restore_video()` in all three configurations (4k+1 frames, so the chunking
  is the same).
- Time per 90-frame clip (median after warmup) and peak VRAM (allocated /
  reserved):

| Configuration | 1 clip | Peak VRAM |
|---------------|--------|-----------|
| scale 4, FP8 + compile | 1.58 s (57 fps) | 7.9 / 8.6 GiB |
| scale 2, FP8 + compile | 0.41 s (217 fps) | 5.6 / 6.1 GiB |
| scale 4, bf16 (`restore_video`, 89 frames) | 4.2 s | 12.1 / 12.6 GiB |

### Through the worker

Checked by starting the real worker from jasna's restorer (random crops, color
correction and wire transfer included). Startup about 7 s (model load 2 s,
warmup 4 s). A 90-frame clip takes 2.1 s at scale 4, 0.6 s at scale 2, and
3.9 s at scale 4 in bf16 (`--no-swiftvr-accel`).

### End to end

Completion on real material, GPU-wide peak, wall time and the color-correction
gate are in "[VRAM and speed](#vram-and-speed)". The visual A/B against FlashVSR
is the user's.

## VRAM and speed

Linux, RTX 5080 16 GB (GPU-wide peak including ~1.9 GB of desktop residency,
`nvidia-smi` polled every second), the default clip 90, `--fp8-recon` (auto),
wavelet color correction. The FlashVSR rows were taken the same day on the same
material (fork, `--flashvsr-accel`, `--flashvsr-tiles 2` at scale 4). The output
frame count matched the input in every run.

| Material | Configuration | Wall time | GPU-wide peak |
|----------|---------------|-----------|---------------|
| 480p, 4930 frames | primary only | 25.0 s | 3.8 GB |
| 480p | `swiftvr-inline` scale 4 | **102.2 s** | **12.7 GB** |
| 480p | `swiftvr-inline` scale 2 | **44.8 s** | **10.4 GB** |
| 480p | `swiftvr-inline` scale 4, color fix none | 97.7 s | 12.8 GB |
| 480p | `flashvsr-inline` scale 4 tiles 2 | 411.7 s | 14.4 GB |
| 480p | `flashvsr-inline` scale 2 | 100.7 s | 10.6 GB |
| 1080p, 4203 frames | `swiftvr-inline` scale 4 | **139.5 s** | **14.0 GB** |
| 1080p | `swiftvr-inline` scale 2 | **56.9 s** | **12.0 GB** |

- Whole runs are about 4x (scale 4) and 2.2x (scale 2) faster than with
  FlashVSR inline. Net of the 25 s primary-only run, the secondary share is
  about 5x faster at scale 4 (387 to 77 s) and 3.8x at scale 2 (76 to 20 s).
  Earlier 1080p FlashVSR measurements: scale 2 127 s / 10.1 GB, scale 4 tiles 2
  556 s / 13.8 GB.
- Scale 4 at 1080p reaches 14.0 GB, a little over 2 GB below the ceiling, so
  use scale 2 where more is resident. There is no strip tiling.
- The color correction costs about 4.6% of the wall time (102.2 s vs 97.7 s).
- `--no-swiftvr-accel` (bf16, 480p, scale 4): jasna warned and continued, the
  worker ran out of VRAM on the first clip (one retry, then a clip error), and
  the secondary thread's exception ended the run with exit code 1 after 13 s
  (no hang; 14.6 GB GPU-wide). As designed.

### Gates

- **Color drift** (per-channel median |Δmean| inside the pixels the secondary
  changed, against the primary-only output, 480p,
  `scripts/evaluation/flashvsr-color-fix-report.py`): uncorrected 0.765 to
  **wavelet 0.225 at scale 4, 0.241 at scale 2** (pass). FlashVSR's same metric
  is 1.73 to 0.25: SwiftVR drifts less uncorrected and lands at the same level
  corrected.
- **Temporal change** (adjacent-frame difference inside the changed region,
  relative to the primary-only output; a crude proxy without flow
  compensation): 1.021 at scale 4, 1.056 at scale 2, 1.030 uncorrected (target
  at most 1.2).
- The visual A/B is the user's. FlashVSR outputs for it were produced the same
  day on the same material.

## Implementation

- `jasna/restorer/swiftvr_common.py`: registration of `--swiftvr-*` and path
  resolution (no torch import).
- `jasna/restorer/swiftvr_inline_secondary_restorer.py`: the synchronous
  `SecondaryRestorer`. Worker spawn and handshake, wire I/O, RGB/BGR flips, the
  keep-window slice. Same structure as the FlashVSR inline restorer, minus the
  patch check, the acceleration environment variables, runtime demotion
  reports and respawn.
- `jasna/restorer/swiftvr_inline_worker.py`: the worker under the SwiftVR venv.
  Imports neither jasna nor lada (it can be carried over to lada-ex as is).
  Acceleration decision, model load, warmup, per-clip `restore_clip()` and color
  correction. The color-correction primitives are shared with
  `flashvsr_inline_worker.py` by loading it by path.
- `jasna/session_config.py` / `session_factory.py` / `main.py`: config fields,
  restorer construction, startup checks (fp8-recon auto-enable, rejection of
  frame-gen and of the SeedVR2 primary).
- `scripts/build_nuitka.py`: copies the worker as a real file to
  `<dist>/jasna/restorer/` (next to the FlashVSR worker, for the shared color
  correction).
- Tests: `tests/test_swiftvr_inline.py` (stub worker: wire, flags, handshake,
  the GPU color fix against the FlashVSR version), `tests/test_main.py`
  (choices and defaults).

Fork side: `restore_chunk()` in `swiftvr/runner.py` (shared with the offline
runner) and `restore_clip()` in `swiftvr/pipeline.py`.

## Notes for Windows

Unverified on Windows. Expected differences:

- `expandable_segments` is unavailable, so the worker's reserved VRAM runs
  above Linux; the fork's FP8 figure (8.0 GiB) was measured on Windows, though,
  so no large gap is expected.
- Triton comes as `triton-windows` through `uv sync`; no C++ compiler is needed
  (fork README).
- `--swiftvr-python` defaults to `<repo>/.venv/Scripts/python.exe`.
