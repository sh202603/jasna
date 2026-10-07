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

There are two modes. `swiftvr-inline` is a single pass that co-resides SwiftVR
with the primary pipeline, for the `basicvsrpp` primary on a 16 GB card.
`swiftvr` is the same offline 3-phase pass as FlashVSR's `flashvsr` and runs
SwiftVR alone on the GPU ("[Offline 3-phase mode](#offline-3-phase-mode---secondary-restoration-swiftvr)");
it serves 12 GB-class GPUs, GPUs without FP8 and the SeedVR2 primary. The
inline description below (processing details, quality) applies to offline as
is; the differences are collected in the offline section.

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

The crops the worker receives are not the primary output as is but a **view**
of it whose placement is smoothed over time. The primary crops follow the
detection box and move from frame to frame, and SwiftVR redraws its detail on
that motion, so at the end of the primary stage each frame's 256px grid is
resampled into the placement averaged over 15 frames before it is sent, and
the blend composites SwiftVR's output straight into the frame with the view's
placement ("[Crop view smoothing](#crop-view-smoothing---swiftvr-view-window)").

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

# offline 3-phase (12 GB-class GPUs, GPUs without FP8, the SeedVR2 primary)
jasna --input in.mp4 --output out.mkv \
      --secondary-restoration swiftvr \
      --swiftvr-repo ~/SwiftVR \
      --swiftvr-bundle-dir /data/jasna_bundle \
      --log-level info
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--swiftvr-repo` | (required) | Path to the SwiftVR checkout (the fork, with `restore_clip()`). |
| `--swiftvr-python` | `<repo>/.venv/bin/python` | Python of the SwiftVR env (the venv `uv sync` creates). |
| `--swiftvr-model-dir` | `<repo>/checkpoints` | Checkpoint directory (`reae.safetensors`, `prompt_embedding.safetensors`, `transformer/`). |
| `--swiftvr-scale` | `4` | Processing scale. `4` = model-native 1024px, `2` = 512px (faster, lower VRAM). See "[Processing scale](#processing-scale---swiftvr-scale)". |
| `--swiftvr-view-window` | `15` | Smooth the position and scale of the crop view SwiftVR sees with a moving average over N frames (`0` disables). See "[Crop view smoothing](#crop-view-smoothing---swiftvr-view-window)". |
| `--swiftvr-accel` / `--no-swiftvr-accel` | **on** | FP8 DiT and torch.compile. Needs an RTX 40 series or newer GPU and a working Triton; the worker drops unavailable parts at startup with a warning. See "[Acceleration](#acceleration---swiftvr-accel)". |
| `--swiftvr-bundle-dir` | temp | `swiftvr` (offline) only: persist the intermediate bundle here (enables stage resume). |
| `--swiftvr-keep-bundle` | off | `swiftvr` (offline) only: keep the bundle after completion (implied by `--swiftvr-bundle-dir`). |

There is no counterpart of FlashVSR's `--flashvsr-version`, `--flashvsr-dtype`,
`--flashvsr-tiles`, `--flashvsr-lora` or `--flashvsr-max-clip-frames`: there is
one model, bf16 only (FP8 requires it), no strip tiling is needed, there is no
LoRA, and VRAM does not depend on the clip length, so no clip cap is needed.
Color correction has no flag either (always on, see below).

## Processing details

### Processing scale (`--swiftvr-scale`)

SwiftVR is a 4x model: the 256px crops are pre-upscaled bilinearly by the scale
and the DiT processes the result. `4` processes at 1024px as in training; `2`
processes at 512px. The output video resolution is the same for both.

Scale 2 is about 2.5 times faster than scale 4 over a whole run and peaks 2 to
2.5 GB lower GPU-wide ("[VRAM and speed](#vram-and-speed)"). In quality, with
crop view smoothing (on by default) it is visually on par with scale 4: no
difference in temporal steadiness, texture, placement accuracy or the look of
clip boundaries ("[Quality gates](#quality-gates)"). The default is scale 4,
the training configuration; use scale 2 where speed and VRAM matter.

What still differs between the two is the structure of the generation. At
512px the DiT's token grid is 16x16, one window, so the window shift has no
effect and the attention structure differs from training. And since the
one-step generation makes texture at the scale of its output pixels, scale 2
generates at a scale twice as coarse relative to the subject as scale 4. No
setting changes this; settings that weaken the generation (downscaling or
blurring the input, scaling down the DiT prediction, a lower timestep) remove
texture as fast as they remove unsteadiness
("[What determines temporal stability](#what-determines-temporal-stability)").

### Crop view smoothing (`--swiftvr-view-window`)

On by default (15 frames, `0` disables), for scale 2 and scale 4 alike.

The primary crops follow the detection box. jasna shrinks each crop with one
scale per clip (fitted to the clip's largest box), centres it in the 256px grid
and mirrors the margins, so the subject's position in the grid moves from frame
to frame (measured on 1080p material: 3.0 grid px per frame on average, box
width changes of up to 18%). SwiftVR reacts to sub-pixel input shifts by
redrawing its detail, and its output follows the input shift only about half
way (phase correlation: 1.2 to 1.6 px at 512 per frame). Left alone, the
restored region drifts against its surroundings every frame and the thickness
of edges and the texture appear to flicker. At scale 2 this is clearly visible
on 1080p material. The cause and the fix come from a report by
[mioh-labs](https://github.com/mioh-labs/mioh)
("[Acknowledgement](#acknowledgement)").

The smoothing stabilises only the framing of what SwiftVR sees and puts its
output back exactly where it belongs. The primary restoration, the blend mask,
the worker and the wire are unchanged.

- At the end of the primary stage (on the GPU), each frame's 256px grid is
  resampled bilinearly into a view whose placement is the moving average of
  the placements over N frames. Outside the grid the content is mirrored.
- Where the smoothed view would not cover the frame's own crop (about 10% of
  the frames at window 15; the uncovered band is 5 px in the median and 82 px
  at most), the view is shifted by the minimum that covers it. After this
  clamp the framing jitter is still less than half the unsmoothed one (3.04 to
  1.45 grid px per frame).
- The blend samples SwiftVR's output (in the view grid) straight into the
  frame's pixels, in one resample. With the frame's own placement this samples
  the same positions as the legacy composite, so the disabled case and the
  other secondary restorers are unchanged.

The effect saturates from a window of 15 (crop-level flow-warping error ratio
2.38 at 7, 2.24 at 15, 2.16 at 31), and a wider window makes the clamp bind on
more frames, so 15 is the default. Filling the outside of the grid from the
source frame instead of mirroring differs little on this metric (2.24 vs 2.34)
and would need a path that hands the source frame to the secondary stage, so
it is not used.

The effect is a crop-level flow-warping error ratio of 4.99 to 2.57 at scale 2
(scale 4: 2.16), and visually scale 2 becomes on par with scale 4 in temporal
steadiness and texture ("[Quality gates](#quality-gates)",
"[What determines temporal stability](#what-determines-temporal-stability)").
The cost is one `grid_sample` per clip; speed and VRAM are within measurement
noise ("[VRAM and speed](#vram-and-speed)").

What remains after the smoothing is a low-frequency variation with a 2-frame
period that is SwiftVR's own (the TAE's temporal compression) and the coarser
generation scale of scale 2.

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
`--no-swiftvr-accel` also runs. On Windows a 16 GB card was measured to
complete instead of running out of memory (about 2x slower, with VRAM at the
limit; see "[Notes for Windows](#notes-for-windows)"). In the offline `swiftvr` mode SwiftVR has the
GPU to itself, so bf16 is a regular path there and fits 16 GB
("[Offline 3-phase mode](#offline-3-phase-mode---secondary-restoration-swiftvr)").

### Color correction

SwiftVR's generated crops can also drift in tone from the primary restoration
they were built from, which after the blend reads as a color mismatch between
the restored region and its surroundings. Every output crop is therefore
corrected against the **bicubic-upscaled input crop (the primary output)**. The
method is FlashVSR's wavelet reconstruction (SwiftVR's high frequencies on the
input's low frequencies), the same functions as the FlashVSR worker's (inlined,
checked bit for bit by a test). Since SwiftVR's output lives on the GPU, the correction
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
  rejected at startup; the offline `swiftvr` mode combines with it).
- Synchronous. SwiftVR is the rate limiter, so mosaic-dense stretches run at its
  speed (frames without mosaic are primary-only and fast).
- VR modes and `--stream` are not rejected, as with FlashVSR inline.
- Starting the worker takes the checkpoint read (~20 GB; a few seconds when it
  is in the page cache) plus the warmup (a few seconds including
  torch.compile). The handshake wait gives up after 600 s.
- **Not bundled, unrelated to the supporter models.** SwiftVR is an Apache-2.0
  third-party model; the checkout, checkpoint and venv are the user's. Not
  exposed in the GUI.

## Offline 3-phase mode (`--secondary-restoration swiftvr`)

`swiftvr` is the same offline 3-phase pass as FlashVSR's `flashvsr`: each phase
runs as its own process, one after the other, so the SwiftVR phase has the GPU
to itself. The same crops go through the same functions as inline, so the
output matches inline's ("[Offline measurements](#offline-measurements)");
what differs is the VRAM requirement, the intermediate files and the resume.

| Phase | Env | What |
|-------|-----|------|
| 1 (dump) | jasna | decode + detect + primary restoration (BasicVSR++ or SeedVR2); serialize every clip's 256px crops (the same smoothed crop view as inline) + masks + geometry to a **bundle** on disk. blend/encode is throwaway. |
| 2 (SwiftVR) | SwiftVR venv | restore each clip's 256px crops to 256*scale px, color-correct them, write them back into the bundle (`jasna/restorer/swiftvr_phase2_driver.py`). |
| 3 (reblend) | jasna | re-decode the source, re-assemble the restore results from the bundle, blend with the view placements and encode the final output. |

Phase 1 and Phase 3 are FlashVSR's code (`jasna/restorer/flashvsr_offline.py`),
run as `jasna --flashvsr-phase dump` / `reblend` subprocesses (the internal name
stays `flashvsr`). The Phase 2 driver runs under the SwiftVR venv's Python and
loads the inline worker (`swiftvr_inline_worker.py`) by path to share its
acceleration decision, model load, warmup, the frame-count-checked
`restore_clip()` call and the GPU color correction. There is no wire, so no
BGR flip either (the bundle is RGB).

The **bundle** has FlashVSR's format, now version 2. Phase 1 builds the same
smoothed crop view as inline (`--swiftvr-view-window`, default 15) and writes
its placement into the clip geometry as `view_placements`; Phase 3 composites
straight from the view into the frame when placements are present (the inline
blend's path). FlashVSR bundles carry null placements and behave as before.
Phase 3 also reads version 1 bundles and refuses newer versions.

### When to use it

Inline co-resides SwiftVR with the primary pipeline: about 8 GiB in FP8, which
fits a 16 GB card next to the primary. It cannot be arranged in three cases,
and offline is the path for them.

1. A GPU without FP8 (RTX 30 series and older, compute capability below 8.9).
   The DiT runs in bf16 at about 12.4 GiB and does not co-reside with the
   primary even on 16 GB.
2. A 12 GB-class GPU. Even in FP8, SwiftVR plus the primary exceeds 12 GB
   (scale 4 needs over 11 GB for the application alone; scale 2 is borderline
   at 1080p).
3. The SeedVR2 primary. Two resident workers exceed 16 GB, so inline rejects
   the combination at startup.

Whether offline helps is decided by SwiftVR's standalone peak. Measured on an
RTX 5080 (16 GB, about 2.0 GB of desktop residency) after the same load and
warmup as Phase 2, over 3 clips of 90 256px frames (fork e7f186b, which loads
the FP8 DiT block by block; earlier forks peak at 10.4 GB during the load and
cannot load on a 12 GB card). allocated / reserved are torch's numbers, the
process peak is `nvidia-smi`'s.

| Configuration | Load peak | Run peak (allocated / reserved) | Process peak (expandable segments) | Same (without, Windows stand-in) | One 90-frame clip |
|---|---|---|---|---|---|
| scale 4, FP8 + compile | 5.3 GiB | 7.8 / 8.2 GiB | 8.8 GB | 9.0 GB | 1.6 s |
| scale 4, bf16 | 9.4 GiB | 12.4 / 12.6 GiB | 13.3 GB | 14.0 GB (measured on Windows) | 3.4 s |
| scale 2, FP8 + compile | 5.3 GiB | 5.6 / 5.9 GiB | 6.4 GB | 6.5 GB | 0.4 s |
| scale 2, bf16 | 9.4 GiB | 10.2 / 10.3 GiB | 10.9 GB | not measured | 0.8 s |

To simulate a 12 GB card, the FP8 runs were repeated with torch's allocation
capped at 10.5 GiB and at 9.5 GiB: scale 4 and scale 2 both completed under
10.5 GiB, and scale 4 also under 9.5 GiB (same peak as without the cap).

- **12 GB-class with FP8 (RTX 4070, 5070 and the like):** scale 4 in FP8 fits
  at 8.8 GB, with room for 3 GB of desktop residency. Scale 2 is 6.4 GB.
- **RTX 30 series, 16 GB:** scale 4 in bf16 fits at 13.3 GB
  (`--no-swiftvr-accel` is not needed: the driver drops FP8 and falls back to
  bf16 without a warning). On Windows (no `expandable_segments`) it is 14.0 GB
  and completed at 15.4 GB including 1.4 GB of desktop residents. Above about
  2 GB of residents it reaches the ceiling, so keep the desktop residents small.
- **12 GB-class without FP8 (RTX 3060 12 GB and the like):** scale 4 in bf16
  does not fit; scale 2 in bf16 is borderline at 10.9 GB.
- **SeedVR2 primary:** the SeedVR2 worker runs in Phase 1 and SwiftVR in
  Phase 2, so the combination works.

Inline's GPU-wide peak (same fork, RTX 5080, about 1.85 GB of desktop
residency included) is 10.5 GB at 480p / 11.4 GB at 1080p for scale 2 and
12.8 / 14.0 GB for scale 4. On a 12 GB card the measurements say: scale 4
offline; scale 2 inline first, and offline if VRAM pressure shows (offloader
spills, worker out-of-memory warnings).

### Offline behavior and constraints

- Rejected at startup, as in FlashVSR's offline mode: `--stream`,
  `--frame-gen`, `--retarget-high-fps`, `--segments`, VR modes (including
  `--vr-mode auto` detecting VR content), folder input, image input.
- `--max-clip-size` passes through as given (default 90); there is no
  counterpart of FlashVSR offline's `--flashvsr-max-clip-frames`.
  `--swiftvr-scale`, `--swiftvr-view-window` and `--swiftvr-accel` mean the
  same as inline.
- **bf16 is a regular path.** On a GPU without FP8 the driver drops FP8 and
  runs bf16, without inline's warning.
- **Combinable with the SeedVR2 primary.** Inline's fp8-recon auto-enable does
  not apply: Phase 1 runs the primary on the standard TensorRT path unless
  `--fp8-recon` is passed (inline enables it for its co-residence VRAM budget).
  Pass `--fp8-recon` for the same primary as inline.
- Two encodes (Phase 1's throwaway and Phase 3's final) and a re-decode of the
  source in Phase 3. This fixed cost is FlashVSR's too; with SwiftVR's shorter
  secondary time it is a larger share ("[Offline measurements](#offline-measurements)").
- **Stage resume.** With `--swiftvr-bundle-dir` the bundle persists; re-running
  the same command after a failure makes Phase 2 skip the completed clips
  (Phase 1 runs again). If VRAM runs out mid-clip in Phase 2, the driver
  retries once, then exits non-zero and leaves the bundle.
- **Disk space.** The bundle is dominated by Phase 2's uncompressed restored
  crops, 3 x (256 x scale)^2 bytes per frame (3 MiB at scale 4, 0.75 MiB at
  scale 2). The rule of thumb (about 8 GB per mosaic minute at scale 4), the
  `/tmp` on tmpfs caveat, the warning before Phase 1 and the space gate before
  Phase 2 are those of [FlashVSR's disk space section](flashvsr.md#disk-space),
  with `--swiftvr-bundle-dir` in the messages.
- The Phase 2 driver sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on
  Linux (like the inline worker; with fork e7f186b it moves the FP8 peak by
  only 0.2 GB, but it keeps the bf16 path's 12.4 GiB inside 16 GB). On Windows
  it sets `PYTHONUTF8=1`.
- The frozen build copies the driver as a real file to
  `<dist>/jasna/restorer/` as well.

### Offline measurements

Linux, RTX 5080 16 GB, fork e7f186b. The inline runs were repeated the same
day with the same jasna and fork (within noise of the inline table above). The
GPU-wide peak includes about 1.9 GB of desktop residency (0.5 s `nvidia-smi`
poll); per-phase peaks and times are per process. Default clip 90, wavelet
color correction, view smoothing 15. The output frame count matched the input
in every run.

| Material | scale | inline wall / GPU-wide peak | offline wall / GPU-wide peak | Phase 1 / 2 / 3 process peak | Phase 1 / 2 / 3 time |
|----------|-------|------|------|------|------|
| 480p | 2 | 44.4 s / 10.3 GB | **62.8 s / 8.0 GB** | 2.9 / 6.4 / 0.5 GB | 27.3 / 27.9 / 2.1 s |
| 480p | 4 | 101.3 s / 12.6 GB | **118.0 s / 10.4 GB** | 2.9 / 8.8 / 0.5 GB | 27.3 / 80.7 / 4.7 s |
| 1080p | 2 | 54.7 s / 11.1 GB | **95.0 s / 8.0 GB** | 4.0 / 6.4 / 1.0 GB | 42.0 / 38.4 / 9.5 s |
| 1080p | 4 | 137.5 s / 13.8 GB | **172.9 s / 10.4 GB** | 4.0 / 8.8 / 1.0 GB | 41.5 / 113.9 / 12.2 s |
| 480p | 4, bf16 (`--no-swiftvr-accel`) | (inline runs out of VRAM) | **186.6 s / 14.9 GB** | 2.9 / 13.3 / 0.5 GB | 27.4 / 149.7 / 4.3 s |

- The GPU-wide peak is Phase 2's (SwiftVR alone) and matches the standalone
  table above. Phase 1 is the primary-only run; Phase 3 is light.
- The wall-time difference is the fixed cost of Phase 1's throwaway encode and
  Phase 3's re-decode and encode; with SwiftVR's short secondary time the
  ratio to inline is larger than FlashVSR's.
- **Equivalence gate** (inline and offline outputs decoded and compared frame
  by frame with `ffmpeg`'s psnr filter). With the default encode (HEVC NVENC)
  the result was a mean of 46.22 dB at 480p (min
  42.16 dB) and 47.97 to 48.85 dB at 1080p (min
  42.07 to 44.02 dB), but that is the encode noise floor: on
  the same material, FP8 vs bf16 offline outputs (restorations that really
  differ) sit at 46.29 dB and FlashVSR tiny vs tiny-long at
  46.36 dB, indistinguishable, while frames without mosaic are bit-identical
  between inline and offline (1490 of 4930). Seeing the
  difference needs the encode out of the way, so 480p scale 2 was repeated
  with `--encoder-settings tune=lossless,spatial_aq=0,temporal-aq=0` (NVENC lossless; the default
  adaptive quantization cannot be combined with it, so it is switched off):
  - inline vs offline (FP8): mean 65.31 dB, min 53.17 dB, 2221 of 4930 frames identical
  - offline bf16 vs FP8 (control: the restorations differ, so this must differ): mean 69.87 dB, min 60.69 dB, 2221 of 4930 frames identical
  - inline vs offline bf16: mean 65.31 dB, min 53.30 dB, 2221 of 4930 frames identical
  - inline vs offline with `--fp8-recon` passed to Phase 1: all 4930 frames bit-identical (PSNR inf)
  The residual against inline is the primary restoration: inline auto-enables
  `--fp8-recon` for its co-residence VRAM budget, while Phase 1 runs the primary
  alone and does not. With `--fp8-recon` passed to Phase 1, all 4930 frames are
  bit-identical to inline. The SwiftVR phase is deterministic on the same crops
  (the resume test is bit-identical too), and the bf16 control sits 69.87 dB
  from FP8, so the gate does discriminate.
- **bf16** (`--no-swiftvr-accel`, scale 4, 480p): the driver reported
  `accel: off` and completed in bf16. PSNR against the FP8 output: mean
  46.29 dB (min 42.12 dB), the level of the fork's FP8 vs bf16
  measurement (about 47 dB).
- **Resume:** killing the driver mid-Phase 2 (after 3 clips) ended the run
  non-zero with `SwiftVR run failed. Bundle kept for resume at:` and the
  `--swiftvr-bundle-dir` hint. The same command re-run completed with Phase 2
  reporting `done: 54 restored, 3 already present`, and the output matches the uninterrupted run:
  all 4930 frames bit-identical (PSNR inf). This machine runs MPS in Exclusive_Process mode, which refuses
  new CUDA contexts with `cudaErrorDevicesUnavailable` right after a client is
  killed, so the re-run waited until the GPU accepted one (5 s
  after the kill; not a jasna constraint).
- **SeedVR2 primary:** on a 20 s (500-frame) 480p cut,
  `--restoration-model-name seedvr2` + `swiftvr` (scale 2) was accepted at
  startup and completed (57.1 s, GPU-wide peak 12.6 GB;
  Phase 1 is jasna 1.2 GB plus the SeedVR2 worker
  9.8 GB, Phase 2 6.4 GB). On the same cut,
  `swiftvr-inline` was rejected at startup with `cannot be combined`.
- **Short clips:** the 480p bundle holds 57 clips, the shortest 2 frames (4 of up to 2 frames, 0 without a view); the 1080p bundle 58 clips, the shortest 2 frames (1 of up to 2 frames, 0 without a view). A single-frame clip has no view and Phase 3
  composites it through the legacy path (pinned by a unit test).
- **FlashVSR regression:** `flashvsr` offline (scale 2, `--flashvsr-accel`)
  completed as before on a version 2 bundle (placements null for every clip:
  57 / 57), 144.0 s, 4930 frames
  (PSNR mean 46.36 dB against the old FlashVSR inline output, the tiny
  vs tiny-long difference).

## SwiftVR distill (`--secondary-restoration swiftvr-distill`)

`swiftvr-distill` runs a small convolutional network distilled from SwiftVR's
outputs (24 channels, 12 residual blocks, about 131k parameters) in place of
SwiftVR itself. It takes the same 256px primary crops, five frames at a time,
and returns the centre frame at 512px (2x); the blend shrink-composites it back
as with `--swiftvr-scale 2`. It runs in PyTorch FP32 inside the jasna process:
no SwiftVR checkout, venv or resident worker, about 0.2 GB of VRAM on top of
the primary, and a secondary stage 4 to 6 times faster than `swiftvr-inline`
at scale 2. What it adds is tighter outlines and a plausible texture synthesized
from the restored crop. It is not a restoration model on its own, and nothing
in its output is information recovered from the source.

The model (`TinyROIEnhancer`, checkpoint `roi-distill-pilot-v1`) was trained by
the author of [mioh](https://github.com/mioh-labs/mioh) as a student of SwiftVR
run on lada-restored crops, then fine-tuned with a detail loss against aligned
uncensored frames. It is published at
[`okatti/swiftvr-distill`](https://huggingface.co/okatti/swiftvr-distill) under
AGPL-3.0 (the same license as jasna; the repository carries the
`not-for-all-audiences` tag). jasna does not bundle it: download
`swiftvr-distill.pt` from there into `model_weights/`, where the other weights
live (`--swiftvr-distill-model` points at another location). The network jasna
builds is the one that repository's README defines; the published state dict
loads unchanged.

### Usage

```bash
# the weights, once, into model_weights/
wget -O model_weights/swiftvr-distill.pt \
  https://huggingface.co/okatti/swiftvr-distill/resolve/main/swiftvr-distill.pt

jasna --input in.mp4 --output out.mkv --secondary-restoration swiftvr-distill
```

`model_weights/` is searched as for the other models (`$JASNA_MODEL_WEIGHTS_DIR`,
next to the executable, the current directory, next to the package). Without the
file, jasna stops before engine compilation and names the download.

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--swiftvr-distill-model` | `<model_weights>/swiftvr-distill.pt` | Path to the checkpoint. Unset, it is `swiftvr-distill.pt` in `model_weights/`; a bare file name is looked up there too. The file is a dict with `version`, `architecture` and `model`, read with `weights_only=True`. |
| `--swiftvr-distill-view-window` | `15` | Crop view smoothing over N frames, the same mechanism as `--swiftvr-view-window` (`0` disables). See "[Crop view smoothing](#crop-view-smoothing---swiftvr-view-window)". |
| `--swiftvr-distill-strength` | `0.75` | Scale of the detail the model adds to the bilinear 2x of its input (0 to 2). `0` is the plain upscale, `1` the model's output. Flicker and texture both grow roughly in proportion to it. |
| `--swiftvr-distill-stabilize` | `0` | Experimental. Blend the added detail over N frames on each side (0 to 8) where the inputs agree. jasna's own addition, not something the author's app does; on the test clip it traded detail for flicker like a lower strength, and it is not measured on real footage. |

### Behavior and constraints

- Synchronous in-process `SecondaryRestorer`, the same frame as
  `rtx-super-res`: no worker, no handshake, no clip cap. It runs in batches of
  4 centre frames, so VRAM is flat in the clip length.
- The five-frame window repeats the clip's edge frames; frames outside the
  kept range are not returned but still serve as temporal context.
- Unlike `swiftvr-inline`, nothing is forced or rejected at startup: fp8-recon
  is not auto-enabled, `--frame-gen` is not turned off and the SeedVR2 primary
  is not excluded (nothing else is resident). There is no color correction
  either: the model adds detail to a bilinear upscale of the primary output,
  so the colours stay the primary's.
- The author's app also passes the composited frame through macOS's
  VideoToolbox temporal noise filter, which it credits with much of its
  steadiness. jasna has no counterpart on Windows or Linux and does not
  replicate it; the view smoothing and the default strength are what hold the
  flicker down here.
- Not exposed in the GUI. `--stream` and `--segments` are not rejected but were
  not tested with it (`--segments` could not be exercised: on the material at
  hand the smart render fails without a secondary restorer as well). VR was run
  once on Linux on an 8K SBS clip; it completed with the same frame count as
  without a secondary restorer, and the user's visual check confirmed the
  effect. Verified on Windows (RTX 5060 Ti) and Linux (RTX 5080).

### Measurements

Windows, RTX 5060 Ti 16 GB, BasicVSR++ (TensorRT) primary. Three 12 s segments
(361 frames each) of one 1080p source were taken where the mosaic is detected
throughout, with little, medium and much motion in the primary crops. Flicker
increase and texture ratio are the `flicker_increase` / `texture_ratio` of
mioh-labs' `swiftvr_view_smoothing.py` (the reference script of their flicker
report, see "Acknowledgement"), relative to no secondary restoration, with the
mask from the `swiftvr-inline` output. `swiftvr-inline` ran at scale 2, view window
15, acceleration on. Average of the three segments:

| Secondary restoration | Flicker increase | Texture | Secondary stage |
|-----------------------|------------------|---------|-----------------|
| `swiftvr-distill`, strength 1.0, window 0 | +19.1% | 132.2% | 2.0 s |
| `swiftvr-distill`, strength 1.0, window 15 | +13.9% | 133.7% | 1.9 s |
| `swiftvr-distill`, strength 0.75, window 15 (default) | +8.6% | 122.4% | 2.0 s |
| `swiftvr-distill`, strength 0.5, window 15 | +4.2% | 111.6% | 2.0 s |
| `rtx-super-res` 2x | +0.8% | 106.1% | 0.8 s |
| `swiftvr-inline` scale 2 | +10.9% | 130.0% | 8.2 to 12.0 s |

The GPU-wide peak was about 6.0 GB with no secondary restoration, 6.0 to
6.4 GB with `swiftvr-distill` and 11.4 to 11.8 GB with `swiftvr-inline`
(about 2.7 GB of residents included).

- **View smoothing works for this model too.** At strength 1.0 it takes the
  flicker from +19.1% to +13.9% at the same texture; on the quiet segment from
  +38.7% to +25.7%. (On the 1080p test clip it moved the number by one point,
  which did not generalize: the crop framing there barely moves.)
- **Flicker and texture scale with the strength.** At equal texture the flicker
  is 1 to 2 points above `swiftvr-inline` at scale 2.
- **It behaves differently on motion.** On the high-motion segment
  `swiftvr-inline` adds almost nothing (texture 101.5%) while `swiftvr-distill`
  raises the texture by 23%.
- **A still input does not flicker**: on repeated identical frames the added
  detail is exactly constant. On the test clip about 70% of the flicker
  increase is also produced by a linear sharpen of the same texture, so most of
  it comes with stronger outlines as such.

The defaults (strength 0.75, window 15) were chosen from these numbers and the
user's visual check of several clips: the flicker is unobtrusive and the
texture acceptable. Outlines come out tighter than SwiftVR's, surfaces
smoother.

### Measurements on Linux

Linux, RTX 5080 16 GB, 2026-10-08. BasicVSR++ (TensorRT) primary; `swiftvr-inline`
at scale 2, view window 15, acceleration on. The GPU-wide peak includes about
1.9 GB of desktop residency and is the maximum of a 500 ms `nvidia-smi` poll.
The secondary stage is the `restore` of the `[timing] secondary` log line
(a synchronous restorer, so close to the GPU time); the output frame count
matched the input in every run.

| Material | Secondary restoration | Secondary stage | Wall time | GPU-wide peak |
|---|---|---|---|---|
| Test clip, 1080p, 300 frames | none | 0.0 s | 5.9 s | 4.8 GB |
| same | `swiftvr-distill` (defaults) | 0.4 s | 5.3 s | 4.9 GB |
| same | `swiftvr-inline` scale 2 | 2.7 s | 35.3 s | 10.5 GB |
| 480p, 4931 frames | none | 0.0 s | 18.4 s | 4.5 GB |
| same | `swiftvr-distill` (defaults) | 3.8 s | 21.7 s | 4.8 GB |
| same | `swiftvr-inline` scale 2 | 26.0 s | 42.3 s | 10.4 GB |
| 1080p, 4203 frames | none | 0.1 s | 24.0 s | 5.5 GB |
| same | `swiftvr-distill` (defaults) | 5.8 s | 27.5 s | 5.7 GB |
| same | `swiftvr-inline` scale 2 | 38.9 s | 52.4 s | 11.2 GB |

- The secondary stage takes 1/6.7 to 1/6.8 of `swiftvr-inline` at scale 2; on
  the test clip 0.4 s against 1.0 s on the RTX 5060 Ti (Windows). The
  `swiftvr-inline` wall time includes the worker start (model load 9 s,
  warmup 9 s).
- The GPU-wide peak rises by 0.05 to 0.3 GB over no secondary restoration.
- Strength 0 against no secondary restoration: Y PSNR 61 dB between lossless
  encodes. It is not bit-identical, since the primary crop makes a round trip
  through the bilinear 2x and the shrink composite, but the difference is far
  below what can be seen. `--swiftvr-distill-stabilize 2` adds 0.1 s to the
  secondary stage and 0.44 GB of VRAM on the test clip.
- On an 8K SBS VR clip (2998 frames) no secondary restoration took 103.6 s at
  a 14.0 GB peak and `swiftvr-distill` 112.1 s at 13.8 GB; the peak difference
  is within the measurement noise (the primary dominates).

The flicker and texture metrics are the same `flicker_increase` /
`texture_ratio` as on Windows, taken on three 361-frame windows of the 1080p
outputs above (chosen by masked area, largest first), relative to no secondary
restoration, with the mask from the `swiftvr-inline` output. Average of the
three windows:

| Secondary restoration | Flicker increase | Texture |
|---|---|---|
| `swiftvr-distill`, strength 1.0, window 0 | +3.4% | 139.3% |
| `swiftvr-distill`, strength 0.75, window 15 (default) | +2.0% | 119.9% |
| `swiftvr-inline` scale 2 | +1.0% | 212.4% |

The texture ranking (inline highest, strength 1.0 next, the default lowest) and
the default flickering less than strength 1.0 / window 0 match Windows. The
flicker of `swiftvr-inline`, however, came out below the default
`swiftvr-distill`, the reverse of the Windows ranking (+10.9% against +8.6%).
All three windows of this material carry much motion (the base luma changes by
9 to 10 levels per frame), and every configuration's flicker increase stays
below a third of the Windows values, so the ranking depends on the material.
The numbers themselves are material-specific and are not compared with the
Windows table.

The user compared the 1080p and 480p outputs with no secondary restoration and
with `swiftvr-inline` at scale 2, and checked the 8K SBS VR output as well:
no issues, and the effect is visible on 8K SBS too.

### Implementation

`jasna/restorer/swiftvr_distill_model.py` (the network and the checkpoint
checks: known version, `architecture` bounds, finite tensors, strict load),
`swiftvr_distill_secondary_restorer.py` (the restorer, strength and
stabilization), the wiring in `session_config.py` / `session_factory.py` /
`main.py` (the model path goes through the same `model_weights/` lookup as the
other weights and is checked before engine compilation as well).
Tests: `tests/test_swiftvr_distill.py` (CPU; the cases on the real weights run
when `model_weights/swiftvr-distill.pt` exists or `JASNA_SWIFTVR_DISTILL_MODEL`
points at the checkpoint) and `tests/test_main.py`.

## Measurements

Linux, RTX 5080 16 GB. The GPU-wide peak includes about 1.9 GB of desktop
residency and is the maximum of a 1 s `nvidia-smi` poll. Default clip 90,
`--fp8-recon` (auto), wavelet color correction, view smoothing at its default
(15). The material is 480p (4930 frames) and 1080p (4203 frames); the output
frame count matched the input in every run.

### VRAM and speed

| Material | Configuration | Wall time | GPU-wide peak |
|----------|---------------|-----------|---------------|
| 480p | primary only | 25.0 s | 3.8 GB |
| 480p | `swiftvr-inline` scale 4 | **102.2 s** | **12.7 GB** |
| 480p | `swiftvr-inline` scale 2 | **44.2 s** | **10.3 GB** |
| 480p | `swiftvr-inline` scale 4, color fix none | 97.7 s | 12.8 GB |
| 480p | `flashvsr-inline` scale 4 tiles 2 | 411.7 s | 14.4 GB |
| 480p | `flashvsr-inline` scale 2 | 100.7 s | 10.6 GB |
| 1080p | `swiftvr-inline` scale 4 | **137.0 s** | **13.9 GB** |
| 1080p | `swiftvr-inline` scale 2 | **54.1 s** | **11.2 GB** |

- Whole runs are about 4x (scale 4) and 2.3x (scale 2) faster than with
  FlashVSR inline. Net of the 25 s primary-only run, the secondary share is
  about 5x faster at scale 4 (387 to 77 s) and 4x at scale 2 (76 to 19 s).
  Earlier 1080p FlashVSR measurements: scale 2 127 s / 10.1 GB, scale 4 tiles 2
  556 s / 13.8 GB.
- Scale 4 at 1080p reaches 13.9 GB, a little over 2 GB below the ceiling.
  There is no strip tiling.
- View smoothing costs nothing measurable: `--swiftvr-view-window 0` gives
  53.8 s / 11.2 GB at 1080p scale 2 and 43.8 s / 10.3 GB at 480p scale 2 (0.3
  to 0.4 s from the default). The FlashVSR comparison rows and the 480p scale 4
  row are runs without the smoothing; the difference is within the same noise.
- The color correction costs about 4.6% of the wall time (102.2 s vs 97.7 s).
- `--no-swiftvr-accel` (bf16, 480p, scale 4): jasna warned and continued, the
  worker ran out of VRAM on the first clip (one retry, then a clip error), and
  the secondary thread's exception ended the run with exit code 1 after 13 s
  (no hang; 14.6 GB GPU-wide). As designed.

### Quality gates

- **Color drift** (per-channel median |Δmean| inside the pixels the secondary
  changed, against the primary-only output, 480p,
  `scripts/evaluation/flashvsr-color-fix-report.py`): uncorrected 0.765 to
  **wavelet 0.225 at scale 4, 0.224 at scale 2** (pass). FlashVSR's same metric
  is 1.73 to 0.25: SwiftVR drifts less uncorrected and lands at the same level
  corrected.
- **Temporal change** (adjacent-frame difference inside the changed region,
  relative to the primary-only output; a crude proxy without flow
  compensation): at 480p 1.021 at scale 4 and 1.029 at scale 2; at 1080p (every
  4th frame pair) 1.33 at scale 4 and 1.43 at scale 2. Scale 2 without view
  smoothing is 1.054 at 480p and 1.50 at 1080p, so the smoothing brings it
  close to scale 4 without reaching it. This proxy also counts the detail
  change that comes with subject motion and the 2-frame variation the
  smoothing does not address, so it falls less than the crop-level
  flow-warping error ratio.
- **Visual A/B** (the user's, 480p and 1080p): texture and detail at scale 4
  are on par with FlashVSR inline's output of the same material. Scale 2 is on
  par with scale 4 in temporal steadiness and texture, with no misplacement
  and no visible seam at the clip boundaries (every 74 frames). Smoothing at
  scale 4 loses no texture. Scale 2 without view smoothing is temporally less
  stable than scale 4, clearly visible on 1080p material.

### What determines temporal stability

The causes of the scale 2 unsteadiness were isolated on the primary-restored
crops of the 1080p material: 58 crop clips, 5085 frames. SwiftVR's
`restore_clip()` is run directly, its output gets the production wavelet color
correction, and it is compared against the primary-only crops
(bicubic-upscaled). The metric is the flow-warping error ratio (flow computed
once from the primary-only crops with SPyNet and applied to both; lower is
steadier over time), evaluated at 512px inside the crops' valid region.

| Variant | Flow-warping error ratio |
| --- | --- |
| scale 2 | 4.99 |
| scale 2, view smoothing (mirrored margins + clamp, the production setup) | 2.57 |
| scale 2, view smoothing (margins filled from the source frame, no clamp) | 2.24 |
| scale 4 | 2.16 |
| scale 4, view smoothing (margins filled from the source frame, no clamp) | 1.59 |

- **The input framing is the main cause:** a blur-only control (the output
  shifted by the same sub-pixel amounts as the smoothing) gives 4.23, so the
  drop to 2.2 to 2.6 comes from the stabilised input itself. On sharpness, the
  bilinear resample that maps the output back to the own grid for this
  evaluation halves the Laplacian variance, but the production blend does not
  take that path, and in view space the variance is kept (324 to 323).
- **The DiT generation is where the unsteadiness arises:** a TAE round trip
  alone, without the DiT, is steady even at scale 2 (ratio 1.26, first 20
  clips, 768px evaluation). The generation reacts to the input shifts.
- **Not the chunk boundaries:** pairs across a chunk boundary (output frames
  25, 49, 73) and all other pairs have the same ratio (3.758 and 3.763).
  Overlap (1 or 2 latents of the previous chunk as context) and a chunk length
  of 48 do not help.
- **Not FP8:** bf16 gives 3.75, the same (FP8 3.76).
- **Not the window size:** an 8x8 window, which restores the window shift at
  512px, does not help.
- **Weakening the generation removes texture too:** downscaling the input
  (128px then 4x: 3.11) or blurring it, scaling down the DiT prediction, and a
  lower timestep all lose as much sharpness as unsteadiness and do not reach
  scale 4.

### SwiftVR-side checks

The fork's `restore_clip()` was checked on an RTX 5080 with an 89-frame 256px
clip.

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

Started from jasna's restorer as the real worker (random crops, color
correction and wire transfer included), startup takes about 7 s (model load
2 s, warmup 4 s), and a 90-frame clip 2.1 s at scale 4, 0.6 s at scale 2, and
3.9 s at scale 4 in bf16 (`--no-swiftvr-accel`).

## Implementation

- `jasna/restorer/swiftvr_common.py`: registration of `--swiftvr-*`, path
  resolution and the offline Phase 2 command (no torch import).
- `jasna/restorer/swiftvr_inline_secondary_restorer.py`: the synchronous
  `SecondaryRestorer`. Worker spawn and handshake, wire I/O, RGB/BGR flips, the
  keep-window slice; it announces the view smoothing window through its
  `view_smoothing_window` attribute. Same structure as the FlashVSR inline
  restorer, minus the patch check, the acceleration environment variables,
  runtime demotion reports and respawn.
- `jasna/restorer/swiftvr_inline_worker.py`: the worker under the SwiftVR venv.
  Imports neither jasna nor lada (it can be carried over to lada-ex as is).
  Acceleration decision, model load, warmup, per-clip `restore_clip()` and color
  correction. The color-correction primitives (wavelet / AdaIN) are inlined, the
  same math as the FlashVSR worker's (a test checks the two agree bit for bit),
  so the file stays identical with lada-ex's copy.
- `jasna/restorer/swiftvr_phase2_driver.py`: the offline Phase 2 driver (SwiftVR
  venv). Imports no jasna; loads the sibling worker by path and shares its
  acceleration decision, model load, warmup, frame-count-checked `restore_clip()`
  call and GPU color correction. Walks the bundle's clips, skipping completed ones.
- `jasna/restorer/flashvsr_offline.py`: the offline orchestrator shared with
  FlashVSR. An engine record (`OfflineEngine`) swaps in the path resolution and
  flag names, the scale, Phase 1's clip cap (FlashVSR only) and view window
  (SwiftVR only), and the Phase 2 command (`swiftvr_common.swiftvr_phase2_command()`).
  The Phase 1 dump hook overrides `RestorationPipeline.view_smoothing_window` with
  the configured window to build the view, written as bundle version 2's
  `view_placements`; Phase 3 blends through the view path when they are present.
- `jasna/tracking/crop_view.py`: the crop view geometry (placements,
  smoothing, clamp, resampling into the view, sampling for the blend).
  `restorer/restoration_pipeline.py` reads the secondary restorer's
  `view_smoothing_window` and builds the view at the end of the primary stage,
  `pipeline_items.py` carries the placements as `view_placements`, and
  `blend_buffer.py` composites straight from the view only when they are set
  (0 or no attribute means the legacy path; the FlashVSR inline restorer could
  use the same mechanism).
- `jasna/session_config.py` / `session_factory.py` / `main.py`: config fields,
  restorer construction, startup checks (fp8-recon auto-enable, rejection of
  frame-gen and of the SeedVR2 primary).
- `scripts/build_nuitka.py`: copies the worker and the Phase 2 driver as real
  files to `<dist>/jasna/restorer/` (next to the FlashVSR worker: the driver
  loads the worker by path).
- Tests: `tests/test_swiftvr_inline.py` (stub worker: wire, flags, handshake,
  the GPU color fix against the FlashVSR version), `tests/test_swiftvr_offline.py`
  (startup checks for both engines, the Phase 2 command, bundle version 2 round
  trip and version check, the Phase 1 hook's view, Phase 3's view blend matching
  the inline path, the driver), `tests/test_main.py` (choices and defaults), `tests/test_crop_view.py` (view geometry: the own
  placement reproduces the legacy path, round trip through a smoothed view,
  clamp coverage), and the view path in `test_restoration_pipeline.py` and
  `test_blend_buffer.py`.

Fork side: `restore_chunk()` in `swiftvr/runner.py` (shared with the offline
runner) and `restore_clip()` in `swiftvr/pipeline.py`.

## Acknowledgement

The scale 2 temporal stability fix (crop view smoothing) is due to a report and
reference implementation by [mioh-labs](https://github.com/mioh-labs/mioh).
They traced the flicker not to SwiftVR itself but to the framing of the crop
SwiftVR sees moving from frame to frame, and published the countermeasure
(smooth the crop's position and scale with a 15-frame moving average and map
the output back to the original box) together with the probe experiments and
the evaluation metrics. jasna's implementation places that geometry in the
pipeline's primary and blend stages; the mirrored margins and the clamp are
jasna's own design.

The SwiftVR distill model (`roi-distill-pilot-v1`) was trained by okatti, the
same author of mioh, as a student of SwiftVR, and published together with its
network definition at [`okatti/swiftvr-distill`](https://huggingface.co/okatti/swiftvr-distill)
under AGPL-3.0. Its integration into jasna follows the layout of the proposal
they submitted (PR #4 of this repository); the restorer frame (a synchronous
`SecondaryRestorer` reusing the crop view smoothing), the strength and
stabilization options and the choice of defaults are jasna's own design.

## Notes for Windows

Inline (`swiftvr-inline`) was verified on Windows 11 with an RTX 5060 Ti 16 GB
(2026-09-27, jasna `32168df`, SwiftVR fork `e7f186b`). The offline 3-phase mode
(`swiftvr`) was verified on the same machine (same day, jasna `afe8d18`; see
"[Offline 3-phase measurements](#offline-3-phase-measurements-windows)").

- Setup is the same as on Linux. Triton comes as `triton-windows` through
  `uv sync`; neither a C++ compiler nor dev headers are needed. Acceleration
  (FP8 DiT, torch.compile) passed its checks and was enabled.
  `--swiftvr-python` defaults to `<repo>/.venv/Scripts/python.exe`.
- Worker startup took 5 to 6 s of model load and 11 to 12 s of warmup (with the
  checkpoints in the page cache). The first time only, compiling Triton's probe
  kernel prints one line, `remark: ... instructions in function`, on the
  worker's stderr. It is harmless and gone once the kernel cache is warm.
- **It fits 16 GB.** Without `expandable_segments`, the whole-GPU peak at 1080p
  scale 4 was 13.2 GB (including 1.5 GB of desktop residents, out of 15.9 GB),
  leaving 2.7 GB of headroom. More resident usage raises it by as much (14.2 GB
  in a run with 2.5 GB resident).
- **`--no-swiftvr-accel` (bf16) completed instead of stopping with an
  out-of-memory error** (480p, scale 4), unlike on Linux. It took about 1.9x
  the accelerated wall time, and the whole-GPU peak of 15.6 GB sat at the
  card's limit (15.9 GB). Whether the Windows driver's CUDA Sysmem Fallback
  (on by default) spilled the shortfall to system RAM, or the primary's small
  VRAM at 480p just left enough room, was not isolated. With Sysmem Fallback
  disabled or on larger inputs it can still run out of memory, so on a 16 GB
  card without acceleration use the offline `swiftvr` mode.
- In PowerShell, set the test-only variable `JASNA_SWIFTVR_COLOR_FIX` with
  `$env:JASNA_SWIFTVR_COLOR_FIX="none"` and clear it after the run with
  `Remove-Item Env:JASNA_SWIFTVR_COLOR_FIX` (left in the shell it applies to
  every later run). In Git Bash, `JASNA_SWIFTVR_COLOR_FIX=none jasna ...`
  passes it for one command.
- View smoothing is GPU work on jasna's side only, so nothing about it is
  Windows-specific. Its cost was within noise, as on Linux.

Measurements (Windows 11, RTX 5060 Ti 16 GB. The inputs differ from Linux, with
10661 frames at 480p and 6242 at 1080p, so do not compare wall times with the
Linux table above. The whole-GPU peak is the maximum of `nvidia-smi` polled
every 0.5 s and includes 1.4 to 1.6 GB of desktop residents. GB is MiB / 1024.
Output frame counts matched the input in every run):

| Input | Configuration | Wall time | Whole-GPU peak |
|-------|---------------|-----------|----------------|
| 480p | primary only | 114 s | 3.7 GB |
| 480p | `swiftvr-inline` scale 4 | 671 s | 12.8 GB |
| 480p | `swiftvr-inline` scale 2 | 280 s | 9.8 GB |
| 480p | `swiftvr-inline` scale 4, color fix none | 613 s | 13.0 GB |
| 480p | `swiftvr-inline` scale 4, `--no-swiftvr-accel` | 1287 s | 15.6 GB |
| 1080p | `swiftvr-inline` scale 4 | 477 s | 13.2 GB |
| 1080p | `swiftvr-inline` scale 2 | 189 s | 10.4 GB |
| 1080p | `swiftvr-inline` scale 2, `--swiftvr-view-window 0` | 197 s | 10.4 GB |
| 1080p | `flashvsr-inline` scale 2 (earlier run) | 463 s | 11.0 GB |
| 1080p | `flashvsr-inline` scale 4 tiles 1 (earlier run) | 1555 s | 15.5 GB |

- Against FlashVSR inline on the same machine at 1080p, scale 2 is about 2.4x
  and scale 4 about 3.3x faster.
- Color correction costs about 9.5% of wall time (671 s vs 613 s), more than
  Linux's 4.6% and in line with FlashVSR's Windows figure (about 10%).
- The `--no-swiftvr-accel` row was measured with 2.9 GB of desktop residents,
  the FlashVSR rows with 3.1 GB (scale 2) and 1.3 GB (scale 4).
- Color drift (480p, against the primary-only output, measured every 10th frame
  over a shared mask of the pixels the uncorrected output changed; the mask
  differs from the Linux gate, so do not compare absolute values): per-channel
  median |Δmean| went from 3.98 uncorrected to 0.71 with wavelet at scale 4 and
  0.66 at scale 2 (pass; wavelet is below uncorrected in over 94% of frames).
- Visual check (user): the scale 4 and scale 2 outputs (with and without view
  smoothing), side by side with FlashVSR inline on the same inputs, showed no
  problems.

### Offline 3-phase measurements (Windows)

On the same machine and inputs, every offline 3-phase run completed with output
frame counts matching the input.

- The Phase 2 driver is started with `<repo>\.venv\Scripts\python.exe`
  (`PYTHONUTF8=1`), and acceleration passed its checks and was enabled. Ready
  took 6 to 18 s for the model load and 8 to 24 s for warmup.
- Per-process VRAM from `nvidia-smi` is unavailable under WDDM, so the GPU-wide
  value was split by phase interval, and each phase's peak is its increment over
  the residents measured before the run.
- **Phase 2's peak is within 0.2 GB of the Linux stand-in (no
  `expandable_segments`)**: 9.2 GB at scale 4 in FP8 and 6.3 GB at scale 2 (the
  same at 480p and 1080p). Scale 4 in bf16 is 14.0 GB.
- The GPU-wide increment is 2.1 to 2.5 GB below inline, and the wall clock is
  1.4 to 1.7 times inline's.
- With a lossless encode and `--fp8-recon` passed to Phase 1, all 10661 frames
  are bit-identical to inline (as on Linux). `tune=lossless` goes through without
  an error.
- Stopping the driver mid-Phase 2 keeps the bundle and prints the resume hint;
  re-running the same command skips the finished clips, completes, and the output
  is bit-identical to the uninterrupted run. The `cudaErrorDevicesUnavailable`
  seen on Linux on an immediate re-run did not occur.
- FlashVSR's offline mode (`flashvsr`) still completes as before (no regression).

| Input | Configuration | Wall clock | Residents | GPU-wide peak | Phase 1 / 2 / 3 increment | Phase 1 / 2 / 3 time | Bundle |
|-------|---------------|------------|-----------|---------------|---------------------------|----------------------|--------|
| 480p | scale 4 | 913 s | 2.0 GB | 11.2 GB | 2.7 / 9.2 / 0.2 GB | 215 / 633 / 59 s | 30 GB |
| 480p | scale 2 | 484 s | 1.9 GB | 8.2 GB | 2.8 / 6.3 / 0.3 GB | 224 / 224 / 33 s | 8.0 GB |
| 480p | scale 4, `--no-swiftvr-accel` | 1392 s | 1.4 GB | 15.4 GB | 2.7 / 14.0 / 0 GB | 213 / 1120 / 54 s | — |
| 1080p | scale 4 | 691 s | 0.4 GB | 9.6 GB | 3.2 / 9.2 / 0.7 GB | 159 / 490 / 38 s | 22 GB |
| 1080p | scale 2 | 323 s | 0.4 GB | 6.7 GB | 3.2 / 6.3 / 0.7 GB | 150 / 147 / 22 s | 5.9 GB |
| 480p | `flashvsr` scale 2, `--flashvsr-accel` | 962 s | 0.4 GB | 6.4 GB | 2.7 / 6.0 / 0.3 GB | 224 / 702 / 32 s | — |

The residents dropped from 2.0 GB to 0.4 GB during the runs, so compare the
increments. Scale 4 in FP8 on a 12 GB-class card (RTX 4070, 5070) comes to about
11.2 GB with 2 GB of residents (the 480p scale 4 row is exactly that condition).
