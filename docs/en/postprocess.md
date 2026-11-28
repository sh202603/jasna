# Post-processing CLI `jasna-postprocess` (`+modi`)

`jasna-postprocess` is a separate command that applies only whole-frame post-processing to a
finished video. It runs no mosaic detection and no restoration. There are two kinds of
post-processing, and you can ask for either or both:

- **Whole-frame upscaling** (`--upscale`): the `span`, `realplksr`, `rtx-super-res` and `dlss5`
  upscalers of jasna's own `--upscale`, and `swiftvr` and `relayvsr` (jasna's own `--upscale` runs
  them too, see [In jasna's own `--upscale`](#in-jasnas-own---upscale)).
- **Frame generation** (`--frame-generation 2x|4x`): RIFE or RTX Video Frame Generation; see
  [frame_generation.md](frame_generation.md).

It is `jasna-framegen` (up to `0.12.3+modi`) renamed and extended. The old name is gone, and the
arguments now carry the main CLI's names (see [Moving from the old command](#moving-from-the-old-command)).

## When to use it

The main CLI's `--upscale` sits in the output stage of the restoration pipeline, so it only runs
together with a restoration. Use this command to upscale afterwards when:

- You want to upscale a restored video without restoring it again.
- You restored with a setup the main CLI refuses next to `--upscale`: `seedvr2`, the offline and
  inline `flashvsr` / `swiftvr`, `--ltx-crop-vsr`, `--segments`, `--ltx-auto-split`.
- You want to redo the output with other upscale settings or another codec.

The video is encoded a second time, so write pass 1 at a high bitrate (for example
`--encoder-settings cq=16`).

## Usage

```bash
# Pass 1: restore only
jasna --input in.mp4 --output restored.mkv --encoder-settings cq=16

# Pass 2: 1080p to 2160p
jasna-postprocess --input restored.mkv --output out.mkv --upscale rtx-super-res --upscale-height 2160

# Upscale and generate frames in one run (upscaled first, then generated: the main CLI's order)
jasna-postprocess --input restored.mkv --output out.mkv \
    --upscale rtx-super-res --upscale-height 1440 \
    --frame-generation 2x --frame-generation-backend rtx

# SwiftVR on whole frames
jasna-postprocess --input restored.mkv --output out.mkv \
    --upscale swiftvr --upscale-height 2160 --swiftvr-repo ~/SwiftVR

# RelayVSR on whole frames
jasna-postprocess --input restored.mkv --output out.mkv \
    --upscale relayvsr --upscale-height 2160 --relayvsr-repo ~/RelayVSR
```

Passing neither `--upscale` nor `--frame-generation` is an error. When `--input` is a folder,
`--output` is the output folder and every video in it is processed (`--output-pattern` means the
same as in the main CLI; the default is `{original}_out`). Images in the folder are skipped.

A video that is already `--upscale-height` tall is not upscaled. If no frames are generated
either, it is not re-encoded: a folder batch skips it, a single file ends with an error.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--upscale` | `none` | `span`, `realplksr`, `rtx-super-res`, `dlss5`, `swiftvr`, `relayvsr`. |
| `--upscale-height` | `2160` | `720`, `1080`, `1440`, `2160`. The width keeps the aspect ratio. |
| `--upscale-strength` | `1.0` | 0 to 1. Lower values mix in a bicubic resize. |
| `--upscale-rtx-*`, `--upscale-dlss5-*` | as in the main CLI | See [cli.md](cli.md). |
| `--frame-generation` | `none` | `2x`, `4x`. |
| `--frame-generation-backend` | `rife` | `rife`, `rtx`. The main CLI defaults to `rtx`; this is the one difference. |
| `--frame-generation-rtx-mode` | `medium` | `low`, `medium`, `high`. |
| `--frame-generation-model-path` | `model_weights/rife.pth` | RIFE weights. |
| `--codec` | `hevc` | `hevc`, `av1`. |
| `--encoder-settings` | (the main CLI's defaults) | Same format as the main CLI. |
| `--device`, `--fp16`, `--batch-size`, the `--video-backend` family, `--log-level`, `--no-progress`, `--disable-ffmpeg-check` | | Same meaning as in the main CLI. |

The SwiftVR options are listed under [SwiftVR on whole frames](#swiftvr-on-whole-frames---upscale-swiftvr),
the RelayVSR options under [RelayVSR on whole frames](#relayvsr-on-whole-frames---upscale-relayvsr).

### The per-frame upscalers

`span`, `realplksr`, `rtx-super-res` and `dlss5` are the main CLI's implementations
(`jasna/upscale/`), with the same requirements:

- `span` and `realplksr` need their weights in `model_weights/`. On NVIDIA they compile a
  TensorRT engine per input size, so a folder with mixed resolutions compiles more than once.
- `rtx-super-res` is NVIDIA only and needs `nvidia-vfx`.
- `dlss5` runs on Windows with an RTX 30-series or newer GPU and offers to download its runtime
  the first time.

### Moving from the old command

| `jasna-framegen` | `jasna-postprocess` |
|---|---|
| `--factor 2x` | `--frame-generation 2x` (the default is now `none`) |
| `--backend` | `--frame-generation-backend` |
| `--rtx-mode` | `--frame-generation-rtx-mode` |
| `--model-path` | `--frame-generation-model-path` |

## SwiftVR on whole frames (`--upscale swiftvr`)

[SwiftVR](swiftvr.md) is the one-step diffusion video restorer the secondary restoration runs on
256px mosaic crops. `--upscale swiftvr` runs it on whole frames. It is far slower than the other
upscalers: about 1.9 frames per second for 1080p to 2160p on an RTX 5060 Ti (see
[Measurements](#measurements)). It suits short videos and excerpts.

### Requirements

The same checkout, checkpoint and venv as the SwiftVR secondary restoration; the setup is in
[swiftvr.md](swiftvr.md#requirements).

| Option | Default | Meaning |
|---|---|---|
| `--swiftvr-repo` | (required) | Checkout of the fork `sh202603/SwiftVR`. |
| `--swiftvr-python` | the Python of `<repo>/.venv` | Python of the SwiftVR venv. |
| `--swiftvr-model-dir` | `<repo>/checkpoints` | Checkpoint directory. |
| `--swiftvr-accel` / `--no-swiftvr-accel` | on | FP8 DiT and torch.compile. The worker drops whatever is unavailable, with a warning. |
| `--upscale-swiftvr-tile` | `1.25` | Largest tile in megapixels after upscaling. Decides the VRAM. |
| `--upscale-swiftvr-window` | `89` | Frames restored per call. |

### How it works

SwiftVR's VRAM follows the output pixels of one call, and a whole 4K frame in one call does not
fit a consumer GPU. The `restore_clip()` jasna uses also keeps no state between calls. So the
frame is cut into tiles in space and the video into windows in time, and one call restores one
tile of one window.

- **Tiles**: the frame is divided evenly into rectangles, as few as keep each tile's output
  within `--upscale-swiftvr-tile`; neighbours share at least 96 output pixels. 1080p at 2x is
  4 columns by 2 rows, 8 tiles of 528×576, which is 1.12 times the frame's pixels. Shared pixels
  are mixed with linear weights.
- **Windows**: 89 frames per window, 16 of them shared with the next. Of the shared frames, the
  first 5 come from the earlier window, the last 5 from the later one, and the middle 6 are mixed
  linearly. The start of a window is generated without past context; the overlap keeps that part
  out of the output. The numbers are those of the main pipeline's clip boundaries.
- **Factor**: SwiftVR upscales 2x or 4x only. It generates at 2x when the target height is at
  most 2.5 times the input's, else at 4x, and a bicubic resize makes up the difference. 1080p to
  1440p is generated at 2x and shrunk, 2.25 times the pixels the output has.
- **Color correction**: the blended frame gets a wavelet correction against the bicubic-upscaled
  input frame. It is the function of the SwiftVR secondary restoration and always on
  (`JASNA_SWIFTVR_COLOR_FIX` overrides it for verification).

The tiles and the color correction run in a worker of the SwiftVR venv
(`jasna/restorer/swiftvr_frame_worker.py`); the windows and their crossfade are jasna's writer
(`jasna/upscale/clip_writer.py`). The worker starts once and serves every video of a folder
batch. When the frame size or the factor changes, torch.compile recompiles during that video's
first window.

### Measurements

Windows 11, RTX 5060 Ti 16 GB, 2026-11-17. SwiftVR fork `137de20`, FP8 and torch.compile on,
default tile and window. The whole-GPU peak is the largest `nvidia-smi` reading at 0.5 s
intervals and includes the desktop (about 3.5 GB).

| Input | Output | Factor and tiles | Time per window | Whole-GPU peak |
|---|---|---|---|---|
| 1080p, 180 frames | 3840×2160 | 2x, 4×2 (528×576) | 89 frames: restore 33.7 s + blend and transfer 5.5 s | 15.3 GB |
| 720p, 60 frames | 2560×1440 | 2x, 2×2 (672×384) | 60 frames: restore 10.3 s + blend and transfer 1.9 s | not measured |
| 480p, 60 frames | 2562×1440 | 4x, 3×2 (304×256) | 60 frames: restore 49.5 s (recompilation included) + blend and transfer 2.9 s | not measured |

- 1080p to 2160p advances 73 frames per window of about 39 s: about 1.9 frames per second in
  steady state, or about 16 minutes for one minute of 30 fps video.
- The worker started with a model load of about 6.5 s and a warmup of 7 to 41 s; the spread is
  the torch.compile cache.
- The 15.3 GB peak is close to the 16 GB limit. With a larger desktop footprint, or on a 12 GB
  card, lower `--upscale-swiftvr-tile` (more tiles, slower).
- The frame counts and the audio packet counts of all three outputs match their inputs.
- The 720p and 480p rows ran back to back in one folder batch: the same worker handled the
  change of factor and tile shape.

Seams in the 1080p output, measured:

- **Window boundaries**: the output's frame-to-frame difference divided by the input's is 0.995
  on average (1.009 at most) over the shared frames and 0.997 (1.021 at most) elsewhere.
  Nothing jumps at a boundary. The measure is coarse, dominated by the subject's motion, and does
  not capture a mismatch in texture.
- **Tile boundaries**: the difference from the bicubic-upscaled input (the amount of generated
  detail) is about 9% lower than its surroundings in 3 of the 4 shared bands, and equal in the
  fourth. The two tiles' results are averaged there, so a band holds slightly less detail.

The user reviewed this 1080p to 2160p output by eye and found no problem. Another real video
(720p to 1080p, 1763 frames) ran at 4.3 frames per second, and its result looked good.

## RelayVSR on whole frames (`--upscale relayvsr`)

[RelayVSR](https://github.com/kopperx/RelayVSR) is a streaming video super-resolution model. A
large model (Wan) restores a keyframe every few frames, and a small decoder fills in the frames
between two keyframes. `--upscale relayvsr` runs it on whole frames. On an RTX 5060 Ti, 1080p to
2160p runs at about 1.7 frames per second with the defaults (see
[RelayVSR measurements](#relayvsr-measurements)).

### What RelayVSR needs

RelayVSR pins another torch than jasna (2.10, CUDA 12.8), so it runs in a venv of its own.

1. Clone the fork [sh202603/RelayVSR](https://github.com/sh202603/RelayVSR) (branch
   `windows-support`). Upstream has neither the `build_stream()` jasna calls nor the 2x
   checkpoints.
2. Run `uv sync --locked --extra cu128` in the checkout.
3. Put the checkpoints into `weights/`. Each factor has its own pair; the pair of the factor you
   use is enough.

| Factor | Files | Download |
|---|---|---|
| 2x | `wan_x2.safetensors`, `flash_x2.safetensors` | `uvx hf download lightx2v/RelayVSR_X2 wan_x2.safetensors flash_x2.safetensors --local-dir weights` |
| 4x | `wan.safetensors`, `flash.safetensors` | `uvx hf download kopper/RelayVSR wan.safetensors flash.safetensors --local-dir weights` |

| Option | Default | Meaning |
|---|---|---|
| `--relayvsr-repo` | (required) | Checkout of the fork `sh202603/RelayVSR`. |
| `--relayvsr-python` | the Python of `<repo>/.venv` | Python of the RelayVSR venv. |
| `--relayvsr-weights-dir` | `<repo>/weights` | Where the checkpoints are. |
| `--relayvsr-window-size` | `16` | Keyframe interval + 1. Larger is faster, and drifts further from the input where the motion is fast. |
| `--relayvsr-wan-weights` | `fp8` | `fp8`, `bf16`. `fp8` needs an RTX 40 series or newer GPU; elsewhere the worker warns and runs `bf16`. `bf16` gives RelayVSR's reference output. |
| `--relayvsr-kv-cache` | `auto` | `auto`, `gpu`, `cpu`. Where the Wan K/V cache lives. |
| `--relayvsr-kv-cache-dtype` | `native` | `native`, `int8`. `int8` halves the cache and changes the output slightly. |

### How RelayVSR runs

- **The state spans the video.** RelayVSR keeps the Wan K/V cache until the video ends. Cutting
  the video into windows, as for SwiftVR, would throw that context away, so there are no tiles,
  no overlapping windows and no crossfade: the frames go in one by one, in order.
- **The output comes late.** A frame is emitted once the keyframe after it exists. The first 16
  frames come out together after the 16th was read, then 15 frames per 15 frames (with
  `--relayvsr-window-size 16`). jasna's writer (`jasna/upscale/stream_writer.py`) keeps the
  timestamps of the frames that have not come out yet, in order.
- **Factor.** RelayVSR upscales 2x or 4x only, and the factor belongs to the pair of
  checkpoints. It is chosen as for SwiftVR: 2x when the target height is at most 2.5 times the
  input, else 4x, and a bicubic resize covers the rest.
- **Memory follows the generated pixels.** The K/V cache dominates: about 12 GB in bf16 when
  2160p is generated. `auto` looks at the free VRAM when the video starts and keeps the cache in
  host memory when it does not fit (the same output, slower keyframes).
- **Generating above 2160p.** 720p to 2160p generates 5120x2880 at 4x and shrinks it; 1440p to
  2160p generates the same size at 2x. The cache is about 1.8 times that of 2160p, so such a
  video gets a warning when it starts. These combinations have not been measured.

The worker (`jasna/restorer/relayvsr_frame_worker.py`) starts once in the RelayVSR venv, and a
folder batch shares it across its videos. A video of the other factor makes the worker load the
other pair of checkpoints (about 8 seconds here).

### RelayVSR measurements

Windows 11, RTX 5060 Ti 16 GB, 2026-11-17. RelayVSR fork `77be23e`. The speed is the time the
worker spent on one video, so it includes decoding, the transfer and encoding. The GPU peak is
the maximum of `nvidia-smi` polled every 0.5 s and includes the desktop (about 3.0 GB).

151 frames of 1920x1080 to 3840x2160 (2x):

| Settings (changes from the defaults) | Speed | K/V cache | GPU peak |
|---|---|---|---|
| none (`fp8`, `auto`, `native`, window 16) | 1.69 fps | host memory | 13.2 GiB |
| `--relayvsr-window-size 32` | 2.38 fps | host memory | 11.9 GiB |
| `--relayvsr-kv-cache-dtype int8` | 0.72 fps | GPU (chosen by `auto`) | 15.6 GiB |
| `--relayvsr-kv-cache gpu --relayvsr-kv-cache-dtype int8` | 0.73 fps | GPU | 15.6 GiB |
| the same with `--relayvsr-window-size 32`, `--upscale-strength 0.8` | 0.86 fps | GPU | 15.5 GiB |

- **The defaults are the fastest here.** The three runs with the K/V cache on the GPU hit the
  16 GB limit and ran slower than the defaults. That includes the run with `int8` alone: the
  smaller cache made `auto` choose the GPU. `auto` only sees the free VRAM at the start of the
  video, so what is allocated later (jasna's encoder, for one) and a desktop that holds much VRAM
  put its estimate off. The `fp8`, `gpu`, `int8` combination the RelayVSR README recommends for
  16 GB cannot be recommended through jasna on this machine.
- **A larger window is faster and drifts further from the input.** The PSNR of the output,
  shrunk to 1080p, against the input was 34.3 dB on average (minimum 26.8 dB) with window 16 and
  29.7 dB (minimum 20.4 dB) with window 32. The drop is where the motion is fast. These are
  numbers of one clip; which one looks better has not been judged.
- **Folder input.** 48 frames each of 1440x1080 (2x, 1.56 fps) and 720x540 (4x, 2.12 fps) ran
  one after the other on the same worker, across the change of factor. Both generate 2880x2160;
  the peak was 13.9 GiB.
- **Startup.** Loading the checkpoints took about 7 seconds, and about 10 seconds passed from
  spawning the worker to the start of the first video.
- **The worker does not change a pixel.** For 33 frames of 360x270, running RelayVSR's
  `build_stream()` directly and going through the worker gave the same value for every pixel of
  every frame.
- The frame count and the audio packet count of every output match the input. Compared one frame
  off, the PSNR falls to 23.1 dB, so the frames are in the input's order.

### Compared with SwiftVR

On the same day and machine, both upscaled 180 frames of a restored native 1080p video to 2160p,
both encoded with `cq=16`. RelayVSR's GPU peak is lower than in the table above because the
desktop held about 1.75 GiB this time.

| | SwiftVR (defaults) | RelayVSR (defaults) | RelayVSR window 32 | RelayVSR `bf16` |
|---|---|---|---|---|
| Speed (without startup) | 1.81 fps | 1.74 fps | 2.38 fps | 0.79 fps |
| Startup (load and warmup) | 34.9 s | 7.1 s | 7.1 s | 6.4 s |
| Whole run | 155 s | 119 s | 91 s | 244 s |
| GPU peak | 13.3 GiB | 12.1 GiB | 12.3 GiB | 15.2 GiB |
| Peak host memory of the worker | 19.5 GiB | 17.7 GiB | 17.7 GiB | 19.6 GiB |
| Agreement with the input (PSNR of the shrunk output, mean / minimum) | 37.3 / 32.9 dB | 36.4 / 33.8 dB | 33.8 / 18.5 dB | 36.3 / 33.8 dB |
| Sharpness (Laplacian variance, relative to bicubic) | 41.9 | 8.1 | 7.5 | 8.3 |

- **With the defaults, speed and VRAM are about the same.** The difference is the startup:
  RelayVSR has no torch.compile warmup and starts about 28 seconds sooner.
- **Both use close to 20 GiB of host memory.** RelayVSR keeps the K/V cache there; SwiftVR keeps
  the tiles of one window on the CPU.
- **SwiftVR adds about five times the high frequencies.** This is the largest difference in the
  numbers. The agreement with the input is about the same.
- **With window 32 some frames drift far from the input.** The minimum fell to 18.5 dB. The
  default of 16 is recommended.
- **`bf16` is slow on 16 GB.** The VRAM filled up to the limit and the speed more than halved.
  Its numbers are about those of `fp8`.

The user reviewed the outputs of this comparison by eye and judged SwiftVR's and RelayVSR's
results to be on the same level.

## In jasna's own `--upscale`

`jasna --upscale swiftvr` and `jasna --upscale relayvsr` put the same worker and writer at the output
of the restoration pipeline. Restoration and upscaling take one run, with no intermediate video and
no second encode.

```bash
jasna --input in.mp4 --output out.mp4 \
    --upscale relayvsr --upscale-height 1080 --relayvsr-repo ~/RelayVSR
jasna --input in.mp4 --output out.mp4 \
    --upscale swiftvr --upscale-height 1080 --swiftvr-repo ~/SwiftVR
```

- **Options.** `--swiftvr-repo`, `--swiftvr-python`, `--swiftvr-model-dir`, `--swiftvr-accel` and
  `--relayvsr-*` are the arguments of the secondary restoration. `--upscale-swiftvr-tile` and
  `--upscale-swiftvr-window` mean what they mean in this command.
- **Not combinable.** `--secondary-restoration` (anything but `none`) and `--frame-generation`: the
  worker shares the GPU with the restoration, so nothing else is put beside it. The limits of
  `--upscale` in general (`--segments`, `--ltx-auto-split`, `--stream`, `seedvr2`, `--ltx-crop-vsr`)
  apply as well.
- **Worker lifetime.** It starts for the first video that is upscaled and is reused for the later
  videos of a folder input.
- **GUI.** Not offered; CLI only.

### Measured beside the restoration

Linux, RTX 5080 16 GB, 2026-11-28. BasicVSR++ primary without a secondary restoration, a 640x480
input of 1,950 frames (65 s), worker defaults. Wall clock is process start to exit, engine and model
loading included. The whole-card peak is the maximum of `nvidia-smi` polled every 0.5 s and includes
the desktop (about 2.3 GiB).

| `--upscale` | Output | Wall clock | Whole-card peak |
|---|---|---|---|
| none | 640×480 | 11.5 s | 5.2 GiB |
| `rtx-super-res` | 1440×1080 | 12.4 s | 5.6 GiB |
| `swiftvr` | 960×720 | 92.7 s | 14.4 GiB |
| `swiftvr` | 1440×1080 | 71.6 s | 14.5 GiB |
| `relayvsr` | 1440×1080 | 64.7 s | 13.0 GiB |
| `swiftvr` | 2880×2160 | 303.7 s | 15.0 GiB |
| `relayvsr` | 2880×2160 | 219.7 s | 14.0 GiB |

- All seven cases finished, each with 1,950 output frames.
- `swiftvr` at 2160p reached the limit of the 16 GB card: jasna moved frames to host memory 21 times
  (697 MiB in total). No other case did. Where there is no headroom, lower `--upscale-swiftvr-tile`
  or split the job into two passes with this command.
- The K/V cache of `relayvsr` sat on the GPU at 1080p and in host memory at 2160p (`auto`).
- `swiftvr` at 720p generates the same 2x frame (1280x960) as at 1080p and shrinks it, so it is not
  faster. Its longer wall clock comes from the worker's warmup in the first case (26 s).
- Not measured: picture quality, 1080p input.

### Measured on Windows

Windows, RTX 5060 Ti 16 GB, 2026-11-28. Inputs of 720x480 with 890 frames and 1920x1080 with 899
frames (about 30 s each), worker defaults. LTX runs kukkiri's weights, with 9 windows for either
input. The whole-card peak is the maximum of the same 0.5 s polling and includes 1.2 to 2.2 GiB of
other applications.

| Primary | Input | `--upscale` | Output | Wall clock | Compose part | Whole-card peak |
|---|---|---|---|---|---|---|
| BasicVSR++ | 720×480 | none | 720×480 | 39 s | | 4.7 GiB |
| BasicVSR++ | 720×480 | `rtx-super-res` | 1080×720 | 33 s | | 5.0 GiB |
| BasicVSR++ | 720×480 | `relayvsr` | 1620×1080 | 117 s | | 13.4 GiB |
| BasicVSR++ | 720×480 | `swiftvr` | 1620×1080 | 167 s | | 13.0 GiB |
| LTX | 720×480 | none | 720×480 | 112 s | 25.2 s | 14.6 GiB |
| LTX | 720×480 | `relayvsr` | 1080×720 | 190 s | 117.1 s | 13.3 GiB |
| LTX | 720×480 | `relayvsr` | 1620×1080 | 173 s | 99.4 s | 13.2 GiB |
| LTX | 720×480 | `swiftvr` | 1620×1080 | 194 s | 121.5 s | 14.1 GiB |
| LTX | 720×480 | `swiftvr` | 3240×2160 | 394 s | 321.8 s | 15.1 GiB |
| LTX | 720×480 | `relayvsr` | 3240×2160 | 398 s | 324.5 s | 15.1 GiB |
| LTX | 1920×1080 | `relayvsr` | 3840×2160 | 534 s | 456.9 s | 15.3 GiB |
| LTX, 768 px windows | 1920×1080 | `relayvsr` | 3840×2160 | 635 s | 483.3 s | 15.5 GiB |

- All twelve cases finished, each with as many output frames as the input.
- The compose part took 2.1 to 2.6 times as long as on Linux (RTX 5080) with the same clip. Against
  the run without upscaling, `relayvsr` at 1080p took 3.9 times as long, like the 4.0 times on Linux.
- `relayvsr` at 2160p beside LTX ran out of memory on Linux with both the 720x480 input (4x) and the
  1920x1080 input (2x), and finished on Windows. The cause of the difference was not looked into.
- `swiftvr` from 1920x1080 to 2160p beside LTX was stopped after 10 minutes (it finished in 277.7 s
  on Linux). It had raised no error.
- The K/V cache of `relayvsr` sat on the GPU beside BasicVSR++ and in host memory beside LTX.
- With a folder input (BasicVSR++ primary, `relayvsr` at 1080p, two videos) the worker started once
  and was reused for the second video.
- The user looked at the outputs of the 720x480 input (BasicVSR++ and LTX primary, `relayvsr` and
  `swiftvr`), found nothing wrong and judged them to be on the same level as the Linux outputs.
  Picture quality was not measured in numbers.

### Beside an LTX primary

With LTX the worker shares the GPU with the VAE decode of the LTX windows in the compose pass. A
16 GB card cannot hold both peaks at once, so:

- **They take turns.** No window decodes while the worker works, and a decode hands its VRAM back
  when it ends (`share_gpu_turn` in `jasna/ltx/compose_pass.py`).
- **RelayVSR's K/V cache.** On a card with less than 20 GB, `--relayvsr-kv-cache auto` keeps it in
  host memory: the worker decides at the start of the video, where it cannot see the VRAM the later
  decodes take. An explicit `gpu` puts it on the GPU.

Same machine and clip, LTX with kukkiri's weights as the primary (22 windows at 512 px), 2026-11-28.

| `--upscale` | Output | Wall clock | Compose part | Whole-card peak |
|---|---|---|---|---|
| none | 640×480 | 84.3 s | 23.2 s | 13.4 GiB |
| `relayvsr` | 1440×1080 | 137.1 s | 80.6 s | 14.0 GiB |
| `swiftvr` | 1440×1080 | 158.3 s | 93.8 s | 14.1 GiB |
| `swiftvr` | 2880×2160 | 367.3 s | 308.6 s | 15.3 GiB |
| `relayvsr` | 2880×2160 | failed | | 15.4 GiB |

- The peak without upscaling is the denoise: LTX fills free VRAM with resident transformer blocks.
- `relayvsr` at 2160p (4x, generating 2560x1920) ran out of memory in jasna at frame 481, with
  6.1 GiB in jasna and 6.9 GiB in the worker. Such a job warns when it starts. Lower the height to
  get 2x, use `swiftvr`, or split the job into two passes with this command.
- Before the two measures above only `relayvsr` at 1080p finished; the other three cases ran out of
  memory.
- Not measured on this machine: 768 px windows (`--ltx-large-canvas`), picture quality. 1080p input
  and 768 px windows were measured on Windows, see [Measured on Windows](#measured-on-windows).

## Behavior and limits

- `--upscale swiftvr` and `--upscale relayvsr` cannot be combined with `--frame-generation`
  (startup error): a 4K frame generator next to the worker has not been measured. Run two passes
  when you need both.
- There is no resume. A long `swiftvr` or `relayvsr` job that stops starts over (RelayVSR's
  `--resume` belongs to RelayVSR's own command and is not available here).
- `relayvsr` has not been verified on Linux.
- The output carries no settings comment tag.
- The main CLI's output options (LUT, sharpen, frame-rate retargeting) are not available.
- CLI only; not in the GUI.
- It imports neither `jasna.pipeline` nor `jasna.protection` (a test guards this).
- `span`, `realplksr` and `dlss5` have not been run through this command either (no weights or
  runtime at hand; their wiring is tested). The frozen build has not been verified since this
  change.

## Implementation

- `jasna/postprocess_cli.py`: arguments, checks, the writer chain. `UpscaleSettings` holds just
  what `session_factory`'s whole-frame upscale functions read from a `SessionConfig`, and is
  passed in its place.
- `jasna/upscale/clip_writer.py`: the `ClipUpscaler` protocol and `ClipUpscaleWriter` (windows,
  overlap, crossfade).
- `jasna/upscale/swiftvr.py`: `SwiftvrWorker` (spawning the worker, the wire),
  `SwiftvrClipUpscaler`, the choice of factor.
- `jasna/restorer/swiftvr_frame_worker.py`: the worker of the SwiftVR venv (tile layout,
  blending, color correction). It loads `swiftvr_inline_worker.py` by path for the model
  handling.
- `jasna/upscale/stream_writer.py`: the `StreamUpscaler` protocol and `StreamUpscaleWriter`
  (frames that come out late, and their timestamps).
- `jasna/upscale/relayvsr.py`: `RelayvsrWorker` (spawning the worker, the wire),
  `RelayvsrStreamUpscaler`.
- `jasna/restorer/relayvsr_frame_worker.py`: the worker of the RelayVSR venv. It hands the
  video's frames to `stream.frames()` as a generator that reads stdin; the wire is defined at the
  top of the file.
- `jasna/restorer/relayvsr_common.py`: the checks of the `--relayvsr-*` paths and the pair of
  checkpoints of each factor.
- Tests: `tests/test_postprocess_cli_*.py`, `test_clip_upscale_writer.py`,
  `test_swiftvr_frame_worker.py`, `test_upscale_swiftvr.py`, `test_stream_upscale_writer.py`,
  `test_upscale_relayvsr.py`, `test_relayvsr_frame_worker.py` (all CPU).
