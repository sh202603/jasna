# Post-processing CLI `jasna-postprocess` (`+modi`)

`jasna-postprocess` is a separate command that applies only whole-frame post-processing to a
finished video. It runs no mosaic detection and no restoration. There are two kinds of
post-processing, and you can ask for either or both:

- **Whole-frame upscaling** (`--upscale`): the `span`, `realplksr`, `rtx-super-res` and `dlss5`
  upscalers of jasna's own `--upscale`, and `swiftvr`, which only this command has.
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
```

Passing neither `--upscale` nor `--frame-generation` is an error. When `--input` is a folder,
`--output` is the output folder and every video in it is processed (`--output-pattern` means the
same as in the main CLI; the default is `{original}_out`). Images in the folder are skipped.

A video that is already `--upscale-height` tall is not upscaled. If no frames are generated
either, it is not re-encoded: a folder batch skips it, a single file ends with an error.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--upscale` | `none` | `span`, `realplksr`, `rtx-super-res`, `dlss5`, `swiftvr`. |
| `--upscale-height` | `2160` | `1080`, `1440`, `2160`. The width keeps the aspect ratio. |
| `--upscale-strength` | `1.0` | 0 to 1. Lower values mix in a bicubic resize. |
| `--upscale-rtx-*`, `--upscale-dlss5-*` | as in the main CLI | See [cli.md](cli.md). |
| `--frame-generation` | `none` | `2x`, `4x`. |
| `--frame-generation-backend` | `rife` | `rife`, `rtx`. The main CLI defaults to `rtx`; this is the one difference. |
| `--frame-generation-rtx-mode` | `medium` | `low`, `medium`, `high`. |
| `--frame-generation-model-path` | `model_weights/rife.pth` | RIFE weights. |
| `--codec` | `hevc` | `hevc`, `av1`. |
| `--encoder-settings` | (the main CLI's defaults) | Same format as the main CLI. |
| `--device`, `--fp16`, `--batch-size`, the `--video-backend` family, `--log-level`, `--no-progress`, `--disable-ffmpeg-check` | | Same meaning as in the main CLI. |

The SwiftVR options are listed under [SwiftVR on whole frames](#swiftvr-on-whole-frames---upscale-swiftvr).

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

## Behavior and limits

- `--upscale swiftvr` cannot be combined with `--frame-generation` (startup error): a 4K frame
  generator next to the SwiftVR worker has not been measured. Run two passes when you need both.
- There is no resume. A long `swiftvr` job that stops starts over.
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
- Tests: `tests/test_postprocess_cli_*.py`, `test_clip_upscale_writer.py`,
  `test_swiftvr_frame_worker.py`, `test_upscale_swiftvr.py` (all CPU).
