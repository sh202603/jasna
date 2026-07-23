[**English**](README.md) | [日本語](README.ja.md) | [中文](README.zh.md)

# <img width="32" src="https://github.com/Kruk2/jasna/blob/main/assets/jasna-logo.png?raw=true" /> Jasna 

Jasna is a JAV mosaic restoration tool with a simple GUI, a CLI, a GPU-only processing pipeline, NVIDIA TensorRT and experimental AMD ROCm support, optional secondary restoration models, still-image restoration, and streaming support.

It is inspired by, and in some places based on, [Lada](https://codeberg.org/ladaapp/lada). The `mosaic_restoration_1.2` restoration model used by Jasna was trained by ladaapp, the Lada author.

Jasna is free. Supporters get a key that unlocks the extra models trained for this project: the **unet-4x** secondary upscaler and the experimental **SD 1.5 image restoration** model. See [Supporting the project](#supporting-the-project).

> ### ⚙️ This is the `+modi` fork of Jasna
>
> A modified build on top of upstream [Kruk2/jasna](https://github.com/Kruk2/jasna) v0.8.1, adding **frame generation** (`--frame-gen` 2x/4x), an experimental **torchcodec video backend**, an experimental **FP8 restoration backend**, and **FlashVSR secondary restoration**, among other improvements.
>
> - **Source (this fork/branch):** [sh202603/jasna @ `modi`](https://github.com/sh202603/jasna/tree/modi)
> - **Full list of changes vs upstream:** [docs/en/changes_vs_upstream.md](docs/en/changes_vs_upstream.md)
> - **Scope — public (free) features only.** The supporter models (**unet-4x** and **SD 1.5 image restoration**) ship as encrypted checkpoints unlocked by a supporter key, and the decryption code lives in a private submodule that is **not part of this public fork** — so those models **cannot be downloaded, decrypted, or run here**. The upstream code for them rides along but stays inert. If you want the supporter models, use upstream [**Kruk2/jasna**](https://github.com/Kruk2/jasna) and become a supporter. Everything else (detection, video restoration, RTX/TVAI secondary, the segment editor, VR180, post-export actions, frame generation) works normally.

<img width="1200" height="907" alt="image" src="https://github.com/user-attachments/assets/d59a914b-482d-4f37-ae72-5c59eb5dc9bb" />


## Contents

- [What Jasna Does](#what-jasna-does)
- [`+modi` Additions](#modi-additions)
- [Community](#community)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [First Run](#first-run)
- [Learn More](#learn-more)
- [Benchmarks](#benchmarks)
- [Supporting the Project](#supporting-the-project)
- [TODO](#todo)

## What Jasna Does

- Restores mosaics in video files.
- Restores mosaics in still images with the experimental SD 1.5 image model.
- Detects mosaics with RF-DETR models by default; Lada and ZeLeFans YOLO models are also available.
- Processes side-by-side VR180 videos per eye, with optional fisheye reprojection for detection and restoration.
- Reduces clip-boundary flicker with temporal overlap and crossfade.
- Can further improve quality with optional [secondary restoration models](docs/en/models.md#secondary-restoration) — **unet-4x**, **RTX Super Resolution**, or **Topaz Video AI** — which sharpen restored regions, especially large mosaics, close-ups, and 4K video.
- Can stream restored video to the built-in browser player or a supported Stash fork.

## `+modi` Additions

These features are exclusive to the `+modi` fork. See [docs/en/changes_vs_upstream.md](docs/en/changes_vs_upstream.md) for the full list of changes against upstream.

### Frame Generation (frame-rate up-conversion)

`--frame-gen {2x,4x}` raises the output frame rate by inserting AI-interpolated frames (RIFE) between the source frames. File output only (not `--stream`, not `--segments`); audio timecodes are kept, so duration and sync are preserved. Runs fp16 by default — measured ~1.9x faster than fp32 (1080p 2x, RTX 5060 Ti) with visually identical output.

```bash
jasna --input input.mp4 --output output.mkv --frame-gen 2x
```

Backend via `--frame-gen-backend {rife,rtx}` (`rife` is the default and available now; `rtx` is pending NVIDIA's `nvidia-vfx` release).

A standalone `jasna-framegen` command applies **only** frame generation to an already-restored video (no detection/restoration) — handy for a two-pass workflow (restore first, e.g. with the official binary, then up-convert). It also supports folder input/output with an `--output-pattern` naming template (videos only):

```bash
jasna-framegen --input restored.mkv --output out2x.mkv --factor 2x
jasna-framegen --input in_dir --output out_dir --factor 2x --output-pattern "{original}_2x.mkv"
```

Details: [docs/en/frame_generation.md](docs/en/frame_generation.md).

### Video Backend (experimental)

Jasna decodes and encodes through PyAV (NVDEC/NVENC) by default (`--video-backend native`). An **experimental** [torchcodec](https://github.com/meta-pytorch/torchcodec) backend is available as an alternative:

```bash
jasna --input input.mp4 --output output.mkv --video-backend auto
```

`--video-backend {native,auto,torchcodec}` (default `native`, i.e. unchanged behavior): `auto` uses torchcodec for **decode** where available and falls back to native otherwise; `torchcodec` forces it. `--decode-backend` / `--encode-backend` override each side independently. Since v0.8.0 the native encoder always outputs 10-bit HEVC/AV1, which torchcodec's 8-bit NVENC cannot match, so **torchcodec encode runs only when forced** (`--encode-backend torchcodec`), and only for 8-bit sources with mappable NVENC settings; streaming, `--segments`, `--retarget-high-fps`, and frame generation stay on native. Colorspace metadata is preserved either way. Requires the optional dependency (`pip install "torchcodec>=0.15.0"` from the cu130 wheel index). Details: [docs/en/torchcodec_backend.md](docs/en/torchcodec_backend.md).

### FP8 Restoration Backend (experimental)

`--fp8-recon` runs the BasicVSR++ upsample stage as cuDNN FP8 convolutions instead of the TensorRT FP16 sub-engine:

```bash
jasna --input input.mp4 --output output.mp4 --fp8-recon
```

The main benefit is VRAM: the TensorRT upsample engine's load-time arena (~2.2 GB at the default `--max-clip-size 90`) is never allocated, measured as 1.2–1.7 GB lower peak VRAM across 480p–4K clips. The stage itself also runs ~1.5x faster, though end-to-end fps is unchanged because the pipeline is detection-bound. Output stays visually indistinguishable from the FP16 engine and is bit-deterministic across runs. Requires an FP8-capable GPU (sm89+, i.e. RTX 40 series or newer; the speedup is validated on Blackwell only) and fp16 mode; falls back to the TensorRT engine on any failure. Verified on both Linux and Windows. Details: [docs/en/fp8_recon.md](docs/en/fp8_recon.md).

### FlashVSR secondary restoration (experimental)

`--secondary-restoration flashvsr` upscales each restored 256px mosaic crop to 1024px (4x) with [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) diffusion VSR, recovering texture the primary model leaves blurry on large mosaics, close-ups, and 4K:

```bash
jasna --input in.mp4 --output out.mkv --secondary-restoration flashvsr --flashvsr-repo ~/FlashVSR_plus
```

FlashVSR peaks at 12–16 GB VRAM on its own, so it cannot co-reside with the ~9 GB primary pipeline. It runs **offline in three subprocesses whose peak VRAM never overlaps**: (1) primary restoration → serialize crops to a disk *bundle*, (2) FlashVSR 4x under its own venv, (3) re-blend + encode the final output. You supply the `FlashVSR_plus` checkout, its v1.1 weights, and a **uv-managed standalone Python venv** (system Python can't JIT FlashVSR's Triton attention kernel). File-output only; not compatible with `--stream` / `--frame-gen`. A single-pass variant, `--secondary-restoration flashvsr-inline`, runs FlashVSR inside the streaming pipeline with **no intermediate files** (needs a 16 GB card and a checkout with the tiny-long multi-chunk patch). Details: [docs/en/flashvsr.md](docs/en/flashvsr.md).

### Fragmented MP4 output (`--fmp4`)

`--fmp4` writes `.mp4`/`.mov` output as **fragmented MP4**, so the output file can be opened and played (with audio) while processing is still running, and stays playable if the run is interrupted:

```bash
jasna --input in.mp4 --output out.mp4 --fmp4
```

No measurable throughput cost. Default off (unchanged behavior). Also available as a GUI toggle in the encoding settings. Falls back to a normal MP4 with a warning for `--segments`, `--stream` and the torchcodec encode backend. Details: [docs/en/fmp4.md](docs/en/fmp4.md).

## Community

Join the [SLS Discord](https://discord.gg/uNwQ4mHqgv) for examples, support, and settings discussion. Please don't be too weird.

## Requirements

- An NVIDIA **GTX 16-series / RTX 20-series or newer** GPU. GTX 10-series and older cards (GTX 1050/1060/1070/1080) won't work. Not sure about yours? Check NVIDIA's [GPU table](https://developer.nvidia.com/cuda/gpus) — compute capability 7.5+ is required.
- Nvidia driver **610 or newer** on Windows, **580 or newer** on Linux.
- AMD support is experimental and needs a ROCm-supported GPU.
- Install Jasna into a folder whose path contains only English letters and numbers.

Jasna manages VRAM automatically: when it runs low, waiting frames are temporarily moved to system RAM. No configuration needed.

## Quick Start

1. Download the release package for your OS and GPU vendor.
2. Unzip it into a folder with only English characters in the path.
3. Start the app:
   - Windows: double click `jasna.exe`.
   - Linux NVIDIA: run the `jasna` file.
   - Linux AMD: run `run_jasna_amd.sh`.
4. Add a video or image, choose settings, and start processing.

Every setting in the GUI has a tooltip — hover the ⓘ icon next to it. The
[GUI guide](docs/en/gui.md) tours the rest: queue reordering, presets, output
patterns, and more.

Prefer the command line?

```bash
# Single video
jasna --input input.mp4 --output output.mkv

# Still image
jasna --input photo.png --output restored.png

# Whole folder
jasna --input input_folder --output output_folder
```

Run `jasna --help` for all options, or read the [CLI reference](docs/en/cli.md).

## First Run

The first run is slow because Jasna prepares GPU-specific files for your exact card. On NVIDIA this usually takes **15-60 minutes**; on AMD the preparation is much shorter. It only happens once — the results are cached in `model_weights` and reused on every later run. You can copy them from an older Jasna version to a newer one.

Close other applications, including browsers, and avoid using the PC while this runs.

If you run out of VRAM during processing, reduce **max clip size** first, for example from `180` to `60`. See [Tuning VRAM and GPU usage](docs/en/tuning.md).

## Learn More

- **[Using the GUI](docs/en/gui.md)** — the queue (drag & drop, reordering), presets, output patterns and file conflicts, and other easy-to-miss features.
- **[Choosing models](docs/en/models.md)** — which detection model to pick, sharper results with secondary restoration (unet-4x / RTX Super Resolution / Topaz), and SD 1.5 still-image restoration.
- **[Restoring only parts of a video](docs/en/segments.md)** — the Segment Editor, built-in mosaic scanning, suggesting better masks, and the `--segments` CLI flag.
- **[VR180 videos](docs/en/vr180.md)** — how Jasna handles side-by-side VR and when to use fisheye mode.
- **[Tuning VRAM and GPU usage](docs/en/tuning.md)** — clip size, temporal overlap, model compilation, and what to do when VRAM runs out.
- **[Advanced processing](docs/en/advanced_processing.md)** — denoising, 60→30 FPS export, color LUTs, custom encoder settings, and post-export actions.
- **[Streaming](docs/en/streaming.md)** — watch restored video on the fly in your browser or through Stash.
- **[CLI reference](docs/en/cli.md)** — every command-line option, including output templates, encoder settings per codec, and post-export actions.
- **[Running from source](docs/en/development.md)** — developer setup and build notes.

> **`+modi` build guides:** this fork builds the native GPU libraries and runs Jasna **from source** — there is no public packaged/frozen binary (the packaging tooling lives in the private `jasna/protection` submodule, same as upstream). Step-by-step guides covering the CUDA 13.0 toolchain, native libraries, ffmpeg 8, and TensorRT engine setup:
> - Linux: [docs/en/building_linux.md](docs/en/building_linux.md)（[日本語](docs/ja/building_linux.md)）
> - Windows: [docs/en/building_windows.md](docs/en/building_windows.md)（[日本語](docs/ja/building_windows.md)）

`+modi` feature guides:

- **[Frame generation](docs/en/frame_generation.md)** — RIFE 2x/4x frame-rate up-conversion and the standalone `jasna-framegen` tool.
- **[Video backend](docs/en/torchcodec_backend.md)** — the experimental torchcodec decode/encode backend.
- **[FP8 restoration backend](docs/en/fp8_recon.md)** — the cuDNN FP8 upsample stage with lower peak VRAM.
- **[FlashVSR secondary restoration](docs/en/flashvsr.md)** — offline and inline diffusion 4x upscaling.
- **[Fragmented MP4 output](docs/en/fmp4.md)** — `--fmp4`, output playable during processing.
- **[Frozen build](docs/en/frozen_build.md)** — the experimental Nuitka standalone build.
- **[Changes vs upstream](docs/en/changes_vs_upstream.md)** — the full delta of this fork.

## Benchmarks

RTX 5090 + i9 13900k:

| File                            | Clip (s) | lada 0.10.1 | jasna 0.3.0          | jasna 0.5.0          | **jasna 0.6.2**        |
| ------------------------------- | -------: | ----------: | --------------------:| --------------------:| ----------------------:|
| **ABF-017** (4k, 2h 25min)      | 60       | 02:56:26    | 01:20:49 (2.2x faster) | 01:10:00 (2.5x faster) | — |
| **HUBLK-063** (1080p, 3h 10min) | 180      | 01:34:51    | 44:21 (2.1x faster)  | 37:57 (2.5x faster)  | **30:58 (3.1x faster)** |
| **DASS-570_2m**                 | 30       | 01:08       | 00:30 (2.3x faster)  | 00:24 (2.8x faster)  | **00:20 (3.4x faster)** |
| **NASK-223_Test**               | 30       | 03:12       | 01:18 (2.5x faster)  | 01:02 (3.1x faster)  | **00:58 (3.3x faster)** |
| **test-007**                    | 30       | 01:16       | 00:41 (1.9x faster)  | 00:28 (2.7x faster)  | **00:22 (3.5x faster)** |
| **厚码测试2**                   | 30       | 01:52       | 00:43 (2.6x faster)  | 00:36 (3.1x faster)  | **00:34 (3.3x faster)** |

## Supporting the Project

Support pays for training extra models, mainly GPU rental and compute time for larger datasets. Supporters get a key that unlocks:

- **unet-4x** secondary upscaler for sharper 256->1024 restoration.
- **SD 1.5 image restoration**, the experimental still-image model.

Example results:

- [unet-4x / secondary restoration examples on SLS Discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260)
- [SD 1.5 image restoration examples on SLS Discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1492139124348420106) and [more SD 1.5 examples](https://discord.com/channels/1196376491815092265/1199059436199759943/1516571355317800990)

How to get a key:

1. Contribute **$15 USD or more in total**, across any number of contributions and at any time.
2. After your contribution is processed, your supporter key is sent automatically:
   - **[Unifans](https://app.unifans.io/c/kruk2)**: sent by platform message. There might be a slight delay.
   - **[Buy Me a Coffee](https://buymeacoffee.com/kruk2)**, including **crypto**: sent to the email or handle used for the contribution. The key is tied to that email or handle.

## TODO

Current TODO:

- SeedVR support?
- Continued performance and VRAM improvements.
- Better restoration model.
- Better detection model.
