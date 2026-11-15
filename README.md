> **The only places to get a genuine License code for the supporter models are [Ko-fi](https://ko-fi.com/kruk2), [Unifans](https://app.unifans.io/c/kruk2), and for China the seller in the [Chinese README](README.zh.md#在闲鱼购买).**
>
> **支持者模型的正版许可证码只能通过 [Ko-fi](https://ko-fi.com/kruk2)、[Unifans](https://app.unifans.io/c/kruk2) 或 [闲鱼](README.zh.md#在闲鱼购买) 获取。**

[**English**](README.md) | [日本語](README.ja.md) | [中文](README.zh.md)

# <img width="32" src="https://github.com/Kruk2/jasna/blob/main/assets/jasna-logo.png?raw=true" /> Jasna 

Jasna restores mosaics in videos and images. It is free, has a simple app
and a command line, and runs on your GPU.

> ### ⚙️ This is the `+modi` fork of Jasna
>
> ## ⚠️ Built from the public source only. LTX restores video here only with a model of your own.
>
> **This fork contains only the public part of Jasna's source.** The private `jasna/protection` submodule (license activation and the decryption of the encrypted supporter models) is **not part of it**, so **UNet 4x and SD 1.5 do not work in this build, and LTX only with a model of your own**. The LTX weights ship encrypted (`.enc`) and only the official release decrypts them: with the release weights `--restoration-model-name ltx` and the LTX modes of the GUI fail here, and `--ltx-trial` only measures speed without restoring anything. For the release LTX weights or the supporter models use the official [**Kruk2/jasna**](https://github.com/Kruk2/jasna) release. jasna's LTX inference path itself is public and loads plaintext safetensors: [docs/en/ltx_lora_training.md](docs/en/ltx_lora_training.md) builds a model for it from a public LTX-2.3 checkpoint and an IC-LoRA trained on mosaic/clean pairs. A video the release restored with LTX can then be sharpened here with `--ltx-crop-vsr` ([docs/en/ltx_crop_vsr.md](docs/en/ltx_crop_vsr.md)). Everything else (detection, BasicVSR++ restoration, RTX/TVAI secondary restoration, the Segment Editor, VR180, streaming, post-export actions) works normally.
>
> A modified build on top of upstream [Kruk2/jasna](https://github.com/Kruk2/jasna) **v0.12.3**. It adds:
>
> - **Restoration models**: **SeedVR2 + LoRA** primary restoration; **FlashVSR**, **SwiftVR** and **SwiftVR distill** secondary restoration; two extra **RTX Super Res** knobs (`--rtx-strength`, `--rtx-highbitrate`).
> - **Sharpening of LTX outputs** (`--ltx-crop-vsr`): takes a video the official release restored with LTX together with its source video, re-runs LTX's public mosaic scan, upscales the mosaic areas 2x with SwiftVR, SwiftVR distill or RTX Super Res and composites them back with LTX's own blending. It is aimed above all at low-resolution inputs, whose restored crops LTX leaves soft. This is the one way to work with LTX results in this build. See [docs/en/ltx_crop_vsr.md](docs/en/ltx_crop_vsr.md).
> - **Sharpening of LTX windows inside an LTX run** (`--secondary-restoration rtx-super-res` with an LTX model): the same RTX Super Res deblur on every window canvas in LTX's compose pass, before LTX's own blending. Upstream 0.12.1 has the option too, on the enlarged windows only and without deblur by default; this fork keeps every window and deblur `low`. Only usable with LTX weights this build can load (plaintext safetensors, not the encrypted release weights). See [docs/en/changes_vs_upstream.md §3.7](docs/en/changes_vs_upstream.md#37-sharpening-ltx-windows-in-the-run---secondary-restoration-rtx-super-res).
> - **Training your own LTX model** (`scripts/ltx_lora_training/`): converts a public LTX-2.3 checkpoint (for example Sulphur-2-base) into jasna's LTX file layout and trains an IC-LoRA on mosaic/clean window pairs with the Musubi Tuner LTX-2 fork, so the LTX path runs on a model this build can load. It restores mosaics, softer than the release model. See [docs/en/ltx_lora_training.md](docs/en/ltx_lora_training.md).
> - **Frame generation** beyond upstream's 2x: `--frame-generation` 4x, the RIFE backend next to RTX Video Frame Generation, the RTX quality mode, and the standalone `jasna-framegen` for already-restored videos.
> - **Backends**: an experimental **torchcodec video backend**, an experimental **cuDNN FP8 restoration backend**, and the **TensorRT-RTX** flavor (engine builds in seconds instead of minutes).
> - **GUI**: **A/B model comparison** in the Segment Editor and a **restoration checkpoint** selector in the Video Player.
>
> - **Full list of changes vs upstream:** [docs/en/changes_vs_upstream.md](docs/en/changes_vs_upstream.md), by layer, with the status of each feature; the per-feature guides are under [`+modi` guides](#all-guides) below.
> - **Run from source.** Set this fork up with the build guides ([Linux](docs/en/building_linux.md), [Windows](docs/en/building_windows.md)); an experimental Windows bundle can be built with `scripts/build_nuitka.py` ([docs/en/frozen_build.md](docs/en/frozen_build.md)).
> - **Source (this fork/branch):** [sh202603/jasna @ `modi`](https://github.com/sh202603/jasna/tree/modi)

<img width="1200" height="907" alt="image" src="https://github.com/user-attachments/assets/d59a914b-482d-4f37-ae72-5c59eb5dc9bb" />

## Contents

**Start here**

- [What Jasna can do](#what-jasna-can-do)
- [What you need](#what-you-need)
- [Quick start](#quick-start)
- [Help and community](#help-and-community)
- [Supporting the project](#supporting-the-project) (supporter models)

**Advanced**

- [How Jasna works](#how-jasna-works)
- [Command line](#command-line)
- [All guides](#all-guides)
- [Benchmarks](#benchmarks)
- [Licensing](#licensing)
- [Acknowledgments](#acknowledgments)

## What Jasna can do

- **Restore mosaics in videos.** Only the mosaic areas change. [Quick start](#quick-start)
- **Pick fast or detailed.** BasicVSR++ is fast and free; LTX is slower with more detail (supporters). [Choosing models](docs/en/models.md#restoration-model)
- **Test LTX for free.** Trial mode shows how fast LTX runs on your PC. [LTX options](docs/en/models.md#ltx-options)
- **Use LTX only where it helps.** Big mosaics get LTX, the rest BasicVSR++. [Mosaic scanning](docs/en/segments.md#mosaic-scanning)
- **Restore only some scenes.** Choose ranges with a preview in the Segment Editor. [Segment Editor](docs/en/segments.md#segment-editor-gui)
- **Pause and continue.** Long LTX jobs keep their progress. [Pause and continue](docs/en/models.md#pause-and-continue)
- **Sharpen restored areas.** UNet 4x, RTX Super Res or Topaz. [Secondary restoration](docs/en/models.md#secondary-restoration)
- **Upscale the whole video** to 1080p, 1440p or 4K. [Upscale](docs/en/advanced_processing.md#upscale-the-whole-video)
- **Smoother motion.** Double the frame rate on RTX 40 and 50. [Frame generation](docs/en/gui.md#frame-generation)
- **VR180 videos**, side by side or top/bottom. [VR180](docs/en/vr180.md)
- **Still images** with SD 1.5 (supporters). [Images](docs/en/models.md#still-image-restoration-sd-15)
- **Watch while it restores**, in the app or in a browser. [Video Player](docs/en/gui.md#video-player), [Streaming](docs/en/streaming.md)
- **Many videos at once.** Queue, presets and an action when the queue is done. [Using the GUI](docs/en/gui.md#the-queue)
- **Output extras.** Color LUT, sharpening, 60→30 FPS, play while it's being made. [Advanced processing](docs/en/advanced_processing.md)
- **Command line and rented GPUs.** [CLI](docs/en/cli.md), [Cloud GPUs](docs/en/cloud.md)

## What you need

- **NVIDIA** GTX 16-series / RTX 20-series or newer, with a recent driver
  (610+ on Windows, 580+ on Linux). GTX 10-series and older don't work.
- Or **AMD** Radeon RX 6000 or newer (experimental). RX 7000/9000 are tested;
  RX 6000 is untested.
- A folder on a local disk, with no Chinese, Japanese, accented or other
  non-English characters in its path.

<details>
<summary>More details</summary>

- NVIDIA: your GPU needs compute capability 7.5 or higher in NVIDIA's
  [GPU table](https://developer.nvidia.com/cuda/gpus). Data-center GPUs like
  A100, H100 and B200 don't work.
- AMD: see the table below. UNet 4x secondary restoration is supported.
- Disk format: NTFS on Windows; ext4, Btrfs, XFS or ZFS on Linux. Network disks
  are not supported. Use an SSD; on a hard disk (HDD) Jasna starts and loads
  models slowly.
- LTX needs an NVIDIA RTX or AMD Radeon RX 7000/9000 GPU (AMD integrated GPUs
  untested), at least 8 GB VRAM and 16 GB RAM; on Linux, kernel 5.8 or newer.
  See [What LTX needs](docs/en/models.md#what-ltx-needs).

| AMD GPU | Jasna | LTX |
| --- | --- | --- |
| Radeon RX 9060/9070, Radeon AI PRO R9700 | yes | yes (Windows: beta) |
| Radeon RX 7600–7900, Radeon PRO W7000 | yes | yes (Windows: beta) |
| Radeon 740M/760M/780M/880M/890M, Ryzen AI Max (8050S/8060S) | yes, untested | yes, untested |
| Radeon RX 6000, Radeon PRO W6000 | yes, untested | no |
| Radeon 610M/660M/680M/820M/840M/860M | yes, untested | no |
| Radeon RX 5000, Radeon PRO W5000, RX Vega, RX 500 and older | no | no |

</details>

## Quick start

1. Download the release for your system and GPU.
2. Unpack it into a folder with no Chinese, Japanese, accented or other
   non-English characters in the path.
3. Start the app:
   - Windows: double-click `jasna.exe`. On AMD, first install
     [AMD's driver for ROCm 10.1](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-ROCM-10-01.html)
     (**26.10.41.05**, [download .exe](https://drivers.amd.com/drivers/amd-software-adrenalin-edition-26.10.41.05-win11-b-204865.exe))
     and reboot. You don't need to install ROCm.
   - Linux NVIDIA: run the `jasna` file.
   - Linux AMD: first install AMD's AMF video runtime once. We tested
     [AMF 26.20](https://repo.radeon.com/amf/26.20/):

     ```bash
     curl -fLO https://repo.radeon.com/amf/26.20/amf_installer_26.20.zip
     unzip amf_installer_26.20.zip
     sudo bash amf_installer_26.20.sh
     ```

     Accept AMD's license when prompted, then run `run_jasna_amd.sh`.
     If using Distrobox, install AMF inside the same container as Jasna.
4. Add a video, choose settings and click the **▶** button at the bottom.

**Your first job is slow.** On NVIDIA, Jasna first prepares the models for
your exact GPU. This takes about **15–60 minutes**, only once. Close other
apps and don't use the PC during this time. When you update Jasna, extract it
into a new, empty folder (not over the old one) and copy the `model_weights`
folder there, so you don't wait again. LTX may
still ask to download its model again after an update. On AMD, only UNet 4x
needs this preparation, on its first use.

Every setting has a tooltip: move the mouse over the ⓘ icon. Out of GPU
memory (VRAM)? Lower **Max Clip Size**, for example from `90` to `60`. See
[Tuning](docs/en/tuning.md).

## Help and community

- [App guide](docs/en/gui.md): the queue, presets, output names, the Video
  Player and activation.
- Problem? Click **Logs** at the bottom of the app, then **Export** to save a
  log file. Attach it when you ask for help.

## Supporting the project

Jasna is free. Your support pays for training new models, mostly GPU rental.
Supporters get a License code for the supporter models:

| Support | Models |
| --- | --- |
| **$15 USD** (Ko-fi: **14 EUR**) or more in total | **UNet 4x**: makes restored areas sharper.<br>**SD 1.5**: experimental model for still images. |
| **$25 USD** (Ko-fi: **23 EUR**) | **UNet 4x** and **SD 1.5**, plus<br>**LTX**: much slower than BasicVSR++, but more detail. Includes future updates of this LTX model. |

Supporters who gave **$15 USD or more on or before 14 October 2026** get LTX for free.

**Before you choose LTX:**

- Run LTX in **Trial mode (speed test)** on your PC. It is free and shows if your GPU can
  run LTX and how long it takes. The video is not restored. See
  [What LTX needs](docs/en/models.md#what-ltx-needs).

### Where to support

License codes come **only** from here:

- [Ko-fi](https://ko-fi.com/kruk2)
- [Unifans](https://app.unifans.io/c/kruk2)
- For China: see the [Chinese README](README.zh.md#在闲鱼购买)

Anyone else selling Jasna, codes or modified files is a scam: their keys don't
work, and their files can run hidden code on your PC. Download Jasna only from
GitHub.

### How to activate

1. Support on one of the pages above. You get a **License code**.
2. In Jasna, open **Activate Jasna**, enter the License code, choose
   **This is my PC** or **Rented or shared PC** and press **Activate**.

One License code works on **3 devices at the same time**. On your own PC with
a TPM 2.0 security chip (most Windows 11 PCs), Jasna then works offline.
Elsewhere it needs internet while it uses supporter models. Details and help:
[Activate supporter models](docs/en/gui.md#activate-supporter-models). Rented
GPU: see [Cloud GPUs](docs/en/cloud.md).

Using a License code means you accept the
[supporter license terms](SUPPORTER_TERMS.md).

## Advanced

### How Jasna works

Jasna finds each mosaic, follows it from frame to frame, restores only that
area and puts it back. The rest of the picture is not touched. Almost
everything runs on the GPU. LTX is slower but can add more real detail; see
[Choosing models](docs/en/models.md#restoration-model).

### Command line

The command line can do most of what the app does (not the Video Player,
LTX seed previews, mask suggestions or interactive image mode).

```bash
# One video
jasna --input input.mp4 --output output.mkv

# One image
jasna --input photo.png --output restored.png

# A whole folder
jasna --input input_folder --output output_folder
```

`jasna --help` lists all options. See the [CLI reference](docs/en/cli.md).

### All guides

- [Using the GUI](docs/en/gui.md): queue, presets, output names, Video
  Player, activation.
- [Choosing models](docs/en/models.md): detection models, BasicVSR++ or LTX,
  secondary restoration, SD 1.5 for images.
- [Restoring only parts of a video](docs/en/segments.md): Segment Editor,
  mosaic scan, LTX only for big mosaics.
- [VR180 videos](docs/en/vr180.md): how side-by-side and top/bottom VR are handled.
- [Tuning VRAM and GPU usage](docs/en/tuning.md): clip size, overlap, what to
  do when VRAM runs out.
- [Advanced processing](docs/en/advanced_processing.md): denoise, 60→30 FPS,
  color LUT, sharpening, upscaling the whole video, encoder quality, settings
  saved in the video, actions after the queue.
- [Streaming](docs/en/streaming.md): watch in the browser or in Stash.
- [Cloud GPUs](docs/en/cloud.md): Jasna on a rented GPU (RunPod, Vast).
- [CLI reference](docs/en/cli.md): all command line options.
- [Running from source](docs/en/development.md): for developers. From source
  you can use the free BasicVSR++; activation and supporter models need the
  official release.

**`+modi` guides** (features of this fork; English and Japanese only):

- [Changes vs upstream](docs/en/changes_vs_upstream.md): every modi feature by
  layer, its status, the compatibility matrix and the rebase history.
- [SeedVR2 primary restoration](docs/en/seedvr2.md), [FlashVSR](docs/en/flashvsr.md)
  and [SwiftVR / SwiftVR distill](docs/en/swiftvr.md) secondary restoration:
  external diffusion models in their own venvs (CLI only).
- [Sharpening an LTX output](docs/en/ltx_crop_vsr.md): `--ltx-crop-vsr` for a
  video the official release restored with LTX.
- [Training your own LTX model](docs/en/ltx_lora_training.md): convert a public
  LTX-2.3 checkpoint and train an IC-LoRA for jasna's LTX path (Linux verified).
- [Frame generation](docs/en/frame_generation.md): `--frame-generation` 2x/4x, the RIFE and
  RTX backends, `jasna-framegen`.
- [torchcodec backend](docs/en/torchcodec_backend.md),
  [FP8 restoration backend](docs/en/fp8_recon.md),
  [TensorRT-RTX flavor](docs/en/tensorrt_rtx.md): experimental backends.
- [Building on Linux](docs/en/building_linux.md),
  [Building on Windows](docs/en/building_windows.md),
  [Frozen Windows bundle](docs/en/frozen_build.md): from-source setup and
  packaging.

### Benchmarks

On an RTX 5090 (Linux), Jasna 0.9.1 restored a 3:22 1080p video with
BasicVSR++ in 34 seconds. Test settings, GPU memory and older versions are in
the [full benchmark report](benchmarks/2026-07-26_release_matrix.md); newer
results are in [benchmarks](benchmarks/).

### Licensing

Jasna's public application source is AGPL-3.0. Models and bundled dependencies
keep their own terms. See [licensing and source availability](LICENSING.md),
[model notices](assets/THIRD_PARTY_MODELS.md) and the
[release source correspondence](RELEASE_SOURCES.md).

### Acknowledgments

- **Lada**: Jasna is inspired
  by Lada and in some places based on it. The `mosaic_restoration_1.2` model
  used by Jasna was trained by ladaapp, the author of Lada.
- **ZeLeFans**: for the VR detection model and the detailed analysis of mosaic
  shapes and VR projection.
