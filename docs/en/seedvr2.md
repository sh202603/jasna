# SeedVR2+LoRA primary restoration (`+modi`)

`--restoration-model-name seedvr2` swaps the primary restorer from BasicVSR++ to
**SeedVR2 3B (one-step SR diffusion) + LoRA** — a quality mode. Detected 256px
mosaic crops go straight into the diffusion forward pass (scale=1; this replaces
the primary restorer, it is not a secondary upscaler). The LoRA is a custom
model trained to treat mosaic as cell-averaging degradation to remove
(rank 16, ~90 MB, [sh202603/lada-seedvr2-lora](https://huggingface.co/sh202603/lada-seedvr2-lora),
AGPL-3.0). The SeedVR2 base weights (Apache-2.0) are auto-downloaded on first
load by the [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
checkout you supply.

Wall clock is roughly 6x BasicVSR++, proportional to mosaic screen time
(measured in lada-ex: 17–21 output crop-fps vs 100–157 fps, RTX 5080). Being
generative, the output is sharp but adds *plausible* detail — it is not signal
recovery. The default model stays `basicvsrpp`; this feature is strictly
opt-in, and the basicvsrpp path's output is bit-identical to before the
feature was added.

## Requirements

- NVIDIA GPU with 16 GB VRAM (measured whole-pipeline peaks: ~12.6 GB at 480p
  on this fork's smoke run; lada-ex measured ~12.7 GB at 480p / ~13.9 GB at 1080p)
- A `ComfyUI-SeedVR2_VideoUpscaler` checkout with its own venv (ComfyUI itself
  is not needed)
- Disk: ~7.3 GB base weights + ~90 MB LoRA
- **No BasicVSR++ TensorRT engines needed** — the first-run compile covers the
  detection engine only

## Setup

```bash
# 1. Checkout + dedicated venv (nothing is installed into jasna's venv)
git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git ~/seedvr2_videoupscaler
cd ~/seedvr2_videoupscaler
uv venv --python 3.13 .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt

# 2. LoRA into jasna's model_weights/
wget -O model_weights/lada_seedvr2_lora_v2.pt \
  https://huggingface.co/sh202603/lada-seedvr2-lora/resolve/main/lada_seedvr2_lora_v2.pt

# 3. Run (base weights auto-download into the checkout on first load)
jasna --input in.mp4 --output out.mp4 \
  --restoration-model-name seedvr2 --seedvr2-repo ~/seedvr2_videoupscaler
```

Worker startup (model load + LoRA injection + warm-up) takes 1–2 minutes with
cached weights. The worker is resident and survives across the files of a
folder input, so that cost is paid once per session.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--seedvr2-repo` | — | Path to the checkout. Required when seedvr2 is selected |
| `--seedvr2-python` | `<repo>/.venv/bin/python` | The venv's Python (`Scripts\python.exe` on Windows) |
| `--seedvr2-model-dir` | `<repo>/models/SEEDVR2` | Base weights directory |
| `--seedvr2-dit` | `seedvr2_ema_3b_fp16.safetensors` | DiT weights filename |
| `--seedvr2-lora` | `<model_weights>/lada_seedvr2_lora_v2.pt` | LoRA checkpoint (swap-in point for fine-tunes) |
| `--seedvr2-lora-rank` | `16` | LoRA rank (must match the checkpoint) |
| `--seedvr2-window` | `33` | Sliding-window length; must be 4n+1 |
| `--seedvr2-overlap` | `9` | Cross-fade overlap between windows |
| `--seedvr2-color-fix` | `lab` | Per-clip color correction (`none`/`lab`/`wavelet`) |

## How it works

- **Windowing**: the model is trained on short 4n+1 frame sequences, so clips
  are inferred in 33-frame windows at stride 24, and the 9-frame overlap
  between adjacent windows is cross-faded with a linear ramp (quantization
  happens once, after blending). Clip-boundary seams are covered by jasna's
  standard overlap+discard splitting and cross-fade.
- **Zero padding**: the 256px crop padding is zero-fill for this restorer
  instead of reflect, matching the LoRA's training distribution. The
  basicvsrpp path keeps reflect padding unchanged.
- **Color correction**: diffusion output can drift the crop's overall color,
  so LAB statistics (mean/variance) are transferred per clip using the input
  mosaic crops themselves as the reference. `wavelet` transplants the
  reference's low frequencies instead and can re-introduce the mosaic cell
  grid into the output, which is why it is not the default.
- **Determinism**: the seed and the CUDA allocator (cudaMallocAsync) are
  pinned; identical input reproduces identical output.
- **Error policy**: a failed clip here means the mosaic stays in the output,
  so there is no pass-through fallback. A worker error or death triggers one
  respawn+retry of the same clip; a second failure stops the job. Frame-count
  or shape mismatches error immediately (no retry).

## Constraints and combinations

| Combination | Behavior |
|---|---|
| `--vr-mode sbs`/`sbs-fisheye`, or `auto` detecting VR | Startup error (jasna's VR crops are projection-conditioned — unverified for the LoRA) |
| `--frame-gen` | Startup error (no VRAM budget next to the resident worker; run it as a separate pass) |
| `--secondary-restoration flashvsr-inline` | Startup error (two resident workers exceed a 16 GB card) |
| `--secondary-restoration flashvsr` (offline) | **Allowed.** The phases never co-reside, so SeedVR2 restoration + FlashVSR 4x composes into the maximum-quality stack |
| `--secondary-restoration tvai` | Warning only (sharpening on top of a diffusion primary risks over-enhancement) |
| `unet-4x` / `rtx-super-res` | Allowed |
| `--stream` | Warning only (mosaic-dense stretches run at the SeedVR2 rate and can stall the player) |
| `--fp8-recon` / `--compile-basicvsrpp` / `--restoration-model-path` | BasicVSR++-only, inert (fp8-recon and model-path warn) |

## Fine-tuning

The training harness (pair dump + LoRA training) is not bundled with jasna:
the dump depends on the BasicVSR++ training degradation pipeline, which
jasna's vendored mmagic subset (inference-only) does not carry. Fine-tune in
[lada-ex](https://github.com/sh202603/lada-ex) under
`scripts/training/seedvr2_lora/` and swap the resulting checkpoint in via
`--seedvr2-lora`. See lada-ex's `docs/seedvr2_setup.md` for the recipe.

## Known limitations

- Generative output adds plausible detail; it is not recovery of the original
  signal.
- VR modes are rejected at startup until evaluated.
- Tracking of numz repo internals is confined to the single worker file
  (`jasna/restorer/seedvr2_lora_worker.py`, kept verbatim-identical with
  lada-ex), but upstream compatibility breaks will require worker-side fixes.
