# Training your own LTX restoration model (`+modi`)

This fork cannot decrypt the official LTX weights, but jasna's LTX inference path is
public code and loads plaintext safetensors from a model folder. This guide builds a model
of your own for it: a public LTX-2.3 checkpoint converted into jasna's file layout, plus an
IC-LoRA (a LoRA conditioned on a reference video) trained on mosaic/clean window pairs so
the model learns to restore. The scripts live in
[`scripts/ltx_lora_training/`](../../scripts/ltx_lora_training/README.md); the dataset dump
lives in the author's lada-ex fork.

What to expect. A stock checkpoint converts and runs, but restores nothing: given the mosaic
crop as its reference it reproduces that crop. With the IC-LoRA from this recipe the model
restores mosaics, visibly softer than the official release model in the author's checks; it
is a way to get a working LTX model on a build without the protection module, not a
replacement for the release weights. The trained model itself is not published; the tools
and the recipe are.

Everything here was run on Linux x86-64 (RTX 5080, 16 GB) and is unverified on Windows and
on ARM64; see [Known limits](#known-limits).

## Requirements

| | |
| --- | --- |
| GPU | 16 GB VRAM. Training peaked at 12.6 GB with the settings in `train_ic_lora.sh`; the text-encoder step holds Gemma 3 12B partly in VRAM (`--gpu-budget`, default 11 GiB) and the rest in host RAM. |
| System RAM | 64 GB recommended. Training used 32 GB; the trainer swaps 36 of the 48 transformer blocks to host memory. |
| Disk | About 43 GB for the Sulphur-2 checkpoint, 23 GB for Gemma 3 12B, 1.5 GB for the LTX-2.5 VAE, 4.6 GB for 1300 training pairs, 3 GB of caches, 13.4 GB per converted model (26.3 GB in bf16). |
| jasna | The dev install of this fork ([Linux](building_linux.md), [Windows](building_windows.md)). The converter, `encode_prompt.py` (it uses `transformers`, already a jasna dependency) and the window check run in its venv. |
| Trainer | [Musubi Tuner, LTX-2 fork](https://github.com/AkaneTendo25/musubi-tuner) (branch `ltx-2-dev`, used at commit `9dc9abc`), set up in its own venv: `uv sync --extra cu130 --python 3.12` in the checkout. |
| Dataset tooling | A [lada-ex](https://codeberg.org/sh202603/lada-ex) checkout on branch `feat/seedvr2-lora-primary` with its venv and a `mosaic_removal_vid` dataset, the clean-video crops lada-ex trains BasicVSR++ on (how to build one: lada-ex `docs/training_and_dataset_creation.md`). |
| Models to download | `sulphur_distil_bf16.safetensors` from [SulphurAI/Sulphur-2-base](https://huggingface.co/SulphurAI/Sulphur-2-base); `vae/ltx-2.5-video-vae-bf16.safetensors` from [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (gated, accept the license to download); [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) (gated, Gemma terms). See [Licenses](#licenses). |

## How the pieces fit

```
lada-ex mosaic_removal_vid ──(1) ltx-ic-lora-dump-pairs.py──> gt/lq 121-frame window pairs, 512 px, fixed caption
                                                                      │
caption ──(2) encode_prompt.py (Gemma 3 + LTX-2.3 projection/connector)──> prompt_context.safetensors
                                                                      │               │
                               (3) cache_latents.sh + write_te_cache.py <──────────────┘
                                                                      │
                               (4) train_ic_lora.sh (Musubi Tuner, IC-LoRA v2v, rank 32) ──> LoRA checkpoints
                                                                      │
LTX-2.3 checkpoint ──(5) convert_ltx_to_jasna.py --lora ──> transformer file ──(5) make_bundle.py + LTX-2.5 VAE ──> model folder
                                                                      │
                               (6) ltx_window_infer.py on val pairs; jasna --restoration-model-name ltx-8step --restoration-model-path <folder>
```

Why this works without touching jasna's code: Sulphur-2-base is a full fine-tune of
LTX-2.3 with no structural change, and the video blocks of LTX-2.3 and LTX-2.5 have the same
tensor names and shapes (2.3 has FFN biases, which jasna treats as optional; 2.5 has a
keyframe embedding, which the converter adds as zeros). The 2.5 VAE encoder is identical to
2.3's, so latents from a 2.3-trained model decode through jasna's 2.5 diffusion decoder.
jasna conditions each window by appending the reference tokens (the VAE-encoded mosaic crop)
after the target tokens at the same positions, with sigma 0 on the reference; that is
exactly the IC-LoRA reference layout the public trainers use. Details and the measurements
behind the recipe are in the author's design memo (not part of this repository).

## Step 1: Dataset

The pairs come from lada-ex's `mosaic_removal_vid` dataset: clean clips with their synthetic
mosaic placed by lada's own degradation code. The dump script cuts them into jasna's
training geometry so that training sees what inference sees:

- the clip is treated as the full frame and jasna's own `track_camera` is run on the
  mosaic block mask (dilated by 20 px, as jasna dilates detections), yielding 121-frame
  windows at hop 88, each cropped to the 512 px canvas with `crop_to_canvas`;
- `lq` (the reference) is the mosaic plus lada's degradation recipe applied on the canvas,
  `gt` (the target) is the clean crop; `mask` holds the mosaic block mask on the canvas;
- areas outside the clip's real content are reflect-filled, the same way jasna letterboxes;
  windows with more than 35% such fill are dropped (`--max-outside`);
- a fixed caption is written next to every `gt` clip (`--caption`; the default is the one the
  author used). It must be the same text you encode in step 2.

```bash
cd <lada-ex checkout>
JASNA_ROOT=<jasna checkout> .venv/bin/python scripts/dataset_creation/ltx-ic-lora-dump-pairs.py \
    --meta-dir datasets/mosaic_removal_vid/train/crop_unscaled_meta --num-clips 0 \
    --min-scene-side 512 --output-dir datasets/ltx_ic_lora_pairs/train
JASNA_ROOT=<jasna checkout> .venv/bin/python scripts/dataset_creation/ltx-ic-lora-dump-pairs.py \
    --meta-dir datasets/mosaic_removal_vid/val/crop_unscaled_meta --num-clips 0 \
    --min-scene-side 512 --max-windows-per-clip 1 --output-dir datasets/ltx_ic_lora_pairs/val
```

Each run writes `gt/`, `lq/`, `mask/`, `params/` (per-pair geometry and degradation
record), `dataset.json`, `manifest.json` and a `musubi_dataset.toml` template. From the
author's dataset this gave about 1300 training pairs (eight workers, about 10 s per pair) and
115 validation pairs. Make a small validation subset for the trainer: copy the `gt/` and
`lq/` clips of a couple of dozen validation pairs into `val24/` (the trainer runs a
validation pass over it every 250 steps; 24 clips take about 2.5 minutes).

## Step 2: The caption as `prompt_context`

jasna has no text encoder. The fixed caption is encoded once, the way ltx-core does it for
LTX-2.3 (Gemma 3 12B hidden states, the checkpoint's own `text_embedding_projection` and
`video_embeddings_connector`), and the result is baked into the model file as
`prompt_context`. Training uses the same tensor as its text cache (step 3), so training and
inference condition on the same constant.

```bash
cd <jasna checkout>
.venv/bin/python scripts/ltx_lora_training/encode_prompt.py \
    --gemma-dir <gemma-3-12b-it> --source <sulphur_distil_bf16.safetensors> \
    --prompt "$(cat <lada-ex>/datasets/ltx_ic_lora_pairs/train/gt/<any pair>.txt)" \
    --output prompt_context.safetensors --hidden-cache gemma_hidden.safetensors
```

Gemma takes `--gpu-budget` (default 11 GiB) of VRAM and the rest of its 23 GB in host RAM.
`--hidden-cache` keeps the stacked hidden states so the projection and connector can be
rerun without reloading Gemma.

## Step 3: Latent and text caches

Copy `scripts/ltx_lora_training/musubi/dataset.toml.example`, replace the `/path/to/...`
entries (the `train/gt`, `train/lq`, `val24/gt`, `val24/lq` folders and four cache folders),
then:

```bash
export MUSUBI_DIR=<musubi-tuner checkout> LTX_CHECKPOINT=<sulphur_distil_bf16.safetensors> DATASET_TOML=<your dataset.toml>
scripts/ltx_lora_training/musubi/cache_latents.sh
.venv/bin/python scripts/ltx_lora_training/musubi/write_te_cache.py --prompt-context prompt_context.safetensors \
    <cache>/train <cache>/val
```

`cache_latents.sh` encodes target and reference clips with the checkpoint's VAE
(1300 pairs plus references: 52 minutes on the RTX 5080; about 3 GB of cache). `write_te_cache.py` writes the
text cache every latent needs from `prompt_context` instead of running Musubi's Gemma
pass, which does not fit in 16 GB even in 8-bit. Point it at the target cache folders only
(`train`, `val`), not the `_ref` folders.

## Step 4: Training

```bash
export OUT_DIR=<folder for runs>
scripts/ltx_lora_training/musubi/train_ic_lora.sh <run-name> 2500
```

The recipe (see the script header for every flag): IC-LoRA `v2v` preset, rank 32, constant
learning rate 1e-4, adamw8bit, bf16 mixed precision with fp8 base weights, 36 blocks swapped
to host memory (host-to-device only), bounded activation offload, two data-loader workers,
a checkpoint every 500 steps and a validation pass every 250. On the RTX 5080 this ran at
11 to 12 s per step with 12.6 GB of VRAM and 32 GB of RAM; 2500 steps took about 8.5 hours.
Fewer swapped blocks run out of VRAM in the backward pass; eight workers take 58 GB of RAM.

What the author's runs taught about the schedule, for choosing a checkpoint:

- With the constant learning rate the run diverged after about 2600 steps (training loss
  jumped, validation loss rose), so stop at 2500 or save often and pick a checkpoint by the
  window check below rather than by the final one.
- Restarting from a checkpoint with a lower learning rate, a decaying schedule, a low-noise
  timestep shift or a soft loss mask around the mosaic gave smoother outputs, not sharper
  ones. Treat such variations as experiments, not as the default.
- The validation loss tracks the direction of training but not visual sharpness; judge
  checkpoints on decoded video.

Checkpoints are written to `$OUT_DIR/<run-name>/<run-name>-stepNNNNNNNN.safetensors` and
`<run-name>.safetensors` at the end. Resuming with `--resume` keeps the saved optimizer state
and ignores a changed learning rate on the command line.

## Step 5: Convert and assemble the model folder

```bash
cd <jasna checkout>
.venv/bin/python scripts/ltx_lora_training/convert_ltx_to_jasna.py \
    --source <sulphur_distil_bf16.safetensors> --prompt-context prompt_context.safetensors \
    --lora $OUT_DIR/<run-name>/<run-name>-step00002500.safetensors --output my-ltx-int8.safetensors
.venv/bin/python scripts/ltx_lora_training/make_bundle.py --transformer my-ltx-int8.safetensors \
    --vae <ltx-2.5-video-vae-bf16.safetensors> --model-dir <model folder>
```

The converter renames the video-side tensors, drops the audio, text and VAE branches, fuses
the LoRA into the weights in fp32, adds the zero keyframe embedding and `prompt_context`,
stores the distilled 8-step sigma schedule as metadata and quantizes every block linear to
ConvRot W8A8 (the format the release weights use; `--quant bf16` keeps bf16 weights at twice
the size). Conversion of the 46 GB checkpoint takes under a minute on the RTX 5080 and streams
the output, so RAM is not a constraint. `make_bundle.py` checks the VAE against jasna's
recorded tensor table and links the files under the names of the 8-step slot. Since the
v0.12.1 base the decoder file is named `vae-decoder-full.safetensors` (it was
`vae-decoder.safetensors`): a folder made earlier needs that name added, for example
`ln vae-decoder.safetensors vae-decoder-full.safetensors` in the folder.

Run jasna with the folder:

```bash
jasna --input in.mp4 --output out.mp4 --restoration-model-name ltx-8step --restoration-model-path <model folder>
```

Copying the three files into `model_weights/ltx-restore/` makes them the default for
`--restoration-model-name ltx-8step` (and the GUI's 8-step LTX choice) in a dev install.
`--secondary-restoration rtx-super-res` sharpens the LTX windows inside the run
([changes_vs_upstream.md §3.7](changes_vs_upstream.md#37-sharpening-ltx-windows-in-the-run---secondary-restoration-rtx-super-res));
on low-resolution footage it recovers some of the softness of a model from this recipe.

## Step 6: Checking a checkpoint

```bash
.venv/bin/python scripts/ltx_lora_training/ltx_window_infer.py --model-dir <model folder> \
    --out-dir <out> --inputs <lada-ex>/datasets/ltx_ic_lora_pairs/val/lq/<pair>.mp4 ...
```

The script feeds a whole `lq` clip as one window, exactly as training did (no detection,
no camera), and writes the decoded window; compare it with the pair's `gt` clip and with the
output of other checkpoints. About 7 s per window once the model is loaded. Do not judge
checkpoints by running the full pipeline on these synthetic clips: jasna's detector splits
a synthetic mosaic into several tracks, so the windows no longer match the training geometry.
Real footage through the pipeline is the final check.

## Known limits

- **Verified on Linux x86-64 only.** The scripts are plain Python and bash; Windows and
  ARM64 are untested. jasna's INT8 LTX path runs its block linears through its Triton
  kernels; a platform without them needs a bf16 model (`--quant bf16`, 26 GB). On the
  author's 16 GB card the bf16 model ran through jasna's torch.compile path with most blocks
  streamed from host RAM, about twice as slow per window as the INT8 model, with output
  matching the INT8 model's closely. Whether that path builds on Windows on ARM is unknown.
- **The base is the distilled 8-step model.** The LoRA was trained on Sulphur's distilled
  checkpoint and runs in jasna's 8-step slot. Training on the dev checkpoint for the
  `ltx-undistilled` slot (15 steps with STG) is untried.
- **Native LTX blocks need a matching CUDA toolkit.** When jasna logs that the native blocks
  failed to build, the run falls back to the torch.compile path and is slower; point
  `CUDA_HOME` at a toolkit that supports your GPU.
- **Quality is below the release model.** Sharper results came from decoder-side work
  and from `--secondary-restoration rtx-super-res`, not from the training knobs listed in
  step 4.

## Licenses

The scripts here are this fork's code (AGPL-3.0) and contain no weights. The models you
download bring their own terms, which you accept when you download them:

- **Sulphur-2-base** ships the [LTX-2 Community License Agreement](https://huggingface.co/SulphurAI/Sulphur-2-base/blob/main/LICENSE.txt)
  unchanged: it is a derivative of Lightricks LTX-2.3, and that license requires
  derivatives to carry it.
- **LTX-2.5 VAE** is under the [LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x) (gated on Hugging Face).
- **Gemma 3 12B** is under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (gated).

Points of the LTX license that matter for this recipe, as the author reads it (not legal
advice): use, fine-tuning and derivatives are allowed for any purpose within its use
restrictions; entities with annual revenue of 10 million USD or more need a paid commercial
license; a model you train is a "Derivative of LTX-2", so if you ever distribute it you must
ship the full license text with it and pass on its use restrictions. The converter records
the source file and the license name in the output's metadata.
