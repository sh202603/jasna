# Using kukkiri's LTX weights (`+modi`)

[kukkiri](https://github.com/treehog2/kukkiri) publishes `kiyo-ltx-v1`, LTX-2.5 weights
fine-tuned to restore mosaics, as plaintext safetensors. This fork cannot read the official
release's LTX weights (they are encrypted), but jasna's LTX inference path is public code, so
these weights run under `--restoration-model-name ltx` once they are converted to jasna's file
layout. This page is the walkthrough.

The weights are not part of jasna. They are a derivative of LTX-2.x under the LTX-2.x Community
License; get them from their publisher ([License and distribution](#license-and-distribution)).

Everything here was run on Linux x86-64 (RTX 5080, 16 GB) and Windows (RTX 5060 Ti, 16 GB).

## Requirements

| | |
| --- | --- |
| GPU | An NVIDIA GPU that runs jasna's INT8 LTX path. Checked with 16 GB. |
| Disk | 14.8 GB for the download and 13.4 GB for the converted transformer. The VAE is a hard link and costs nothing. |
| jasna | A development install of this fork ([Linux](building_linux.md), [Windows](building_windows.md)). A built distribution (frozen build) cannot use these weights ([Known limits](#known-limits)). |

## 1. Download the weights

They are in `ltx-v1/` of <https://huggingface.co/treehog/kukkiri>, with no sign-in.

```bash
mkdir -p ~/kukkiri-ltx-v1 && cd ~/kukkiri-ltx-v1
for n in transformer vae prompt; do
  curl -L -O https://huggingface.co/treehog/kukkiri/resolve/main/ltx-v1/kiyo-ltx-v1-$n.safetensors
done
sha256sum kiyo-ltx-v1-*.safetensors
```

| File | Size (bytes) | SHA-256 |
| --- | --- | --- |
| `kiyo-ltx-v1-transformer.safetensors` | 13,364,075,432 | `9712079e1b0ee20a9a47efbd0278cc4fb0eb7b88f9571e9d4d9fdfec70ccc94e` |
| `kiyo-ltx-v1-vae.safetensors` | 1,452,269,922 | `685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d` |
| `kiyo-ltx-v1-prompt.safetensors` | 8,388,704 | `46f8652e075d12b501f6a21bda80e373dd956aa0f1ce897ca0c82ff7b21af511` |

`kiyo-ltx-v1-lora.safetensors` in the same folder is not used.

## 2. Convert

Run from the jasna checkout with jasna's venv.

```bash
.venv/bin/python scripts/ltx_lora_training/convert_kukkiri_ltx.py ~/kukkiri-ltx-v1 model_weights/ltx-kukkiri
```

The output folder gets three files, named like jasna's 2-step model (`--restoration-model-name ltx`).

| File | Contents |
| --- | --- |
| `ltx-restore-alpha1-distill2-int8.safetensors` | The converted transformer, with kukkiri's prompt embedding as `prompt_context` and the sampler settings as metadata |
| `vae.safetensors` | kukkiri's VAE: a hard link, or a copy where linking fails |
| `vae-decoder-full.safetensors` | An empty file. This VAE takes no tuned decoder, but the model folder needs the name |

The script ends by printing the largest relative error the conversion adds to a layer's output.
About 0.014 is normal (0.0142 on Linux, 0.0144 on Windows). It takes 15 seconds on an RTX 5080.

## 3. Run

```bash
.venv/bin/python -m jasna --input in.mp4 --output out.mp4 \
    --restoration-model-name ltx --restoration-model-path model_weights/ltx-kukkiri
```

The other LTX options (`--ltx-large-canvas`, `--ltx-overlap`, `--ltx-seed`,
`--secondary-restoration rtx-super-res` and so on) work as usual. `--ltx-fast` does not
([Known limits](#known-limits)).

### In the GUI

The GUI has no field for a model folder: it reads the LTX model from `model_weights/ltx-restore/`.
Convert into that folder, or put the three files there.

```bash
.venv/bin/python scripts/ltx_lora_training/convert_kukkiri_ltx.py ~/kukkiri-ltx-v1 model_weights/ltx-restore
```

Pick "LTX (Standard Quality)" (the 2-step model) as the restoration model. The other LTX
qualities and "fast" (NVFP4) have no files here, so a run with them does not start. Leave the VAE
encoder and decoder on "auto".

## What the conversion does

kukkiri's transformer has the tensors of jasna's LTX model files (48 blocks and the top layers).
Five things differ; the converter and two runtime changes bridge them.

| | kukkiri | jasna (upstream's models) | Bridged by |
| --- | --- | --- | --- |
| Tensor names | `model.diffusion_model.transformer_blocks.N.*` | `blocks.N.*` | Renamed at conversion |
| INT8 layout | Plain INT8 with a scale per row | ConvRot W8A8 (INT8 of the weight rotated along its inputs by a Hadamard matrix) | Dequantized, rotated and quantized again at conversion |
| Prompt embedding | `video_encoding` in its own file | `prompt_context` inside the transformer file | Embedded at conversion |
| Frame rate of the temporal positions | 30 | 24 | Written as the `position_fps` metadata, read by the runtime |
| VAE | LTX-2.5's conv VAE (`CausalVideoAutoencoder`) | Diffusion VAE (`CausalDiffusionVAE`) | A conv decoder in the runtime |

The sampler settings are those of kukkiri's `ltx.py`: sigmas 1.0 / 0.725 / 0.0 (2 steps), no CFG,
no STG. jasna reads them from the model file's metadata, so the converter only writes them.

The two runtime changes leave the behaviour with upstream's model files as it was.

- **`position_fps`** (`jasna/ltx/sampler.py`, `jasna/ltx/transformer.py`): when the model file's metadata has `position_fps`, the temporal positions use that frame rate; without it they use upstream's 24. The key is this fork's own.
- **Conv VAE** (`jasna/models/ltx_vae/conv_video_decoder.py`, `loader.py`): for a `CausalVideoAutoencoder` config the existing encoder is used, and the decoder is a convolutional one ported from ltx-core. It decodes a whole window in one pass.

## Status

- **Linux** (RTX 5080, 2026-11-27, v0.12.4 base): the bundled test clip (1080p, 30 fps, 10 s) and a 30 s 1080p clip whose crops exceed 512 px (899 frames, 9 windows) ran to the end with the input's frame count. The LTX part of the 30 s clip took 40 s on the default 512 px canvas (denoise 0.85 s per window-step, decode 0.99 s per window) and 102 s at 768 px with `--ltx-large-canvas` (2.37 s and 2.35 s). Peak VRAM was 13.3 GB in both. LTX keeps blocks resident in whatever VRAM is free (37 of 48 here), so that figure is not what the model needs.
- **No effect on upstream-format models**: a self-trained model (diffusion VAE) run under the same conditions with the code before and after this change gave bit-identical output.
- **Windows** (RTX 5060 Ti, 2026-11-27, v0.12.3 base): eight clips from 480p to 1080p and 24 to 59.94 fps (10 to 355 s) ran to the end with the input's frame count and duration. Denoising took about 2.05 s per step. By the user's eye it restored better than their own IC-LoRA model.
- **Works with the diffusion VAE too**: the output of kukkiri's transformer also decodes with LTX-2.5's diffusion VAE. On the same clip the result is 45 dB PSNR from the conv decoder's, in the same time. Replacing `vae.safetensors` and `vae-decoder-full.safetensors` in the model folder with the diffusion VAE's files selects that path. Which looks better has not been compared.
- Tests: `tests/test_ltx_position_fps_modi.py`, `test_ltx_conv_vae_modi.py`, `test_convert_kukkiri_ltx_modi.py`.

## Known limits

- **Not in a frozen build.** Only a development install reads plaintext model files.
- **No `--ltx-fast` (NVFP4).** The converter writes the INT8 file only.
- **The conv decoder does not split a window.** It has neither the tiling nor the retry after an out-of-memory error that the diffusion decoder has. A 16 GB GPU had room for the 768 px canvas; smaller GPUs have not been checked.
- **The window is longer than in training.** kukkiri trains on 97-frame clips, jasna runs 121-frame windows. Stills showed no defects; the effect over time has not been examined.
- **A second quantization error.** The published weights are INT8 already, so the converted ones carry the rounding of both.
- For the GUI, the model detection, the checks before a start and a run from the folder the GUI reads were verified. A run driven from the window has not been.

## License and distribution

- `kiyo-ltx-v1` is a derivative of LTX-2.x under the [LTX-2.x Community License](https://huggingface.co/treehog/kukkiri/blob/main/ltx-v1/NOTICE.md). So are the converted files.
- Neither the weights nor the converted files are part of jasna's repository or distributions.
- The converter records the source file name and the license in the output's metadata.
