#!/bin/bash
# Cache the target and reference video latents for IC-LoRA training with Musubi Tuner (LTX-2 fork).
# The text-encoder cache is NOT made here: Gemma 3 12B does not fit a 16 GB card even in 8-bit, so run
# write_te_cache.py on the target cache folders afterwards instead of ltx2_cache_text_encoder_outputs.py.
#
# Environment: MUSUBI_DIR (Musubi Tuner checkout with its .venv), LTX_CHECKPOINT (the ltx-core checkpoint,
# for example sulphur_distil_bf16.safetensors; its VAE encodes the latents), DATASET_TOML (see
# dataset.toml.example). Extra arguments go to ltx2_cache_latents.py. Rerunning skips existing files.
# 1328 train + 24 val pairs with their references took 52 minutes on an RTX 5080.
set -eu
: "${MUSUBI_DIR:?set MUSUBI_DIR to the Musubi Tuner checkout}"
: "${LTX_CHECKPOINT:?set LTX_CHECKPOINT to the ltx-core checkpoint}"
: "${DATASET_TOML:?set DATASET_TOML to the dataset config}"
cd "$MUSUBI_DIR"
exec .venv/bin/python ltx2_cache_latents.py --dataset_config "$DATASET_TOML" --ltx2_checkpoint "$LTX_CHECKPOINT" \
  --device cuda --vae_dtype bf16 --ltx2_mode video --reference_frames 121 --reference_downscale 1 \
  --skip_existing --atomic_cache_writes "$@"
