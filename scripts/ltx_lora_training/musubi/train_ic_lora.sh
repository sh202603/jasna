#!/bin/bash
# IC-LoRA (v2v) training on an LTX-2.3 checkpoint with Musubi Tuner (LTX-2 fork): the recipe behind
# docs/en/ltx_lora_training.md. Rank 32, constant LR 1e-4, adamw8bit, fp8 base weights, 36 blocks swapped to
# host RAM (H2D only), bounded activation offload, 2 data loader workers, validation every 250 steps.
# On an RTX 5080 (16 GB) with 64 GB RAM: 11 to 12 s per step, 12.6 GB VRAM, 32 GB RAM. Fewer swapped blocks
# run out of VRAM in the backward pass; 8 workers take 58 GB RAM.
#
# Usage: train_ic_lora.sh <run-name> <max-steps> [extra ltx2_train_network.py args...]
# Environment: MUSUBI_DIR, LTX_CHECKPOINT, DATASET_TOML (as for cache_latents.sh), OUT_DIR (LoRA files and
# TensorBoard logs go to $OUT_DIR/<run-name>); optional RANK (32), LR (1e-4), BLOCKS_TO_SWAP (36), WORKERS (2),
# SAVE_EVERY (500), VALIDATE_EVERY (250), SEED (42).
# Checkpoints: $OUT_DIR/<run-name>/<run-name>-stepNNNNNNNN.safetensors every SAVE_EVERY steps and
# <run-name>.safetensors at the end. Pass one to convert_ltx_to_jasna.py --lora.
set -eu
: "${MUSUBI_DIR:?set MUSUBI_DIR to the Musubi Tuner checkout}"
: "${LTX_CHECKPOINT:?set LTX_CHECKPOINT to the ltx-core checkpoint}"
: "${DATASET_TOML:?set DATASET_TOML to the dataset config}"
: "${OUT_DIR:?set OUT_DIR to the output folder}"
[ $# -ge 2 ] || { echo "usage: $0 <run-name> <max-steps> [extra args...]" >&2; exit 2; }
run=$1; steps=$2; shift 2
out="$OUT_DIR/$run"
mkdir -p "$out/tb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$MUSUBI_DIR"
exec .venv/bin/accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 ltx2_train_network.py \
  --mixed_precision bf16 --dataset_config "$DATASET_TOML" \
  --ltx2_checkpoint "$LTX_CHECKPOINT" --ltx_version 2.3 --ltx_version_check_mode error --ltx2_mode video \
  --fp8_base --fp8_scaled --ltx2_low_ram_load --blocks_to_swap "${BLOCKS_TO_SWAP:-36}" --block_swap_h2d_only \
  --sdpa --gradient_checkpointing --ltx2_bounded_activation_offload \
  --network_module networks.lora_ltx2 --network_dim "${RANK:-32}" --network_alpha "${RANK:-32}" --lora_target_preset v2v \
  --ic_lora_strategy v2v --reference_frames 121 --reference_downscale 1 --ltx2_first_frame_conditioning_p 0.0 \
  --timestep_sampling shifted_logit_normal --learning_rate "${LR:-1e-4}" --optimizer_type adamw8bit --max_grad_norm 1.0 \
  --max_data_loader_n_workers "${WORKERS:-2}" \
  --max_train_steps "$steps" --save_every_n_steps "${SAVE_EVERY:-500}" --no_convert_to_comfy --save_checkpoint_metadata \
  --validate_every_n_steps "${VALIDATE_EVERY:-250}" \
  --log_with tensorboard --logging_dir "$out/tb" --log_prefix "${run}_" \
  --seed "${SEED:-42}" --output_dir "$out" --output_name "$run" "$@"
