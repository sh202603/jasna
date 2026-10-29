# LTX IC-LoRA training tools (`+modi`)

Build your own LTX restoration model for jasna's public LTX path: convert an ltx-core
LTX-2.3 checkpoint (for example Sulphur-2-base) into jasna's file layout, train an IC-LoRA on
mosaic/clean window pairs with the Musubi Tuner LTX-2 fork, fuse it in. The walkthrough,
with requirements, dataset preparation and the licenses involved, is
[docs/en/ltx_lora_training.md](../../docs/en/ltx_lora_training.md)
([Japanese](../../docs/ja/ltx_lora_training.md)). Every script starts with a docstring or
header comment that states what it does and how to run it.

| Step | Script | Runs with |
| --- | --- | --- |
| Caption to `prompt_context` (Gemma 3 12B, projection, connector) | `encode_prompt.py` | jasna venv + `transformers` |
| Dataset: 121-frame window pairs in jasna's geometry | lada-ex `scripts/dataset_creation/ltx-ic-lora-dump-pairs.py` | lada-ex venv |
| Dataset config template | `musubi/dataset.toml.example` | |
| Latent cache | `musubi/cache_latents.sh` | Musubi venv |
| Text cache from `prompt_context` (no Gemma run) | `musubi/write_te_cache.py` | any venv with torch |
| Training | `musubi/train_ic_lora.sh` | Musubi venv |
| Checkpoint, LoRA fused, to jasna's transformer file | `convert_ltx_to_jasna.py` (`st_writer.py` is its helper) | jasna venv |
| Model folder with the public LTX-2.5 VAE | `make_bundle.py` | jasna venv |
| Checking a LoRA on dataset clips without the pipeline | `ltx_window_infer.py` | jasna venv |
