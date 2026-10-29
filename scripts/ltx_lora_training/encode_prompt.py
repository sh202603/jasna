"""Precompute jasna's ``prompt_context`` (1, 1024, 4096) bf16 for a fixed caption, the way
ltx-core (LTX-2.3, 22B "V2" text path) does it:

  Gemma 3 12B it  -> hidden states of all 49 layers (embeddings + 48), tokens padded LEFT to 1024
  FeatureExtractorV2: per-token RMS over the hidden dim per layer, concat layers (3840*49),
                      zero padded tokens, * sqrt(4096/3840), video_aggregate_embed (188160 -> 4096)
  Embeddings1DConnector: valid tokens moved first (right-pad order), padded positions replaced by
                      the 128 learnable registers (tiled x8), 8 transformer_1d_blocks with 1D split
                      RoPE (theta 1e4, max_pos 4096, float64 frequency grid), gated attention,
                      q/k RMSNorm, GELU-tanh FFN, final RMSNorm (no weight); mask ends all-valid.

The projection and connector weights are read from the LTX-2.3 checkpoint itself (``--source``,
the same file convert_ltx_to_jasna.py takes). Reference code: LTX-2
packages/ltx-core/src/ltx_core/text_encoders/gemma/{feature_extractor,embeddings_processor,
embeddings_connector,tokenizer}.py, encoders/base_encoder.py, model/transformer/{rope,attention}.py.

jasna has no text encoder, so this is the only place the caption enters the model: the output
is baked into the transformer file as ``prompt_context`` and, for training, written into
Musubi's text-encoder cache by musubi/write_te_cache.py, so training and inference see the same
constant. Gemma 3 12B is loaded with ``device_map="auto"`` under ``--gpu-budget`` (the rest
goes to host RAM), and its hidden states can be cached (``--hidden-cache``) to rerun the
projection and connector without reloading Gemma.

  .venv/bin/python scripts/ltx_lora_training/encode_prompt.py --gemma-dir gemma-3-12b-it \
      --source sulphur_distil_bf16.safetensors --prompt "$(cat prompt.txt)" \
      --output prompt_context.safetensors --hidden-cache gemma_hidden.safetensors
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

MAX_LENGTH = 1024
CONNECTOR = "model.diffusion_model.video_embeddings_connector."
PROJ = "text_embedding_projection.video_aggregate_embed."
HEADS, HEAD_DIM, DIM, LAYERS, REGISTERS = 32, 128, 4096, 8, 128
ROPE_THETA, ROPE_MAX_POS = 10000.0, 4096


def tokenize(tokenizer, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """ltx-core LTXGemmaTokenizer.tokenize_with_weights: strip, truncate to 1024, ensure leading BOS,
    pad on the LEFT to 1024."""
    text = text.strip()
    bos = tokenizer.bos_token_id
    ids = tokenizer(text, padding=False, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").input_ids[0].tolist()
    if not ids or ids[0] != bos:
        ids = [bos, *ids][:MAX_LENGTH]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    padded = tokenizer.pad({"input_ids": [ids]}, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt", return_attention_mask=True)
    return padded.input_ids, padded.attention_mask


def gemma_hidden_states(gemma_dir: Path, input_ids: torch.Tensor, attention_mask: torch.Tensor, gpu_budget: str) -> torch.Tensor:
    """[1, 1024, 3840, 49] bf16 on CUDA: the tuple of hidden states, stacked on the last dim."""
    from transformers import Gemma3ForConditionalGeneration

    t0 = time.time()
    model = Gemma3ForConditionalGeneration.from_pretrained(
        str(gemma_dir), dtype=torch.bfloat16, device_map="auto", max_memory={0: gpu_budget, "cpu": "52GiB"}, low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"gemma loaded in {time.time() - t0:.0f} s; device map sample: {list(model.hf_device_map.items())[:3]} ...")
    t0 = time.time()
    with torch.inference_mode():
        out = model.model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device), output_hidden_states=True)
    hidden = torch.stack([h.to("cuda", torch.bfloat16) for h in out.hidden_states], dim=-1)
    print(f"gemma forward in {time.time() - t0:.0f} s: {len(out.hidden_states)} hidden states, stacked {tuple(hidden.shape)} {hidden.dtype}")
    del out, model
    torch.cuda.empty_cache()
    return hidden


def feature_extractor_v2(hidden: torch.Tensor, attention_mask: torch.Tensor, proj_w: torch.Tensor, proj_b: torch.Tensor) -> torch.Tensor:
    b, t, d, l = hidden.shape
    variance = torch.mean(hidden ** 2, dim=2, keepdim=True)
    normed = (hidden * torch.rsqrt(variance + 1e-6)).reshape(b, t, d * l)
    normed = torch.where(attention_mask.bool().unsqueeze(-1).to(normed.device), normed, torch.zeros_like(normed))
    normed = normed.to(hidden.dtype) * math.sqrt(DIM / d)
    return F.linear(normed, proj_w, proj_b)


def rope_1d(seq: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [1, heads, seq, head_dim/2] as ltx-core precompute_freqs_cis (split, max_pos [4096], np float64 grid)."""
    count = DIM // 2  # inner_dim // (2 * n_pos_dims), n_pos_dims = 1
    grid = np.power(ROPE_THETA, np.linspace(0.0, 1.0, count, dtype=np.float64))
    indices = torch.as_tensor(grid * math.pi / 2, dtype=torch.float32)
    positions = torch.arange(seq, dtype=torch.float32)
    fractional = positions / ROPE_MAX_POS
    freqs = indices[None, :] * (fractional[:, None] * 2 - 1)  # [seq, count]
    cos, sin = freqs.cos(), freqs.sin()
    cos = cos.reshape(1, seq, HEADS, -1).swapaxes(1, 2)
    sin = sin.reshape(1, seq, HEADS, -1).swapaxes(1, 2)
    return cos.to(device=device, dtype=torch.bfloat16), sin.to(device=device, dtype=torch.bfloat16)


def apply_split_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [1, seq, dim] -> rotated [1, heads, seq, head_dim] (ltx-core apply_split_rotary_emb)."""
    xh = x.unflatten(-1, (HEADS, HEAD_DIM)).transpose(1, 2)  # [1, H, S, D]
    split = xh.unflatten(-1, (2, HEAD_DIM // 2))  # [1, H, S, 2, D/2]
    first_in, second_in = split[..., :1, :], split[..., 1:, :]
    out = split * cos.unsqueeze(-2)
    first = out[..., :1, :].addcmul(-sin.unsqueeze(-2), second_in)
    second = out[..., 1:, :].addcmul(sin.unsqueeze(-2), first_in)
    return torch.cat([first, second], dim=-2).flatten(-2)


def rms(x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=1e-6)


def connector_block(w: dict[str, torch.Tensor], h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x = rms(h)
    v = F.linear(x, w["attn1.to_v.weight"], w["attn1.to_v.bias"])
    q = rms(F.linear(x, w["attn1.to_q.weight"], w["attn1.to_q.bias"]), w["attn1.q_norm.weight"])
    k = rms(F.linear(x, w["attn1.to_k.weight"], w["attn1.to_k.bias"]), w["attn1.k_norm.weight"])
    q, k = apply_split_rope(q, cos, sin), apply_split_rope(k, cos, sin)
    vh = v.unflatten(-1, (HEADS, HEAD_DIM)).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, vh)  # mask is all-valid after register substitution
    out = out.transpose(1, 2)  # [1, S, H, D]
    gates = 2.0 * torch.sigmoid(F.linear(x, w["attn1.to_gate_logits.weight"], w["attn1.to_gate_logits.bias"]))
    out = (out * gates.unsqueeze(-1)).flatten(2)
    h = h + F.linear(out, w["attn1.to_out.0.weight"], w["attn1.to_out.0.bias"])
    x = rms(h)
    ff = F.gelu(F.linear(x, w["ff.net.0.proj.weight"], w["ff.net.0.proj.bias"]), approximate="tanh")
    return h + F.linear(ff, w["ff.net.2.weight"], w["ff.net.2.bias"])


def connector(features: torch.Tensor, attention_mask: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    device = features.device
    binary = attention_mask.to(torch.int32).to(device)  # [1, S]
    order = torch.argsort(binary, dim=-1, descending=True, stable=True)
    features = torch.gather(features, 1, order.unsqueeze(-1).expand_as(features))
    binary = torch.gather(binary, 1, order).to(features.dtype).unsqueeze(-1)
    registers = weights["learnable_registers"].repeat(MAX_LENGTH // REGISTERS, 1).to(features.dtype)[None]
    h = binary * features + (1 - binary) * registers
    cos, sin = rope_1d(MAX_LENGTH, device)
    for i in range(LAYERS):
        w = {k.removeprefix(f"transformer_1d_blocks.{i}."): v for k, v in weights.items() if k.startswith(f"transformer_1d_blocks.{i}.")}
        h = connector_block(w, h, cos, sin)
    return rms(h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma-dir", type=Path, required=True, help="Gemma 3 12B it checkpoint folder (HF layout)")
    ap.add_argument("--source", type=Path, default=None, help="ltx-core LTX-2.3 checkpoint (projection + connector weights)")
    ap.add_argument("--prompt", required=True, help="the fixed caption")
    ap.add_argument("--output", type=Path, required=True, help="safetensors with 'prompt_context' (1,1024,4096) bf16")
    ap.add_argument("--gpu-budget", default="11GiB", help="VRAM Gemma may take; the rest of the model goes to host RAM")
    ap.add_argument("--hidden-cache", type=Path, default=None, help="save/reuse the stacked Gemma hidden states")
    ap.add_argument("--gemma-only", action="store_true", help="stop after caching the Gemma hidden states")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.gemma_dir))
    input_ids, attention_mask = tokenize(tokenizer, args.prompt)
    valid = int(attention_mask.sum())
    print(f"prompt tokens: {valid} valid of {MAX_LENGTH} (left padded); first ids {input_ids[0, -valid:][:8].tolist()}")
    if args.hidden_cache and args.hidden_cache.is_file():
        with safe_open(str(args.hidden_cache), framework="pt", device="cuda") as f:
            hidden = f.get_tensor("hidden")
        print(f"hidden states loaded from {args.hidden_cache}: {tuple(hidden.shape)}")
    else:
        hidden = gemma_hidden_states(args.gemma_dir, input_ids, attention_mask, args.gpu_budget)
        if args.hidden_cache:
            save_file({"hidden": hidden.cpu()}, str(args.hidden_cache), metadata={"prompt": args.prompt, "valid_tokens": str(valid)})
            print(f"hidden states cached at {args.hidden_cache}")
    if args.gemma_only:
        return
    if args.source is None:
        raise SystemExit("--source is required (the LTX-2.3 checkpoint holds the projection and connector weights)")
    device = hidden.device
    with safe_open(str(args.source), framework="pt", device="cpu") as f:
        proj_w = f.get_tensor(PROJ + "weight").to(device, torch.bfloat16)
        proj_b = f.get_tensor(PROJ + "bias").to(device, torch.bfloat16)
        weights = {k.removeprefix(CONNECTOR): f.get_tensor(k).to(device, torch.bfloat16) for k in f.keys() if k.startswith(CONNECTOR)}
    print(f"projection {tuple(proj_w.shape)}, connector tensors {len(weights)}")
    with torch.inference_mode():
        features = feature_extractor_v2(hidden, attention_mask, proj_w, proj_b)
        print(f"features {tuple(features.shape)} std {features.float().std():.4f}")
        context = connector(features, attention_mask, weights)
    print(f"prompt_context {tuple(context.shape)} {context.dtype} mean {context.float().mean():.4f} std {context.float().std():.4f} "
          f"finite {bool(torch.isfinite(context.float()).all())}")
    save_file({"prompt_context": context.cpu().contiguous()}, str(args.output), metadata={"prompt": args.prompt, "valid_tokens": str(valid), "gemma": args.gemma_dir.name})
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
