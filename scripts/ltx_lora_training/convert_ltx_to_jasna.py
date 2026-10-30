"""Convert an ltx-core LTX-2.3 checkpoint into the transformer file jasna's LTX path reads.

Input: one safetensors in ltx-core's bundle layout (``model.diffusion_model.*`` plus the audio,
text and VAE branches), for example ``sulphur_distil_bf16.safetensors`` from
SulphurAI/Sulphur-2-base. Output: one safetensors with jasna's tensor names, optionally with a
Musubi Tuner LTX-2 LoRA fused in, quantized to ConvRot W8A8 (``--quant int8``, what the release
weights use) or left in bf16 (``--quant bf16``, about twice the size, no Triton int8 kernels).

What it does (the guide is docs/en/ltx_lora_training.md, step 5):
  rename   model.diffusion_model.transformer_blocks.N.<rest> -> blocks.N.<rest>,
           model.diffusion_model.<top> -> <top>
  drop     audio / a2v / v2a / av_ca / embeddings_connector / vae / audio_vae / vocoder /
           text_embedding_projection (everything that is not the video transformer)
  dtype    fp32 scale_shift tables -> bf16
  add      keyframes_abs_pos_embedding = zeros (1, 4096): LTX-2.3 has none, jasna adds it to the
           first latent frame unconditionally, so zeros disable it;
           prompt_context (1, 1024, 4096) bf16 from --prompt-context (made by encode_prompt.py)
  lora     --lora fuses W += multiplier * alpha / rank * up @ down in fp32 before quantization
  metadata sigmas (the sampler's sigma list, JSON, ending in 0) and stg_scale
  quant    ConvRot W8A8: qdata = int8 codes of rotate(W) per output row, scales fp32 per row,
           the same rotate / quantize_rows as jasna.ltx.transformer, so the runtime GEMM is
           x_rot_int8 @ qdata.T * act_scale * scales (+ bias)

Run with jasna's venv from the jasna checkout (it imports jasna.ltx.transformer and
jasna.ltx.trial_shapes). ``--device cuda`` quantizes on the GPU with a few hundred MB of VRAM;
``--selftest`` checks the int8 path against an fp32 reference and needs CUDA. The output is
streamed tensor by tensor, so a 46 GB input converts without holding the output in RAM.

  .venv/bin/python scripts/ltx_lora_training/convert_ltx_to_jasna.py \
      --source sulphur_distil_bf16.safetensors --prompt-context prompt_context.safetensors \
      --lora my-lora.safetensors --output my-model-int8.safetensors
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from st_writer import StreamingSafetensorsWriter  # noqa: E402

from jasna.ltx import trial_shapes as S  # noqa: E402
from jasna.ltx import transformer as T  # noqa: E402

PREFIX = "model.diffusion_model."
BLOCK_RE = re.compile(r"^model\.diffusion_model\.transformer_blocks\.(\d+)\.(.+)$")
DROP_SUBSTRINGS = ("audio", "a2v", "v2a", "av_ca", "embeddings_connector")
BLOCK_KEEP = set(
    [f"{a}.{p}.{leaf}" for a in ("attn1", "attn2") for p in ("to_q", "to_k", "to_v", "to_out.0", "to_gate_logits") for leaf in ("weight", "bias")]
    + [f"{a}.{n}.weight" for a in ("attn1", "attn2") for n in ("q_norm", "k_norm")]
    + ["ff.net.0.proj.weight", "ff.net.0.proj.bias", "ff.net.2.weight", "ff.net.2.bias", "scale_shift_table", "prompt_scale_shift_table"]
)
TOP_KEEP_PREFIXES = ("adaln_single.", "prompt_adaln_single.", "patchify_proj.", "proj_out.", "scale_shift_table")
QUANT_NAMES = set(T.LINEAR_NAMES)
TORCH_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def quantize_convrot(weight: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(out, in)`` weight -> (int8 ``qdata`` (out, in), fp32 ``scales`` (out,)); rotation in fp32."""
    rows = weight.shape[0]
    rotated = T.rotate(weight.to(device=device, dtype=torch.float32))
    codes, scales = T.quantize_rows(rotated)
    return codes[:rows].contiguous().cpu(), scales[:rows].contiguous().cpu()


def selftest(device: torch.device) -> None:
    torch.manual_seed(0)
    for out_f, in_f in ((32, 4096), (4096, 4096), (4096, 16384)):
        w = (torch.randn(out_f, in_f) * 0.02).to(torch.bfloat16)
        b = (torch.randn(out_f) * 0.02).to(torch.bfloat16)
        x = torch.randn(1, 300, in_f, device=device).to(torch.bfloat16)
        q, s = quantize_convrot(w, device)
        ref = torch.nn.functional.linear(x.float(), w.float().to(device), b.float().to(device))
        got = T.int8_linear(x, q.to(device), s.to(device), b.to(device)).float()
        err = (got - ref).norm() / ref.norm()
        print(f"selftest {out_f}x{in_f}: relative error {err:.4f}")
        assert err < 0.05, "ConvRot int8 self-test failed"
    print("selftest ok")


LORA_PREFIX = "lora_unet_model_transformer_blocks_"


def load_lora(path: Path, multiplier: float) -> dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, float]]:
    """{(block index, converter weight name e.g. 'attn1.to_q.weight'): (down [r, in], up [out, r], scale)}"""
    names = {base: base.replace(".", "_") for base in (b.removesuffix(".weight") for b in BLOCK_KEEP if b.endswith(".weight"))}
    by_underscore = {v: k for k, v in names.items()}
    out: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, float]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        modules = sorted({k.rsplit(".", 2)[0] if k.endswith(".weight") else k.rsplit(".", 1)[0] for k in f.keys()})
        for module in modules:
            assert module.startswith(LORA_PREFIX), module
            rest = module[len(LORA_PREFIX):]
            index_str, under = rest.split("_", 1)
            base = by_underscore.get(under)
            if base is None:
                raise SystemExit(f"LoRA module {module} does not map onto a converter weight name")
            down = f.get_tensor(f"{module}.lora_down.weight").float()
            up = f.get_tensor(f"{module}.lora_up.weight").float()
            alpha = float(f.get_tensor(f"{module}.alpha")) if f"{module}.alpha" in f.keys() else float(down.shape[0])
            out[(int(index_str), f"{base}.weight")] = (down, up, multiplier * alpha / down.shape[0])
    print(f"lora: {len(out)} modules from {path.name}, scale alpha/dim*mult = {next(iter(out.values()))[2]:.4f}")
    return out


def fuse_lora(weight: torch.Tensor, entry: tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    down, up, scale = entry
    assert up.shape[0] == weight.shape[0] and down.shape[1] == weight.shape[1], (up.shape, down.shape, weight.shape)
    return (weight.float() + scale * (up @ down)).to(torch.bfloat16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, help="ltx-core checkpoint (LTX-2.3 bundle layout)")
    ap.add_argument("--prompt-context", type=Path, help="safetensors with 'prompt_context' (1,1024,4096) bf16 (encode_prompt.py output)")
    ap.add_argument("--output", type=Path, help="jasna transformer file to write")
    ap.add_argument("--quant", choices=("int8", "bf16"), default="int8", help="ConvRot W8A8 (default) or plain bf16 weights")
    ap.add_argument("--lora", type=Path, default=None, help="Musubi LTX-2 LoRA (lora_unet_model_transformer_blocks_N_<module>.lora_{down,up}.weight, .alpha) fused into the weights before quantization")
    ap.add_argument("--lora-multiplier", type=float, default=1.0, help="LoRA strength multiplier (1.0 = as trained)")
    ap.add_argument("--sigmas", default="1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0",
                    help="sampler sigma list stored as metadata (default: LTX-2.3 distilled 8-step schedule)")
    ap.add_argument("--stg-scale", type=float, default=0.0, help="STG scale stored as metadata (0 for distilled models)")
    ap.add_argument("--device", default="cuda", help="device for the ConvRot rotation and quantization")
    ap.add_argument("--selftest", action="store_true", help="check the int8 path against fp32 on random weights and exit (needs CUDA)")
    args = ap.parse_args()
    device = torch.device(args.device)
    if args.selftest:
        selftest(device)
        return
    if not (args.source and args.prompt_context and args.output):
        ap.error("--source, --prompt-context and --output are required (or --selftest)")

    sigmas = [float(v) for v in args.sigmas.split(",")]
    assert sigmas[-1] == 0.0 and sigmas[0] == 1.0
    with safe_open(str(args.prompt_context), framework="pt", device="cpu") as f:
        prompt_context = f.get_tensor("prompt_context").to(torch.bfloat16)
        prompt_meta = f.metadata() or {}
    assert tuple(prompt_context.shape) == (1, 1024, 4096), prompt_context.shape

    lora = load_lora(args.lora, args.lora_multiplier) if args.lora else {}
    fused = 0
    src = safe_open(str(args.source), framework="pt", device="cpu")
    src_meta = src.metadata() or {}
    # ---- plan -------------------------------------------------------------------------
    top: dict[str, str] = {}          # jasna name -> source name
    blocks: dict[int, dict[str, str]] = {}
    dropped, unknown = [], []
    for key in src.keys():
        if not key.startswith(PREFIX):
            dropped.append(key)
            continue
        if any(s in key for s in DROP_SUBSTRINGS):
            dropped.append(key)
            continue
        m = BLOCK_RE.match(key)
        if m:
            index, rest = int(m.group(1)), m.group(2)
            if rest not in BLOCK_KEEP:
                unknown.append(key)
                continue
            blocks.setdefault(index, {})[rest] = key
        else:
            name = key[len(PREFIX):]
            if not name.startswith(TOP_KEEP_PREFIXES):
                unknown.append(key)
                continue
            top[name] = key
    if unknown:
        raise SystemExit(f"unexpected video-side tensors ({len(unknown)}), e.g. {unknown[:5]}")
    assert sorted(blocks) == list(range(T.NUM_BLOCKS)), sorted(blocks)[:3]
    print(f"source: {len(top)} top tensors, {len(blocks)} blocks, {len(dropped)} dropped (audio/vae/text/connector)")

    def spec(key: str) -> tuple[torch.dtype, tuple[int, ...]]:
        sl = src.get_slice(key)
        return TORCH_DTYPES[sl.get_dtype()], tuple(sl.get_shape())

    plan: list[tuple[str, torch.dtype, tuple[int, ...]]] = []
    plan.append(("keyframes_abs_pos_embedding", torch.bfloat16, (1, 4096)))
    plan.append(("prompt_context", torch.bfloat16, (1, 1024, 4096)))
    for name in sorted(top):
        _, shape = spec(top[name])
        plan.append((name, torch.bfloat16, shape))
    for index in range(T.NUM_BLOCKS):
        for rest in sorted(blocks[index]):
            dtype, shape = spec(blocks[index][rest])
            base = rest.removesuffix(".weight")
            if args.quant == "int8" and rest.endswith(".weight") and base in QUANT_NAMES:
                plan.append((f"blocks.{index}.{base}.qdata", torch.int8, shape))
                plan.append((f"blocks.{index}.{base}.scales", torch.float32, (shape[0],)))
            else:
                plan.append((f"blocks.{index}.{rest}", torch.bfloat16, shape))
    # ---- check against jasna's recorded shapes ----------------------------------------
    expected = S.TRANSFORMER["int8"]
    exp_top = dict(expected["top"])
    if args.quant == "int8":
        exp_block = dict(expected["block"])
    else:  # bf16 keeps every linear weight under its original name, with no scales
        exp_block = {}
        for k, (dt, sh) in expected["block"].items():
            if k.endswith(".scales"):
                continue
            exp_block[k.removesuffix(".qdata") + ".weight" if k.endswith(".qdata") else k] = ("BF16", sh)
    planned = {name: (dtype, shape) for name, dtype, shape in plan}
    missing_top = [k for k in exp_top if k not in planned]
    missing_block = [k for k in exp_block if f"blocks.0.{k}" not in planned]
    extra_block = sorted(k.split(".", 2)[2] for k in planned if k.startswith("blocks.0.") and k.split(".", 2)[2] not in exp_block)
    extra_top = sorted(k for k in planned if not k.startswith("blocks.") and k not in exp_top)
    for k, (dt, sh) in exp_top.items():
        if k in planned and tuple(planned[k][1]) != tuple(sh):
            raise SystemExit(f"shape mismatch {k}: {planned[k][1]} vs expected {sh}")
    for k, (dt, sh) in exp_block.items():
        pk = f"blocks.0.{k}"
        if pk in planned and tuple(planned[pk][1]) != tuple(sh):
            raise SystemExit(f"shape mismatch {pk}: {planned[pk][1]} vs expected {sh}")
    print(f"vs trial_shapes ({args.quant}): missing top {missing_top}, missing block {missing_block}, extra top {extra_top}, extra block {extra_block}")
    if missing_top or missing_block:
        raise SystemExit("missing tensors")

    metadata = {
        "sigmas": json.dumps(sigmas),
        "stg_scale": json.dumps(args.stg_scale),
        "converter": "jasna scripts/ltx_lora_training/convert_ltx_to_jasna.py",
        "source_file": args.source.name,
        "source_model_version": src_meta.get("model_version", ""),
        "quantization": "convrot-w8a8" if args.quant == "int8" else "bf16",
        "prompt": prompt_meta.get("prompt", ""),
        "lora_file": args.lora.name if args.lora else "",
        "lora_multiplier": json.dumps(args.lora_multiplier) if args.lora else "",
        "license": f"Derivative of Lightricks LTX-2.x ({args.source.name}); subject to the LTX-2 Community License Agreement",
    }
    writer = StreamingSafetensorsWriter(args.output, plan, metadata)
    print(f"writing {args.output} ({writer.total / 1e9:.2f} GB of tensor data)")
    t0 = time.time()
    writer.add("keyframes_abs_pos_embedding", torch.zeros(1, 4096, dtype=torch.bfloat16))
    writer.add("prompt_context", prompt_context)
    for name in sorted(top):
        writer.add(name, src.get_tensor(top[name]).to(torch.bfloat16))
    for index in range(T.NUM_BLOCKS):
        for rest in sorted(blocks[index]):
            tensor = src.get_tensor(blocks[index][rest])
            if (index, rest) in lora:
                tensor = fuse_lora(tensor, lora.pop((index, rest)))
                fused += 1
            base = rest.removesuffix(".weight")
            if args.quant == "int8" and rest.endswith(".weight") and base in QUANT_NAMES:
                qdata, scales = quantize_convrot(tensor, device)
                writer.add(f"blocks.{index}.{base}.qdata", qdata)
                writer.add(f"blocks.{index}.{base}.scales", scales)
            else:
                writer.add(f"blocks.{index}.{rest}", tensor.to(torch.bfloat16))
        if index % 8 == 7:
            print(f"  block {index + 1}/{T.NUM_BLOCKS} done, {time.time() - t0:.0f} s")
    writer.close()
    if lora:
        raise SystemExit(f"{len(lora)} LoRA modules were not fused: {sorted(lora)[:5]}")
    print(f"fused LoRA modules: {fused}")
    print(f"done in {time.time() - t0:.0f} s: {args.output} {args.output.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
