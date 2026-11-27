"""Convert kukkiri's kiyo-ltx-v1 files into a model folder jasna's LTX restoration loads (+modi).

kukkiri (https://github.com/treehog2/kukkiri) publishes an LTX-2.5 22B video-only transformer
fine-tuned to restore mosaics (https://huggingface.co/treehog/kukkiri, folder ``ltx-v1``). Its
tensors are those of jasna's LTX model files, in another INT8 layout: plain per-row int8
(``<name>.weight`` + ``<name>.weight_scale``), where jasna runs ConvRot W8A8 (``<name>.qdata`` +
``<name>.scales``, the weight rotated along its inputs by a block regular Hadamard matrix). Each
weight is dequantized, rotated and quantized again (``convert_ltx_to_jasna.quantize_convrot``),
so it carries the rounding of both quantizations; the script prints the worst error this adds.

Written into the output folder, under the file names of jasna's 2-step model
(``--restoration-model-name ltx``):

  transformer   kukkiri's transformer, with its prompt embedding as ``prompt_context`` and its
                sampler settings (sigmas, no STG, positions at 30 fps) as metadata
  VAE           kukkiri's VAE (LTX-2.5's conv VAE), hard-linked or copied
  decoder       an empty file: the conv VAE takes no fine-tuned decoder, the bundle needs the name

The converted files are derivatives of LTX-2.x under the LTX-2.x Community License, like their
sources; they are not part of jasna and must not be redistributed with it.

Run with jasna's venv from the jasna checkout (the guide is docs/en/ltx_kukkiri.md):

  .venv/bin/python scripts/ltx_lora_training/convert_kukkiri_ltx.py <folder with kiyo-ltx-v1-*.safetensors> <output folder>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_ltx_to_jasna import quantize_convrot  # noqa: E402
from st_writer import StreamingSafetensorsWriter  # noqa: E402

from jasna.ltx import transformer as T  # noqa: E402
from jasna.ltx.model_files import bundle_names  # noqa: E402

SOURCE_PREFIX = "model.diffusion_model."
SOURCE_FILES = {name: f"kiyo-ltx-v1-{name}.safetensors" for name in ("transformer", "vae", "prompt")}
MODEL_SLOT = "distilled2"  # --restoration-model-name ltx
SIGMAS = [1.0, 0.725, 0.0]  # kukkiri/ltx.py: 2 Euler steps, no CFG, no STG
POSITION_FPS = 30.0  # kukkiri/ltx.py FPS; read by jasna.ltx.sampler.position_fps
PROBE_ROWS = 64


def target_name(source: str) -> str:
    """kukkiri's tensor name -> jasna's (``transformer_blocks.N`` -> ``blocks.N``, prefix dropped)."""
    name = source.removeprefix(SOURCE_PREFIX)
    if name.startswith("transformer_blocks."):
        name = "blocks." + name.removeprefix("transformer_blocks.")
    return name


def tensor_plan(source_shapes: dict[str, tuple[int, ...]]) -> list[tuple[str, str, torch.dtype, tuple[int, ...]]]:
    """(target name, source name, dtype, shape) of every output tensor, in file order. A weight
    with a ``.weight_scale`` beside it becomes ``.qdata`` + ``.scales``; the rest is stored in bf16."""
    quantized = {name.removesuffix("_scale") for name in source_shapes if name.endswith(".weight_scale")}
    out = []
    for source, shape in source_shapes.items():
        name = target_name(source)
        if source in quantized:
            stem = name.removesuffix(".weight")
            out.append((stem + ".qdata", source, torch.int8, shape))
            out.append((stem + ".scales", source + "_scale", torch.float32, shape[:1]))
        elif source.endswith(".weight_scale"):
            continue  # written right after its weight
        else:
            out.append((name, source, torch.bfloat16, shape))
    return out


def dequantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Plain int8 rows and their per-row scales -> the fp32 weight."""
    return weight.float() * scale.float()[:, None]


def requantization_error(
    weight: torch.Tensor, qdata: torch.Tensor, scales: torch.Tensor, generator: torch.Generator
) -> float:
    """Relative error of the ConvRot layer's output against the source layer's, on random inputs."""
    probe = torch.randn(PROBE_ROWS, weight.shape[1], device=weight.device, generator=generator)
    expected = probe @ weight.t()
    actual = T.rotate(probe) @ dequantize(qdata.to(weight.device), scales.to(weight.device)).t()
    return float((actual - expected).norm() / expected.norm())


def convert_transformer(source: Path, prompt: Path, target: Path, device: torch.device) -> float:
    """Writes ``target``; returns the worst relative error a layer's second quantization adds."""
    with safe_open(str(prompt), framework="pt") as file:
        context = file.get_tensor("video_encoding").to(torch.bfloat16)
    worst = 0.0
    generator = torch.Generator(device=device).manual_seed(0)
    with safe_open(str(source), framework="pt") as file:
        shapes = {name: tuple(file.get_slice(name).get_shape()) for name in file.offset_keys()}
        entries = tensor_plan(shapes)
        writer = StreamingSafetensorsWriter(
            target,
            [(name, dtype, shape) for name, _source, dtype, shape in entries]
            + [("prompt_context", torch.bfloat16, tuple(context.shape))],
            metadata={
                "sigmas": json.dumps(SIGMAS),
                "stg_scale": "0.0",
                "position_fps": json.dumps(POSITION_FPS),
                "converted_from": f"kukkiri {SOURCE_FILES['transformer']} (plain int8 -> ConvRot int8)",
                "license": "LTX-2.x Community License (a derivative of LTX-2.x; see the source file's metadata)",
            },
        )
        try:
            scales: torch.Tensor | None = None
            for index, (name, source_name, _dtype, _shape) in enumerate(entries):
                if name.endswith(".qdata"):
                    weight = dequantize(
                        file.get_tensor(source_name).to(device), file.get_tensor(source_name + "_scale").to(device)
                    )
                    qdata, scales = quantize_convrot(weight, device)
                    worst = max(worst, requantization_error(weight, qdata, scales, generator))
                    writer.add(name, qdata)
                elif name.endswith(".scales"):
                    writer.add(name, scales)
                    scales = None
                else:
                    writer.add(name, file.get_tensor(source_name).to(torch.bfloat16))
                if index % 200 == 0:
                    print(f"  {index}/{len(entries)} tensors", flush=True)
            writer.add("prompt_context", context)
        except BaseException:
            writer.out.close()
            writer.tmp.unlink(missing_ok=True)
            raise
        writer.close()
    return worst


def link(source: Path, target: Path) -> None:
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="folder with kukkiri's kiyo-ltx-v1-*.safetensors")
    parser.add_argument("output", type=Path, help="model folder to write, for --restoration-model-path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    sources = {name: args.source / file for name, file in SOURCE_FILES.items()}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing: {', '.join(missing)}")
    args.output.mkdir(parents=True, exist_ok=True)
    # The conv VAE is the full encoder and decoder; name the files for those.
    transformer_name, vae_name, decoder_name = bundle_names(MODEL_SLOT, fast=False, encoder="full", decoder="full")

    print("transformer: plain int8 -> ConvRot int8")
    worst = convert_transformer(
        sources["transformer"], sources["prompt"], args.output / transformer_name, torch.device(args.device)
    )
    print(f"  worst relative output error added by a layer's second quantization: {worst:.4f}")
    link(sources["vae"], args.output / vae_name)
    save_file({}, str(args.output / decoder_name))
    for path in sorted(args.output.iterdir()):
        print(f"  {path.name}: {path.stat().st_size / 1e9:.3f} GB")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
