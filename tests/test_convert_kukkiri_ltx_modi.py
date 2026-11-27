"""+modi: the converter from kukkiri's plain INT8 transformer to jasna's ConvRot layout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from jasna.ltx import sampler
from jasna.ltx.model_files import LtxModelFiles
from jasna.ltx import transformer as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ltx_lora_training"))
import convert_kukkiri_ltx as convert  # noqa: E402


def test_target_names() -> None:
    prefix = convert.SOURCE_PREFIX
    assert convert.target_name(prefix + "transformer_blocks.7.attn1.to_q.weight") == "blocks.7.attn1.to_q.weight"
    assert convert.target_name(prefix + "patchify_proj.bias") == "patchify_proj.bias"


def test_tensor_plan_turns_scaled_weights_into_qdata_and_scales() -> None:
    prefix = convert.SOURCE_PREFIX + "transformer_blocks.0.attn1.to_q."
    plan = convert.tensor_plan(
        {
            prefix + "bias": (4,),
            prefix + "weight": (4, 256),
            prefix + "weight_scale": (4,),
            convert.SOURCE_PREFIX + "scale_shift_table": (2, 8),
        }
    )

    assert [(name, dtype, shape) for name, _source, dtype, shape in plan] == [
        ("blocks.0.attn1.to_q.bias", torch.bfloat16, (4,)),
        ("blocks.0.attn1.to_q.qdata", torch.int8, (4, 256)),
        ("blocks.0.attn1.to_q.scales", torch.float32, (4,)),
        ("scale_shift_table", torch.bfloat16, (2, 8)),
    ]


def test_converted_layer_matches_the_plain_int8_layer(tmp_path: Path) -> None:
    torch.manual_seed(0)
    prefix = convert.SOURCE_PREFIX + "transformer_blocks.0.attn1.to_q."
    weight = torch.randint(-127, 128, (4, 256), dtype=torch.int8)
    scale = torch.rand(4) * 0.01 + 0.001
    bias = torch.randn(4).to(torch.bfloat16)
    source, prompt, target = tmp_path / "t.safetensors", tmp_path / "p.safetensors", tmp_path / "out.safetensors"
    save_file({prefix + "weight": weight, prefix + "weight_scale": scale, prefix + "bias": bias}, str(source))
    save_file({"video_encoding": torch.randn(1, 3, 8)}, str(prompt))

    worst = convert.convert_transformer(source, prompt, target, torch.device("cpu"))

    assert worst < 0.05
    assert not target.with_name(target.name + ".tmp").exists()
    with safe_open(str(target), framework="pt") as file:
        metadata = file.metadata()
        tensors = {name: file.get_tensor(name) for name in file.keys()}
    assert set(tensors) == {
        "blocks.0.attn1.to_q.qdata", "blocks.0.attn1.to_q.scales", "blocks.0.attn1.to_q.bias", "prompt_context",
    }
    assert json.loads(metadata["sigmas"]) == convert.SIGMAS
    assert float(json.loads(metadata["stg_scale"])) == 0.0
    assert sampler.position_fps(metadata) == 30.0
    assert LtxModelFiles(target, target, target).transformer_source == (
        "kukkiri kiyo-ltx-v1-transformer.safetensors (plain int8 -> ConvRot int8)"
    )
    assert tensors["blocks.0.attn1.to_q.qdata"].dtype == torch.int8
    assert tensors["prompt_context"].dtype == torch.bfloat16
    assert torch.equal(tensors["blocks.0.attn1.to_q.bias"], bias)

    probe = torch.randn(16, 256)
    expected = probe @ convert.dequantize(weight, scale).t()
    qdata, scales = tensors["blocks.0.attn1.to_q.qdata"], tensors["blocks.0.attn1.to_q.scales"]
    actual = T.rotate(probe) @ convert.dequantize(qdata, scales).t()
    assert float((actual - expected).norm() / expected.norm()) < 0.05
