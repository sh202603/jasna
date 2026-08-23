# SPDX-FileCopyrightText: 2026 sh202603
# SPDX-License-Identifier: AGPL-3.0
"""
Persistent SeedVR2+LoRA inference worker — runs inside the SeedVR2
(seedvr2_videoupscaler) venv, NOT the lada venv; spawned by Seedvr2LoraRestorer.

This is the **primary** mosaic restorer worker: it receives raw 256px mosaic
crops (not a prior restoration) and returns restored crops at the same
resolution (scale=1, the LoRA's training distribution). Per clip it runs an
overlapped sliding-window loop (default 33-frame windows, stride 24, 9-frame
linear crossfade — "ov9") over a 1-step SR diffusion forward with the LoRA
injected into the DiT at load time, then applies an optional clip-level color
correction against the mosaic input and quantizes once.

This script is intentionally free of any ``lada`` import: it is executed by
the SeedVR2 env's Python (``--seedvr2-python``) and only uses numpy/torch/cv2
plus the SeedVR2 repo (added to ``sys.path`` at runtime). The wire color order
is lada-native BGR; the worker converts BGR<->RGB around inference.

Wire protocol (parent = lada venv, child = this):
  parent -> child : header ``{"seq","n","h","w"}\\n`` (UTF-8) then n*h*w*3 raw
                    uint8 BGR bytes  (the 256px mosaic crops, HWC)
  child  -> parent: header ``{"seq","n","h","w"}\\n`` then n*h*w*3 raw uint8
                    BGR bytes  (the restored crops, HWC), exactly n frames
  child  -> parent (once, at startup): ``{"status":"ready"}\\n``
  child  -> parent (on per-clip failure): ``{"seq","error":"..."}\\n`` then
                    stays alive for the next clip.

fd handling: the *real* stdout fd is dup'd to a private protocol fd, then fd 1
is repointed to /dev/null (or stderr if --verbose) so library banners / prints
never corrupt the wire. ``TQDM_DISABLE=1`` is set before tqdm is imported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

# アロケータ差はポインタ整列経由で cuBLAS/cuDNN のカーネル選択を変え、訓練/検証
# ハーネスとの bit 整合を壊す。torch import 前に必要 (親も spawn 前に設定する)。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SeedVR2+LoRA primary restorer worker (256px, scale=1)")
    ap.add_argument("--repo", required=True, help="seedvr2_videoupscaler checkout")
    ap.add_argument("--model-dir", required=True, help="SeedVR2 base weights dir (auto-download target)")
    ap.add_argument("--dit", default="seedvr2_ema_3b_fp16.safetensors", help="base DiT weights name")
    ap.add_argument("--lora", required=True, help="LoRA state-dict checkpoint (.pt)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--window", type=int, default=33, help="sliding window length (4n+1)")
    ap.add_argument("--overlap", type=int, default=9, help="crossfade width in frames (< window)")
    ap.add_argument("--color-fix", default="lab", choices=["none", "lab", "wavelet"],
                    help="clip-level color correction against the mosaic input "
                         "(applied post-crossfade, pre-quantization)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--verbose", action="store_true", help="send worker stdout to stderr, not /dev/null")
    return ap.parse_args()


def _install_protocol_fd(verbose: bool):
    """Reserve the real stdout as the protocol channel and mute fd 1."""
    proto_fd = os.dup(1)
    proto = os.fdopen(proto_fd, "wb", buffering=0)
    sink = sys.stderr.fileno() if verbose else os.open(os.devnull, os.O_WRONLY)
    os.dup2(sink, 1)
    return proto


def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("stdin closed while reading payload")
        buf.extend(chunk)
    return bytes(buf)


def _read_header(stream) -> dict | None:
    line = bytearray()
    while True:
        b = stream.read(1)
        if not b:
            return None
        if b == b"\n":
            break
        line.extend(b)
    return json.loads(line.decode("utf-8"))


# ---------------------------------------------------------------------------- #
# SeedVR2 1-step SR forward (inference subset of the training-verified phase
# functions; numerically identical to the pilot harness, which the LoRA's
# training/validation bit-parity gates were established against).
# ---------------------------------------------------------------------------- #

# blocks.{i}.attn.proj_qkv/.proj_out の .vid と .all (txt 凍結) +
# blocks.{i}.mlp.{vid|all}.proj_in_gate/.proj_in/.proj_out
_LORA_TARGET_REGEX = (
    r"blocks\.\d+\.(attn\.proj_(qkv|out)\.(vid|all)"
    r"|mlp\.(vid|all)\.proj_(in_gate|in|out))"
)


def _build_runner(repo: str, model_dir: str, dit_model: str, device: str):
    """CLI と同一経路で runner + ctx を構築し、両モデルを materialize して返す。"""
    import torch
    from src.utils.debug import Debug
    from src.utils.model_registry import DEFAULT_VAE
    from src.core.model_loader import materialize_model
    from src.core.generation_utils import (
        load_text_embeddings, prepare_video_transforms, setup_generation_context,
        prepare_runner, ensure_precision_initialized)

    debug = Debug(enabled=False)
    dev = torch.device(device)
    ctx = setup_generation_context(
        dit_device=dev, vae_device=dev,
        dit_offload_device=None, vae_offload_device=None,
        tensor_offload_device=None, debug=debug)
    runner, cache_context = prepare_runner(
        dit_model=dit_model, vae_model=DEFAULT_VAE,
        model_dir=model_dir,
        debug=debug, ctx=ctx,
        dit_cache=False, vae_cache=False, dit_id=None, vae_id=None,
        block_swap_config={"blocks_to_swap": 0, "swap_io_components": False, "offload_device": None},
        attention_mode="sdpa")
    ctx["cache_context"] = cache_context
    materialize_model(runner, "vae", dev, runner.config, debug)
    materialize_model(runner, "dit", dev, runner.config, debug)
    ensure_precision_initialized(ctx, runner, debug)
    # 1-step 蒸留モデルの推論設定 (generation_phases.upscale_all_batches と同一)
    runner.config.diffusion.cfg.scale = 1.0
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion(device=dev, dtype=ctx["compute_dtype"])
    ctx["text_embeds"] = load_text_embeddings(repo, dev, ctx["compute_dtype"], debug)
    ctx["video_transform"] = prepare_video_transforms(256)
    return runner, ctx


def _inject_lora(runner, ckpt_path: str, rank: int, alpha: int) -> int:
    """peft の inject_adapter_in_model でラッパー無しにその場注入し、LoRA 重みを読む。
    lora_B ゼロ初期化 + strict=False ロードなので、キー不一致はゼロ寄与のまま残る —
    ロード数 0 は設定ミス (rank/alpha 違い等) として起動失敗にする。"""
    import torch
    # seedvr2 の compatibility シム (src/optimization/compatibility.py) は
    # bitsandbytes が無い/壊れている環境で sys.modules へ空スタブを登録する。
    # peft は find_spec がスタブの spec を返すため bnb 利用可能と誤認し、
    # bnb.nn 参照の AttributeError で起動に失敗する (bnb 未インストールの
    # Windows で実測)。実体の無いスタブは捨てて「bnb 無し」に倒す。
    _bnb_stub = sys.modules.get("bitsandbytes")
    if _bnb_stub is not None and not hasattr(_bnb_stub, "nn"):
        del sys.modules["bitsandbytes"]
    from peft import LoraConfig, inject_adapter_in_model

    dit = runner.dit.dit_model if hasattr(runner.dit, "dit_model") else runner.dit
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0,
                     target_modules=_LORA_TARGET_REGEX, bias="none")
    inject_adapter_in_model(cfg, dit)
    # peft は adapter を基底層と同じ dtype (fp16) で作るが、ckpt は fp32 で保存されて
    # おり、訓練/評価ハーネス (lora_setup.inject_lora adapter_dtype=fp32) も fp32 の
    # まま推論する。fp16 のままロードすると adapter が切り捨てられ、ハーネスとの
    # 出力一致 (§10-3 worker 一致ゲート) が量子化誤差レベルを超えて崩れる。
    for n, p in dit.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(torch.float32)
    for p in runner.dit.parameters():
        p.requires_grad_(False)
    for p in runner.vae.parameters():
        p.requires_grad_(False)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing_unexpected = dit.load_state_dict(sd, strict=False)
    loaded = sum(1 for k in sd if k not in missing_unexpected.unexpected_keys)
    if loaded == 0:
        raise RuntimeError(f"no LoRA tensors from {ckpt_path} matched the injected adapters "
                           f"(rank={rank}? target modules changed?)")
    return loaded


def _vae_decode(runner, latent):
    """infer.vae_decode (grouping=False) 等価。latent: (t',h',w',c)。"""
    import torch
    from omegaconf import ListConfig
    from src.optimization.performance import optimized_channels_to_second

    cfg = runner.config.vae
    device = next(runner.vae.parameters()).device
    dtype = getattr(torch, cfg.dtype)
    scale = cfg.scaling_factor
    shift = cfg.get("shifting_factor", 0.0)
    if isinstance(scale, ListConfig):
        scale = torch.tensor(scale, device=device, dtype=dtype)
    if isinstance(shift, ListConfig):
        shift = torch.tensor(shift, device=device, dtype=dtype)
    lat = latent.unsqueeze(0)
    lat = lat / scale + shift
    lat = optimized_channels_to_second(lat)
    lat = lat.squeeze(2)
    vae_dtype = next(runner.vae.parameters()).dtype
    if vae_dtype != lat.dtype:
        with torch.autocast(device.type, lat.dtype, enabled=True):
            sample = runner.vae.decode(
                lat, tiled=runner.decode_tiled, tile_size=runner.decode_tile_size,
                tile_overlap=runner.decode_tile_overlap).sample
    else:
        sample = runner.vae.decode(
            lat, tiled=runner.decode_tiled, tile_size=runner.decode_tile_size,
            tile_overlap=runner.decode_tile_overlap).sample
    if hasattr(runner.vae, "postprocess"):
        sample = runner.vae.postprocess(sample)
    return sample.squeeze(0)


def _sr_forward(runner, ctx, lq01_tchw, seed: int):
    """1-step SR forward。lq01_tchw: (t,c,h,w) RGB float [0,1]。
    返り値: (c,t,h,w)…t=1 のときは (c,h,w)…の [0,1] float (clamp 済み、量子化前)。"""
    import torch
    from src.common.seed import set_seed
    from src.models.dit_3b import na

    dev = ctx["dit_device"]
    cdt = ctx["compute_dtype"]
    # CLI (encode_all_batches) は manage_tensor で bf16 化してから transform を適用する。
    # 同じ精度で Normalize 等が走るよう、cast してから transform する
    video = ctx["video_transform"](lq01_tchw.to(device=dev, dtype=cdt))  # (c,t,h,w) [-1,1]

    set_seed(seed + 1000000)  # generation_phases.encode_all_batches と同一
    with torch.no_grad():
        latent = runner.vae_encode([video])[0]  # (t',h',w',16) スケール済み
        latent = latent.to(device=dev, dtype=cdt)

        set_seed(seed)  # upscale_all_batches と同一
        noise = torch.randn_like(latent, dtype=cdt)
        cond = runner.get_condition(noise, latent_blur=latent, task="sr")

        vid_flat, vid_shape = na.flatten([noise])
        cond_flat, _ = na.flatten([cond])
        pos_flat, pos_shape = na.flatten(ctx["text_embeds"]["texts_pos"])
        t = torch.tensor([1000.0], device=dev, dtype=cdt)

        # generation_phases.upscale_all_batches と同一: DiT 重み dtype が compute_dtype と
        # 異なる場合 (fp16 重み × bf16 compute) は autocast で実行する
        dit_model = runner.dit.dit_model if hasattr(runner.dit, "dit_model") else runner.dit
        dit_dtype = next(dit_model.parameters()).dtype
        use_autocast = dit_dtype != cdt and dev.type != "mps"

        with torch.autocast(dev.type, cdt, enabled=use_autocast):
            pred = runner.dit(
                vid=torch.cat([vid_flat, cond_flat], dim=-1),
                txt=pos_flat,
                vid_shape=vid_shape,
                txt_shape=pos_shape,
                timestep=t,
            ).vid_sample
        x0_flat = vid_flat - pred  # lerp/v_lerp endpoint at t=T: A=0, B=1
        x0 = na.unflatten(x0_flat, vid_shape)[0]
        sample = _vae_decode(runner, x0)  # (c,t,h,w) / (c,h,w) [-1,1]
        out01 = sample.clamp(-1, 1) * 0.5 + 0.5
    return out01


# ---------------------------------------------------------------------------- #
# 🎨 clip-level color correction (post-crossfade, pre-quantization)
#
# Reference = the mosaic input clip itself: mosaicing is cell-averaging, which
# roughly preserves region-global color statistics, so it is a valid reference
# for *global* statistics matching. Whole-clip (not per-window, not per-frame)
# statistics keep the correction temporally stable — per-window stats would
# re-introduce the chunk-boundary stepping the ov9 crossfade removes.
# ---------------------------------------------------------------------------- #


def _color_fix_lab(blended, ref_frames):
    """Global LAB mean/std transfer over the whole clip. Moves no structural
    information, so it cannot re-introduce mosaic grids.

    blended: (n,h,w,3) float32 RGB [0,1] (mutated logic-free, returns new array).
    ref_frames: (n,h,w,3) uint8 RGB (the mosaic input crops).
    """
    import cv2
    import numpy as np

    n = blended.shape[0]
    out_lab = np.stack([cv2.cvtColor(blended[i], cv2.COLOR_RGB2LAB) for i in range(n)])
    ref01 = ref_frames.astype(np.float32) / 255.0
    ref_lab = np.stack([cv2.cvtColor(ref01[i], cv2.COLOR_RGB2LAB) for i in range(n)])
    o_mean = out_lab.reshape(-1, 3).mean(axis=0)
    o_std = out_lab.reshape(-1, 3).std(axis=0) + 1e-6
    r_mean = ref_lab.reshape(-1, 3).mean(axis=0)
    r_std = ref_lab.reshape(-1, 3).std(axis=0) + 1e-6
    fixed_lab = (out_lab - o_mean) / o_std * r_std + r_mean
    fixed = np.stack([cv2.cvtColor(fixed_lab[i], cv2.COLOR_LAB2RGB) for i in range(n)])
    return np.clip(fixed, 0.0, 1.0)


def _wavelet_blur(x, radius):
    import torch
    import torch.nn.functional as F

    vals = [[0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]]
    kernel = torch.tensor(vals, dtype=x.dtype, device=x.device)
    weight = kernel.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    x_pad = F.pad(x, (radius,) * 4, mode="replicate")
    return F.conv2d(x_pad, weight, bias=None, stride=1, padding=0,
                    dilation=radius, groups=x.shape[1])


def _wavelet_decompose(x, levels=5):
    import torch

    high = torch.zeros_like(x)
    low = x
    for i in range(levels):
        blurred = _wavelet_blur(low, 2 ** i)
        high = high + (low - blurred)
        low = blurred
    return high, low


def _color_fix_wavelet(blended, ref_frames, device):
    """Output's high-frequency detail on the mosaic input's low frequencies.
    CAUTION (design §7.2): with a mosaic reference the low band can retain cell
    structure for large/high-contrast grids — gated by sum_cov before it can
    become a default. Kept for the V5 measurement matrix.
    """
    import numpy as np
    import torch

    n = blended.shape[0]
    out = np.empty_like(blended)
    with torch.no_grad():
        for i in range(n):
            content = (torch.from_numpy(np.ascontiguousarray(blended[i]))
                       .to(device).permute(2, 0, 1).unsqueeze(0))
            style = (torch.from_numpy(np.ascontiguousarray(ref_frames[i]))
                     .to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                     .div_(255.0))
            c_high, _ = _wavelet_decompose(content)
            _, s_low = _wavelet_decompose(style)
            fixed = (c_high + s_low).clamp_(0.0, 1.0).squeeze(0).permute(1, 2, 0)
            out[i] = fixed.to(device="cpu", dtype=torch.float32).numpy()
    return out


def main() -> None:
    args = _parse_args()
    assert args.window % 4 == 1, "--window must be 4n+1"
    assert 0 <= args.overlap < args.window

    # Mute banners/tqdm before importing the repo; reserve the protocol fd.
    os.environ.setdefault("TQDM_DISABLE", "1")
    proto = _install_protocol_fd(args.verbose)
    stdin = sys.stdin.buffer

    # Import the SeedVR2 repo. chdir: embedded files (pos_emb.pt etc.) are
    # referenced checkout-relative.
    sys.path.insert(0, args.repo)
    os.chdir(args.repo)

    import numpy as np
    import torch
    from src.core.generation_phases import _apply_4n1_padding

    device = args.device
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)

    runner, ctx = _build_runner(args.repo, args.model_dir, args.dit, device)
    n_lora = _inject_lora(runner, args.lora, args.rank, args.alpha)
    print(f"seedvr2_lora_worker: injected {n_lora} LoRA tensors from {args.lora}", file=sys.stderr)

    # Warm-up: one minimal window through the full forward. Surfaces CUDA/model
    # errors before the ready handshake and pays the one-off kernel-selection
    # cost outside the first real clip.
    _sr_forward(runner, ctx, torch.zeros(1, 3, 256, 256), seed=args.seed)
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    # Handshake: tell the parent we're ready to accept clips.
    proto.write((json.dumps({"status": "ready"}) + "\n").encode("utf-8"))

    window, overlap, stride = args.window, args.overlap, args.window - args.overlap

    while True:
        header = _read_header(stdin)
        if header is None:
            break  # parent closed stdin -> shut down
        seq = int(header.get("seq", -1))
        n = int(header["n"])
        h = int(header["h"])
        w = int(header["w"])
        payload = _read_exact(stdin, n * h * w * 3)

        try:
            if (h, w) != (256, 256):
                raise ValueError(f"expected 256x256 crops (scale=1 contract), got {h}x{w}")
            crops = np.frombuffer(payload, dtype=np.uint8).reshape(n, h, w, 3)
            rgb = np.ascontiguousarray(crops[..., ::-1])  # wire BGR -> RGB

            # ov9 窓ループ: 重複部は線形ランプで float32 アキュムレータへクロスフェード
            # 合成し、合成完了後に 1 回だけ量子化 (二重量子化を避ける)。
            acc = np.zeros((n, h, w, 3), np.float32)
            wsum = np.zeros((n, 1, 1, 1), np.float32)
            starts = list(range(0, max(n - overlap, 1), stride))
            for s in starts:
                batch = rgb[s:s + window]
                lq01 = torch.from_numpy(batch.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
                t_orig = lq01.shape[0]
                if t_orig % 4 != 1:
                    lq01 = _apply_4n1_padding(lq01)  # CLI と同一の末尾パディング
                out01 = _sr_forward(runner, ctx, lq01, seed=args.seed)  # (c,t,h,w)
                if out01.dim() == 3:  # t_orig=1 の端数ウィンドウは (c,h,w) で返る
                    out01 = out01.unsqueeze(1)
                for i in range(t_orig):
                    wgt = 1.0
                    if overlap:
                        if s > 0 and i < overlap:  # 先頭ランプ (前チャンクとの重なり)
                            wgt = (i + 1) / (overlap + 1)
                        if s + window < n and i >= stride:  # 末尾ランプ (次チャンクとの重なり)
                            wgt = min(wgt, (t_orig - i) / (overlap + 1))
                    frame = out01[:, i].detach().to(torch.float32).cpu().numpy().transpose(1, 2, 0)
                    acc[s + i] += wgt * frame
                    wsum[s + i] += wgt
            blended = acc / np.maximum(wsum, 1e-8)

            if args.color_fix == "lab":
                blended = _color_fix_lab(blended, rgb)
            elif args.color_fix == "wavelet":
                blended = _color_fix_wavelet(blended, rgb, device)

            # CLI と同じ ×255 切り捨て量子化、RGB -> BGR (lada wire)
            out_arr = (np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)
            out_arr = np.ascontiguousarray(out_arr[..., ::-1])

            resp = json.dumps({"seq": seq, "n": n, "h": h, "w": w}) + "\n"
            proto.write(resp.encode("utf-8"))
            proto.write(out_arr.tobytes())

            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception as e:  # keep the worker alive; report per-clip failure
            traceback.print_exc()
            err = json.dumps({"seq": seq, "error": f"{type(e).__name__}: {e}"}) + "\n"
            proto.write(err.encode("utf-8"))


if __name__ == "__main__":
    main()
