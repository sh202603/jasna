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
  child  -> parent (once, at startup):
                    ``{"status":"ready","accel":[...],"accel_log":[...]}\\n``
                    (the active acceleration parts among ``fused_vae`` /
                    ``fp8_dit``, and one line per requested part saying whether
                    it was enabled or why not)
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
    ap.add_argument("--empty-cache", default="auto", choices=["auto", "always", "never"],
                    help="per-clip torch.cuda.empty_cache() policy. Returns cached VRAM to the "
                         "co-resident detection/decode process after every clip; auto = always "
                         "if the GPU has < 20GiB total memory, never otherwise")
    ap.add_argument("--lora-dtype", default="fp32", choices=["fp32", "fp16"],
                    help="measurement-only knob: fp16 keeps the LoRA adapters in the peft "
                         "default dtype (truncating the fp32 checkpoint) instead of promoting "
                         "them to fp32. Reproduces the pre-fix behaviour for A/B measurement; "
                         "fp32 is the numerically verified default")
    ap.add_argument("--fused-vae", action="store_true",
                    help="run the VAE through the checkout's fused path (fused GroupNorm+SiLU "
                         "and fp16-accumulate conv kernels, VAE in fp16). Needs a checkout that "
                         "provides it; dropped with a note in the handshake otherwise")
    ap.add_argument("--fp8-dit", action="store_true",
                    help="merge the LoRA into the base weights and run the DiT block linears "
                         "as FP8 GEMM. Needs a checkout that provides it, an RTX 40 series or "
                         "newer GPU and Triton; dropped with a note in the handshake otherwise")
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


def _decide_accel(want_fused_vae: bool, want_fp8_dit: bool, device, compute_dtype):
    """使える高速化部品をモデルのロード前に決める。返り値: (fused_vae, fp8_dit, accel_log)。
    部品の実装は SeedVR2 checkout 側にあり、上流 (numz) の checkout には無い。checkout が
    提供しない、または GPU が対応しない部品は外して理由を accel_log に残す (標準経路で動く)。"""
    log = []

    def _check(name: str, module: str, func: str, *args) -> bool:
        try:
            unsupported_reason = getattr(__import__(module, fromlist=[func]), func)
        except (ImportError, AttributeError):
            reason = "not provided by this SeedVR2 checkout"
        else:
            reason = unsupported_reason(*args)
        log.append(f"enabled {name}" if reason is None else f"{name} unavailable: {reason}")
        return reason is None

    fused_vae = want_fused_vae and _check(
        "fused_vae", "src.optimization.vae_fusion", "get_vae_fused_path_unsupported_reason", device)
    fp8_dit = want_fp8_dit and _check(
        "fp8_dit", "src.optimization.fp8_gemm", "get_fp8_gemm_unsupported_reason", device, compute_dtype)
    return fused_vae, fp8_dit, log


def _build_runner(repo: str, model_dir: str, dit_model: str, device: str,
                  want_fused_vae: bool = False, want_fp8_dit: bool = False):
    """CLI と同一経路で runner + ctx を構築し、両モデルを materialize して返す。
    返り値: (runner, ctx, fp8_dit, accel_parts, accel_log)。fp8_dit は LoRA 注入後に
    _merge_lora_to_fp8 を呼ぶべきかどうか (FP8 化は注入の後でなければならない)。"""
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
    fused_vae, fp8_dit, accel_log = _decide_accel(
        want_fused_vae, want_fp8_dit, dev, ctx["compute_dtype"])
    runner, cache_context = prepare_runner(
        dit_model=dit_model, vae_model=DEFAULT_VAE,
        model_dir=model_dir,
        debug=debug, ctx=ctx,
        dit_cache=False, vae_cache=False, dit_id=None, vae_id=None,
        block_swap_config={"blocks_to_swap": 0, "swap_io_components": False, "offload_device": None},
        attention_mode="sdpa",
        # 上流の checkout は fused_vae 引数を持たないので、使うときだけ渡す
        **({"fused_vae": True} if fused_vae else {}))
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
    accel_parts = []
    if fused_vae:
        if getattr(runner.vae, "fused_path", False):
            accel_parts.append("fused_vae")
        else:  # checkout 側が VAE の重み形式などを理由に標準経路へ戻した
            accel_log[accel_log.index("enabled fused_vae")] = (
                "fused_vae unavailable: the checkout kept the standard VAE path")
    return runner, ctx, fp8_dit, accel_parts, accel_log


def _inject_lora(runner, ckpt_path: str, rank: int, alpha: int, lora_dtype: str = "fp32") -> int:
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
    # --lora-dtype fp16 はその修正前挙動を再現する計測ノブ (fp32 ckpt は load_state_dict
    # の copy_ cast で fp16 に切り捨てられる)。
    if lora_dtype == "fp32":
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


def _merge_lora_to_fp8(runner) -> int:
    """LoRA を基底重みへマージして peft ラッパーを外し、ブロックの線形層を FP8 GEMM に
    差し替える。FP8 化は nn.Linear そのものを置き換えるので、LoRA を側枝として残せない
    (注入対象が無くなる)。そのため注入・ロードの後に呼ぶ。
    マージ後の出力は側枝 (fp32) 版とビット一致しない (256px 合成モザイクの実測で PSNR
    約 46 dB) が、LoRA の効果は保たれる (LoRA 無しとの差は約 26 dB)。
    返り値: FP8 GEMM に差し替えた線形層の数。"""
    import torch
    from peft.tuners.lora import LoraLayer
    from src.optimization.fp8_gemm import convert_dit_to_fp8_gemm

    dit = runner.dit.dit_model if hasattr(runner.dit, "dit_model") else runner.dit
    with torch.no_grad():
        for parent in list(dit.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, LoraLayer):
                    child.merge()
                    setattr(parent, name, child.base_layer)
        return convert_dit_to_fp8_gemm(dit)


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
        # runner.vae_decode は VAE の融合経路 (--fused-vae) の dtype 変換も行う
        sample = runner.vae_decode([x0])[0]  # (c,t,h,w) / (c,h,w) [-1,1]
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

    # クリップ毎 empty_cache は自プロセスの OOM 対策ではない (caching allocator は割り当て
    # 失敗時に自前でキャッシュ解放+リトライする)。目的は同居する検出/デコード側プロセスへの
    # VRAM 返却で、16GB では同居 OOM の実測歴がある保険、大容量 GPU では純損 → 総 VRAM で
    # 自動切替する。
    if args.empty_cache == "auto":
        per_clip_empty_cache = False
        if str(device).startswith("cuda"):
            total = torch.cuda.get_device_properties(device).total_memory
            per_clip_empty_cache = total < 20 * (1 << 30)
            print(f"seedvr2_lora_worker: empty_cache policy: auto -> "
                  f"{'always' if per_clip_empty_cache else 'never'} "
                  f"(total VRAM {total / (1 << 30):.2f}GiB)", file=sys.stderr)
    else:
        per_clip_empty_cache = args.empty_cache == "always"
        print(f"seedvr2_lora_worker: empty_cache policy: {args.empty_cache}", file=sys.stderr)

    runner, ctx, fp8_dit, accel_parts, accel_log = _build_runner(
        args.repo, args.model_dir, args.dit, device, args.fused_vae, args.fp8_dit)
    n_lora = _inject_lora(runner, args.lora, args.rank, args.alpha, args.lora_dtype)
    print(f"seedvr2_lora_worker: injected {n_lora} LoRA tensors ({args.lora_dtype}) from {args.lora}",
          file=sys.stderr)
    if fp8_dit:
        base_dtype = next(runner.dit.parameters()).dtype
        if base_dtype in (torch.float16, torch.bfloat16, torch.float32):
            n_fp8 = _merge_lora_to_fp8(runner)
            accel_parts.append("fp8_dit")
            print(f"seedvr2_lora_worker: merged the LoRA, {n_fp8} linear layers run as FP8 GEMM",
                  file=sys.stderr)
        else:  # FP8 保存の重みなどには LoRA をマージできない
            accel_log[accel_log.index("enabled fp8_dit")] = (
                f"fp8_dit unavailable: the LoRA cannot be merged into {base_dtype} base weights")
    for line in accel_log:
        print(f"seedvr2_lora_worker: accel: {line}", file=sys.stderr)

    # Warm-up: one minimal window through the full forward. Surfaces CUDA/model
    # errors before the ready handshake and pays the one-off kernel-selection
    # cost outside the first real clip.
    _sr_forward(runner, ctx, torch.zeros(1, 3, 256, 256), seed=args.seed)
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    # Handshake: tell the parent we're ready to accept clips.
    ready = {"status": "ready", "accel": accel_parts, "accel_log": accel_log}
    proto.write((json.dumps(ready) + "\n").encode("utf-8"))

    window, overlap, stride = args.window, args.overlap, args.window - args.overlap
    acc_device = torch.device(device)

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
            # 合成し、合成完了後に 1 回だけ量子化 (二重量子化を避ける)。アキュムレータは
            # GPU 常駐 (channel-first)・CPU への転送は合成完了後の 1 回のみ。fp32 の要素毎
            # mul/add は演算順序含め CPU 合成と同一なので出力は bit 不変 (§10-3 ゲート検証済)。
            acc = torch.zeros((n, 3, h, w), dtype=torch.float32, device=acc_device)
            wsum = torch.zeros((n, 1, 1, 1), dtype=torch.float32, device=acc_device)
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
                    acc[s + i] += wgt * out01[:, i].detach().to(torch.float32)
                    wsum[s + i] += wgt
            blended = (acc / wsum.clamp_min(1e-8)).permute(0, 2, 3, 1).contiguous().cpu().numpy()

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

            if per_clip_empty_cache and str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception as e:  # keep the worker alive; report per-clip failure
            traceback.print_exc()
            err = json.dumps({"seq": seq, "error": f"{type(e).__name__}: {e}"}) + "\n"
            proto.write(err.encode("utf-8"))


if __name__ == "__main__":
    main()
