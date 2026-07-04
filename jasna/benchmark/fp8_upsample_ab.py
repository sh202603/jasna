"""A/B benchmark: cuDNN FP8 upsample backend vs the TRT FP16 upsample engine.

Same process, same real-activation input. Measures latency (per clip-length
bucket), quality vs an FP32 eager reference, VRAM footprints and bit
determinism, then prints PASS/FAIL against the acceptance gates plus one
machine-readable ``FP8_AB_JSON: {...}`` line.

Run: ``jasna --benchmark --benchmark-filter fp8``
"""
from __future__ import annotations

import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# Matches the fixed upsample batch: the pipeline chunks the upsample stage to
# UPSAMPLE_BATCH-sized calls and the on-disk engine profile is upsample_dyn_b30.
MAX_CLIP = 30
WARMUP = 10
RUNS = 50
LATENCY_TS = (10, 20, 30)
CAPTURE_VIDEO = Path("assets/test_clip1_1080p.mp4")

GATE_SPEEDUP_T90 = 1.4  # gate is evaluated at T=MAX_CLIP
GATE_PSNR_FP8_DB = 60.0
GATE_SSIM_FP8 = 0.9995
GATE_FP8_VRAM_MB = 512.0
GATE_NET_SAVING_MB = 700.0


def _used_mb(device: torch.device) -> float:
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / (1024 * 1024)


def _median_ms(fn, warmup: int = WARMUP, runs: int = RUNS) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    return statistics.median(times)


def _decode_center_crops(video: Path, count: int, size: int = 256) -> torch.Tensor:
    """(count, size, size, 3) uint8 RGB center crops of the first frames."""
    import av

    frames = []
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            img = frame.to_rgb().to_ndarray()
            h, w = img.shape[:2]
            y0, x0 = (h - size) // 2, (w - size) // 2
            frames.append(torch.from_numpy(img[y0:y0 + size, x0:x0 + size].copy()))
            if len(frames) == count:
                break
    if len(frames) < count:  # loop short clips
        while len(frames) < count:
            frames.append(frames[len(frames) % max(len(frames), 1)])
    return torch.stack(frames)


def _fp32_upsample_ref(generator: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Eager FP32 reference of the upsample chain (non-mutating weight casts)."""

    def conv(m, t):
        return F.conv2d(t, m.weight.float(), m.bias.float(), padding=1)

    cur = F.leaky_relu(conv(generator.reconstruction.main[0], x), 0.1)
    for blk in generator.reconstruction.main[2]:
        h1 = F.relu(conv(blk.conv1, cur))
        cur = cur + conv(blk.conv2, h1)
    cur = F.leaky_relu(F.pixel_shuffle(conv(generator.upsample1.upsample_conv, cur), 2), 0.1)
    cur = F.leaky_relu(F.pixel_shuffle(conv(generator.upsample2.upsample_conv, cur), 2), 0.1)
    cur = F.leaky_relu(conv(generator.conv_hr, cur), 0.1)
    return conv(generator.conv_last, cur)


def _frame_metrics(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    """(mean PSNR, min PSNR, mean SSIM) over frames of [0,1] (T,3,H,W) tensors.

    Computed on float arrays (no uint8 rounding, which would cap PSNR at
    ~59dB) — same methodology as the lada-ex reference measurements."""
    from jasna.models.basicvsrpp.mmagic.psnr import psnr
    from jasna.models.basicvsrpp.mmagic.ssim import ssim

    out_f = out.clamp(0, 1).mul(255.0).permute(0, 2, 3, 1).cpu().numpy()
    ref_f = ref.clamp(0, 1).mul(255.0).permute(0, 2, 3, 1).cpu().numpy()
    psnrs = [float(psnr(ref_f[i], out_f[i])) for i in range(out_f.shape[0])]
    ssims = [float(ssim(ref_f[i], out_f[i])) for i in range(out_f.shape[0])]
    return statistics.mean(psnrs), min(psnrs), statistics.mean(ssims)


def benchmark_fp8_upsample_ab(
    *,
    device: torch.device,
    fp16: bool,
    restoration_model_path: Path | None,
    compile_basicvsrpp: bool,
    **_: object,
) -> None:
    if restoration_model_path is None or str(restoration_model_path).strip() in ("", "."):
        # mirror main()'s default resolution (the CLI default is an empty string)
        from jasna.model_weights_resolver import resolve_model_weights_file

        restoration_model_path = resolve_model_weights_file(
            "lada_mosaic_restoration_model_generic_v1.2.pth",
        )
    if not Path(restoration_model_path).resolve().is_file():
        print("fp8_upsample_ab: restoration model missing, skipping.")
        return
    if not fp16:
        print("fp8_upsample_ab: FP8 backend is fp16-only, skipping (run without --no-fp16).")
        return
    if torch.cuda.get_device_capability(device) < (8, 9):
        print("fp8_upsample_ab: needs an FP8-capable GPU (sm89+), skipping.")
        return
    try:
        import cudnn  # noqa: F401
    except ImportError:
        print("fp8_upsample_ab: nvidia-cudnn-frontend not installed, skipping.")
        return
    if not CAPTURE_VIDEO.exists():
        print(f"fp8_upsample_ab: {CAPTURE_VIDEO} missing, skipping.")
        return

    from jasna.engine_paths import get_basicvsrpp_sub_engine_paths
    from jasna.models.basicvsrpp.inference import load_model
    from jasna.restorer.basicvsrpp_sub_engines import (
        _get_inference_generator,
        _release_torchtrt_module,
        create_split_forward,
    )
    from jasna.restorer.fp8_upsample import CudnnFP8Upsample
    from jasna.trt.torch_tensorrt_export import load_torchtrt_export

    path = str(restoration_model_path.resolve())
    paths = get_basicvsrpp_sub_engine_paths(path, True)
    if not all(os.path.isfile(p) for p in paths.values()):
        print(f"fp8_upsample_ab: sub-engines missing "
              f"(compile them with a normal run first), skipping.")
        return

    print(f"\n=== FP8 upsample A/B (max_clip={MAX_CLIP}, {RUNS} runs/point) ===")
    result: dict[str, object] = {"t": {}}

    # --- VRAM: TRT upsample engine load-time arena (standalone) ---
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before = _used_mb(device)
    t0 = time.perf_counter()
    trt_probe = load_torchtrt_export(checkpoint_path=paths["upsample"], device=device)
    torch.cuda.synchronize()
    trt_load_s = time.perf_counter() - t0
    trt_vram_mb = _used_mb(device) - before
    _release_torchtrt_module(trt_probe)
    del trt_probe
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  TRT upsample engine load: {trt_vram_mb:8.0f} MB arena, {trt_load_s:5.1f}s")

    # --- Build model + TRT split (FP8 gate must stay off here) ---
    fp8_env = os.environ.pop("JASNA_FP8_RECON", None)
    try:
        model = load_model(None, path, device, True)
        split = create_split_forward(
            model=model, model_weights_path=path, device=device,
            fp16=True,
        )
    finally:
        if fp8_env is not None:
            os.environ["JASNA_FP8_RECON"] = fp8_env
    if split is None:
        print("fp8_upsample_ab: split forward unavailable, skipping.")
        return
    generator = _get_inference_generator(model)
    trt_upsample = split._upsample_engine

    # --- Capture real activations: run the split on real video frames ---
    crops = _decode_center_crops(CAPTURE_VIDEO, MAX_CLIP)
    lqs = (crops.permute(0, 3, 1, 2).to(device=device, dtype=torch.float16)
           .div_(255.0).unsqueeze(0))
    holder: dict[str, torch.Tensor] = {}

    class _Recorder(nn.Module):
        def forward(self, x):
            holder["hr"] = x.detach().clone()
            return trt_upsample(x)

    split._upsample_engine = _Recorder()
    with torch.inference_mode():
        split(lqs)
    split._upsample_engine = trt_upsample
    hr = holder["hr"]  # (MAX_CLIP, 5*mid, 64, 64) fp16
    lq = lqs.squeeze(0)
    print(f"  captured hr_batch {tuple(hr.shape)} amax={hr.float().abs().max().item():.2f}")

    # --- FP8 backend construction + warmup (VRAM + startup cost) ---
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before = _used_mb(device)
    t0 = time.perf_counter()
    fp8 = CudnnFP8Upsample(generator, device, MAX_CLIP)
    torch.cuda.synchronize()
    fp8_build_s = time.perf_counter() - t0
    # drop the warmup's transient allocator pool so the delta reflects the
    # backend's steady footprint (buffers + graphs + workspace)
    gc.collect()
    torch.cuda.empty_cache()
    fp8_vram_mb = _used_mb(device) - before
    print(f"  FP8 backend build+warmup: {fp8_vram_mb:8.0f} MB, {fp8_build_s:5.1f}s")

    # --- Latency ---
    print(f"  {'T':>4} {'TRT ms':>9} {'FP8 ms':>9} {'speedup':>8}")
    for t in LATENCY_TS:
        x = hr[:t]
        trt_ms = _median_ms(lambda: trt_upsample(x))
        fp8_ms = _median_ms(lambda: fp8(x))
        speedup = trt_ms / fp8_ms
        result["t"][str(t)] = {"trt_ms": round(trt_ms, 3), "fp8_ms": round(fp8_ms, 3),
                               "speedup": round(speedup, 3)}
        print(f"  {t:>4} {trt_ms:>9.2f} {fp8_ms:>9.2f} {speedup:>7.2f}x")

    # --- Quality vs FP32 eager reference (after the +lq residual) ---
    with torch.inference_mode():
        ref = (_fp32_upsample_ref(generator, hr.float()) + lq.float()).clamp(0, 1)
        out_fp8 = (fp8(hr).float() + lq.float()).clamp(0, 1)
        out_trt = (trt_upsample(hr).float() + lq.float()).clamp(0, 1)
    p8_mean, p8_min, s8_mean = _frame_metrics(out_fp8, ref)
    p16_mean, p16_min, s16_mean = _frame_metrics(out_trt, ref)
    print(f"  quality vs FP32 ref: FP8  PSNR {p8_mean:.1f}dB (min {p8_min:.1f}) SSIM {s8_mean:.5f}")
    print(f"                       FP16 PSNR {p16_mean:.1f}dB (min {p16_min:.1f}) SSIM {s16_mean:.5f} (floor)")

    # --- Determinism (bucket-exact and padded paths) ---
    with torch.inference_mode():
        det_exact = torch.equal(fp8(hr[:60]).clone(), fp8(hr[:60].clone()))
        det_pad = torch.equal(fp8(hr[:55]).clone(), fp8(hr[:55].clone()))
    deterministic = bool(det_exact and det_pad)
    print(f"  bit determinism: bucket-exact={det_exact} padded={det_pad}")

    # --- Gates ---
    speedup_t90 = result["t"][str(MAX_CLIP)]["speedup"]
    net_saving_mb = trt_vram_mb - fp8_vram_mb
    gates = {
        f"speedup@T{MAX_CLIP} >= {GATE_SPEEDUP_T90}": speedup_t90 >= GATE_SPEEDUP_T90,
        f"PSNR(FP8 vs FP32) >= {GATE_PSNR_FP8_DB}dB": p8_mean >= GATE_PSNR_FP8_DB,
        f"SSIM(FP8 vs FP32) >= {GATE_SSIM_FP8}": s8_mean >= GATE_SSIM_FP8,
        f"FP8 VRAM <= {GATE_FP8_VRAM_MB:.0f}MB": fp8_vram_mb <= GATE_FP8_VRAM_MB,
        f"net VRAM saving >= {GATE_NET_SAVING_MB:.0f}MB": net_saving_mb >= GATE_NET_SAVING_MB,
        "bit-deterministic": deterministic,
    }
    all_pass = all(gates.values())
    print("  gates:")
    for name, ok in gates.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")

    result.update({
        "psnr_fp8_db": round(p8_mean, 2), "psnr_fp8_min_db": round(p8_min, 2),
        "psnr_fp16_floor_db": round(p16_mean, 2), "ssim_fp8": round(s8_mean, 5),
        "ssim_fp16_floor": round(s16_mean, 5),
        "vram_trt_mb": round(trt_vram_mb), "vram_fp8_mb": round(fp8_vram_mb),
        "net_saving_mb": round(net_saving_mb),
        "trt_load_s": round(trt_load_s, 1), "fp8_build_s": round(fp8_build_s, 1),
        "deterministic": deterministic, "pass": all_pass,
    })
    print(f"FP8_AB_JSON: {json.dumps(result)}")

    fp8.release()
    split.close()
