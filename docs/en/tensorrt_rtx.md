# TensorRT-RTX flavor (opt-in, fast engine compilation)

This documents the opt-in path, selected by the `nvidia-rtx` extra, that swaps the whole TensorRT stack for **TensorRT-RTX** (TensorRT for RTX).

The standard TensorRT path remains the default. TensorRT-RTX is a JIT design without the ahead-of-time exhaustive kernel search, so engine builds finish in seconds instead of minutes. There are two costs: runtime speed is a notch worse (steady-state throughput about −10% in the long-form measurements below, consistent across both machines), and each process start pays a JIT cost (the detection side is covered by a disk cache from the second start on; the restoration side keeps a few seconds per process). It is an option for users who value first-run compile time above all; it does not replace the default.

**Environment**: the numbers in this document were measured on Python 3.13 / torch 2.12.0+cu130 / torch-tensorrt-rtx 2.12.1 / tensorrt-rtx 1.4.0.76, on two machines: Linux / RTX 5080 (sm120, 16 GB) and Windows 11 / RTX 5060 Ti (sm120, 16 GB). Both OSes are validated on hardware (2026-08-02).

The implementation builds on findings from the sibling lada-ex project's `feat/tensorrt-rtx-migration` branch (strongly-typed rejection of `BuilderFlag.FP16`, the need for flavor-separated engine caches, numerical equivalence L1≈0.0005).

## Installation

Install the `nvidia-rtx` extra instead of `nvidia`:

```bash
uv pip install -e .[dev,nvidia-rtx] \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

`torch-tensorrt` and `torch-tensorrt-rtx` both ship the `torch_tensorrt` package, so **the two extras cannot coexist in one venv** — use a dedicated venv per flavor. torch stays at the same 2.12.0+cu130, and the mmengine patch (build guide §5.1) is required in this venv too.

No configuration is needed after installing. The flavor is detected from the presence of the `tensorrt_rtx` wheel (`jasna.engine_paths.trt_flavor()`) and applies to both the CLI and the GUI. The environment variable `JASNA_TRT_FLAVOR` (`rtx` | `standard`) exists as an emergency override; forcing a flavor whose wheel is missing fails early with ImportError (intended).

## Engine cache coexistence

Engines are not compatible across flavors (the TensorRT-RTX runtime cannot deserialize a standard-TensorRT engine and vice versa). RTX-built engines therefore carry a `.rtx` tag in their names:

```
rfdetr-v6.bs1-4.fp16.linux.engine        # standard
rfdetr-v6.bs1-4.fp16.rtx.linux.engine    # RTX
loop_body_backward_1.trt_fp16.rtx.linux.engine
```

Standard names are unchanged, so existing engine caches stay valid. Both flavors can share one `model_weights` directory without destroying each other's caches. Like standard engines, RTX engines are reusable across GPUs within one OS, but not across OSes.

## What differs internally

TensorRT-RTX builds strongly typed: precision follows the network's tensor dtypes. The standard-TensorRT recipe (fp32 ONNX + `BuilderFlag.FP16` + forcing I/O to HALF) therefore does not apply — parsing the fp32 ONNX as-is produces an fp32 engine, measured 5x slower (RF-DETR v6, 37.6 ms/batch-4).

Under the RTX flavor, jasna instead converts the whole ONNX graph to fp16 before parsing (`jasna.trt._convert_onnx_bytes_to_fp16`, dependency `onnxconverter-common`). The conversion covers weights, activations and I/O, and also rewrites `Cast(to=float32)` nodes baked into the original graph (the DINOv2 embedding block has 44 of them) to fp16 — without that rewrite, fp16 weights meet fp32 activations and the strongly-typed parse fails on the type mismatch.

The BasicVSR++ side (the `torch_tensorrt.compile(ir="dynamo")` path) works unmodified since the import name does not change. Graph partitioning is also identical to standard: 4 TRT segments plus one torch-side deform_conv2d island in both flavors.

## JIT cost and the disk cache

TensorRT-RTX defers kernel generation to engine load time.

The raw-API path (RF-DETR, YOLO) has a disk cache (`TrtRunner._create_execution_context`, a `.trtrtx<major>.<minor>.jitcache` file next to the engine — the tensorrt-rtx version is in the name because the engine filename does not encode it, so a runtime upgrade cannot silently reuse a stale cache; an old untagged `.jitcache` is migrated automatically). Measured: load 3.9 s → 0.18 s, on par with standard TensorRT. The cache is re-serialized after warmup and on close, so the kernels that TensorRT-RTX lazily specializes per shape during inference also persist: the first inference of a warm process start drops from ~150 ms to ~10-20 ms. Cache failures are non-fatal (falls back to a plain execution context).

The dynamo path (BasicVSR++) has no cache. torch-tensorrt-rtx 2.12.1 accepts a `runtime_cache_path` kwarg but writes no cache file and does not speed up reloading the saved export (verified empirically). Loading the 6 sub-engines therefore JIT-compiles on every process start (a few seconds per process). On long videos this amortizes away; it is only felt when repeatedly processing short clips.

## Measurements

Cold-build wall time (whole compile subprocess, including model loading):

| engine | standard → RTX (Linux 5080) | standard → RTX (Windows 5060 Ti) |
|---|---|---|
| RF-DETR v6 (bs1-4 dynamic) | 36.1 → 4.8 s (7.6x) | 118.0 → 16.0 s (7.4x) |
| RF-DETR v5 (bs4 static) | not measured | 121.4 → 13.7 s (8.9x) |
| BasicVSR++ 6 sub-engines | 54.8 → 16.0 s (3.4x) | 143.3 → 51.7 s (2.8x) |

The reduction ratio is stable across GPUs at 3-9x, so the slower the standard build, the larger the absolute saving (on the 5060 Ti the total drops from 4.4 minutes to 1.1).

First-load JIT and the disk cache (RF-DETR engine alone):

| item | Linux 5080 | Windows 5060 Ti |
|---|---|---|
| 1st load (JIT + cache write) | 3.9 s | 11.4 s |
| 2nd load (`.jitcache` hit) | 0.18 s | 0.69 s |
| reference: standard engine load | 0.18 s | 0.46 s |

Runtime speed and quality:

| item | standard TensorRT | TensorRT-RTX |
|---|---|---|
| RF-DETR v6 inference (batch 4, Linux 5080) | 7.5 ms | 8.4 ms (+12%) |
| BasicVSR++ loop_body, one iteration (Linux 5080) | 0.31 ms | 0.35 ms |
| e2e output agreement (10 s clip, both OSes) | baseline | PSNR 46.7 dB / SSIM 0.994 |
| e2e output agreement (1080p, 17.5 min, Windows) | baseline | PSNR avg 50.7 dB (min 44.3) / SSIM 0.996 |
| e2e output agreement (same video, Linux) | baseline | PSNR avg 49.9 dB (min 43.7) / SSIM 0.996 |

### Why the +12%, and what was tried

The detection-engine gap was investigated in depth (2026-08-02, Linux 5080, per-layer IProfiler comparison of the two engines). The GEMM/attention kernels are not the problem — the RTX engine's GEMMs are in fact faster than standard TensorRT's (3.05 vs 3.35 ms per batch-4 pass) and multi-head attention is fused identically in both. The deficit sits in unfused data-movement kernels: standard TensorRT folds the copies around the 33 MB `masks` output into the preceding kernels, TensorRT-RTX's JIT leaves them as three standalone kernels (+0.44 ms) plus assorted reshape/copy glue (+0.5 ms). That is Myelin fusion quality inside the runtime, and nothing at the application level moves it — all of the following measured within ±1% of the 8.5 ms baseline: native whole-graph CUDA graph capture (`IRuntimeConfig.cuda_graph_strategy`), EAGER kernel specialization, AOT builder knobs (compute-capability pinning, tiling level, optimization level 3 vs 5), stripping the 20 identity `Cast` nodes left by the fp16 conversion, and a batch-8 profile (per-image cost got worse). Newer runtimes do not close it either: tensorrt-rtx **1.5.0 is a regression** for RF-DETR v6 (9.2 ms, +22% vs standard), **1.6.1 returns to the 1.4 level** (8.5 ms). The flavor therefore stays on 1.4 and the gap is accepted and documented.

Two runtime improvements did land from that investigation: the specialized-kernel cache persistence described in the previous section, and native CUDA graph capture for the detection engines. The latter ships **opt-in** (`JASNA_TRT_RUNNER_CUDAGRAPHS=1`): on Linux the detection stage is GPU-bound, so capture buys nothing and its staging copy + cross-stream synchronization cost about 1% at the stage level (detect-track 131.0 → 132.5 s over 31,524 frames). It may still pay off under Windows' WDDM scheduling, which has higher launch overhead — unverified. Output parity of the graph path vs graphs-off on the long clip: PSNR mean 45.5 dB (min 41.6) / SSIM 0.992 — the graph path always runs the engine at the full batch (short batches are padded), which shifts numerics in the same threshold-adjacent class as the fp16 conversion itself. `JASNA_TRT_RTX_SPECIALIZATION` (`lazy` | `eager` | `none`) overrides the kernel-specialization strategy for experiments.

On short clips the RTX end-to-end wall time is seconds to tens of seconds behind (10-second clip: 5.1 → 11.1 s on Linux, 14.2 → 30.4 s on Windows). The gap is dominated by the per-process restoration-engine load JIT (previous section), not by the processing rate: restricting the run to the section before restoration starts gives 0.7 s on both flavors, and one loop_body iteration differs by only 0.04 ms.

On long videos the load JIT amortizes and only the steady-state difference remains. Measured with the same 1080p / 29.97 fps / 17.5 min / 31,524-frame video on both machines, the steady-state throughput loss lines up at about 10%:

| environment | standard TensorRT | TensorRT-RTX | wall delta | steady-state throughput |
|---|---|---|---|---|
| Windows 5060 Ti | 315.0 s (147.7 fps) | 374.2 s (132.4 fps) | +18.8% | −10.4% |
| Linux 5080 | 132.4 s (238.2 fps) | 152.0 s (207.4 fps) | +14.9% | about −10% (roughly 6 s of the +19.7 s is load JIT) |

CUDA graphs (default ON), `--fp8-recon` (the cuDNN FP8 upsample) and the torchcodec backend all completed successfully in the RTX venvs on both OSes, with the same graphs-on/off tendency as the standard flavor.

## Constraints and caveats

- torch-tensorrt's RTX support is officially experimental.
- Engines cannot be reused across OSes (`.rtx.linux` / `.rtx.win` are built separately).
- The tensorrt-rtx 1.4 line has a known issue executing FP8 engines; jasna does not use TensorRT FP8, so this does not apply (`--fp8-recon` is a cuDNN path, unrelated).
- Do not bump tensorrt-rtx casually: 1.5.0 measured +22% vs standard on RF-DETR v6 (a regression against 1.4's +12%), 1.6.1 only returns to the 1.4 level. Note for a future bump: tensorrt_rtx 1.5+ removed `NetworkDefinitionCreationFlag.EXPLICIT_BATCH`; the builder already guards for its absence.
- The fp16 conversion of the detection model changes numerics slightly. Detections near the score threshold may differ from the standard flavor, so use the standard flavor where quality is paramount.
