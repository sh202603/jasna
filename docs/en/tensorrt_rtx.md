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

The raw-API path (RF-DETR, YOLO) has a disk cache (`TrtRunner._create_execution_context`, a `.jitcache` file next to the engine). Measured: load 3.9 s → 0.18 s, on par with standard TensorRT. Cache failures are non-fatal (falls back to a plain execution context).

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
- The fp16 conversion of the detection model changes numerics slightly. Detections near the score threshold may differ from the standard flavor, so use the standard flavor where quality is paramount.
