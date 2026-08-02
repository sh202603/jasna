# TensorRT-RTX flavor (opt-in, fast engine compilation)

This documents the opt-in path, selected by the `nvidia-rtx` extra, that swaps the whole TensorRT stack for **TensorRT-RTX** (TensorRT for RTX).

The standard TensorRT path remains the default. TensorRT-RTX is a JIT design without the ahead-of-time exhaustive kernel search, so engine builds finish in seconds instead of minutes. There are two costs: runtime speed is slightly worse (detection engine +12% in the measurements below), and each process start pays a JIT cost (the detection side is covered by a disk cache from the second start on; the restoration side keeps a few seconds per process). It is an option for users who value first-run compile time above all; it does not replace the default.

**Environment**: the numbers in this document were measured on Linux / Python 3.13 / torch 2.12.0+cu130 / torch-tensorrt-rtx 2.12.1 / tensorrt-rtx 1.4.0.76 / RTX 5080 (sm120, 16 GB). Windows is not yet validated.

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

## Measurements (Linux / RTX 5080)

Cold-build wall time (whole compile subprocess, including model loading):

| engine | standard TensorRT | TensorRT-RTX | ratio |
|---|---|---|---|
| RF-DETR v6 (bs1-4 dynamic) | 36.1 s | 4.8 s | 7.6x |
| BasicVSR++ 6 sub-engines | 54.8 s | 16.0 s | 3.4x |

The RTX 5080 is on the fast end for standard-TensorRT builds, which makes the absolute gap look modest. On environments in the "15-60 minutes" class quoted by the GUI's first-run warning (slower GPUs, Windows), the absolute saving should be much larger; confirming that is the main goal of the Windows validation.

Runtime speed and quality:

| item | standard TensorRT | TensorRT-RTX |
|---|---|---|
| RF-DETR v6 inference (batch 4) | 7.5 ms | 8.4 ms (+12%) |
| BasicVSR++ loop_body, one iteration | 0.31 ms | 0.35 ms |
| e2e output agreement (10 s clip) | baseline | PSNR 46.7 dB / SSIM 0.994 |

End-to-end wall time on the 10-second test clip (300 frames) was 5.1 s standard vs 11.1 s RTX. The gap is dominated by the per-process restoration-engine load JIT (previous section), not by the processing rate: restricting the run to the section before restoration starts gives 0.7 s on both flavors, and one loop_body iteration differs by only 0.04 ms. On long videos the load JIT amortizes and the steady-state gap is bounded by the detection engine's +12%.

CUDA graphs (default ON), `--fp8-recon` (the cuDNN FP8 upsample) and the torchcodec backend all completed successfully in the RTX venv, with the same graphs-on/off tendency as the standard flavor.

## Constraints and caveats

- torch-tensorrt's RTX support is officially experimental.
- Engines cannot be reused across OSes (`.rtx.linux` / `.rtx.win` are built separately).
- The tensorrt-rtx 1.4 line has a known issue executing FP8 engines; jasna does not use TensorRT FP8, so this does not apply (`--fp8-recon` is a cuDNN path, unrelated).
- The fp16 conversion of the detection model changes numerics slightly. Detections near the score threshold may differ from the standard flavor, so use the standard flavor where quality is paramount.
