# cuDNN FP8 Restoration Backend (experimental, with TensorRT fallback)

Design notes for the `--fp8-recon` path, which replaces the BasicVSR++ upsample sub-engine with cuDNN graph-API FP8 convolutions.

The existing TensorRT FP16 sub-engine path stays as is.
If the FP8 backend fails to construct (unsupported GPU, missing dependency, and so on), jasna logs a warning and automatically falls back to the TensorRT engine.
The default is off, which is exactly the previous behavior.

**Environment**: the usual GPU-only jasna stack plus the `nvidia-cudnn-frontend` dependency (verified with 1.25.0), cuDNN >= 9.17 (satisfied by the 9.20 bundled with torch 2.12.0+cu130), and an FP8-capable GPU (sm89+, i.e. RTX 40 series or newer).
All numbers in this document were measured on Linux / Python 3.13.14 / torch 2.12.0+cu130 / TensorRT 10.16.1.11 / cuDNN 9.20 / RTX 5080 (sm120, 16GB) / `--max-clip-size 90`.
Windows is verified as well: on Windows 11 with the same RTX 5080, the same A/B benchmark passes every gate with matching numbers (speedup 1.42-1.56x, PSNR 64.1 dB, SSIM 0.99976, net VRAM saving 1976 MB, bit-deterministic), with the glue compiled through the `triton-windows` wheel. See Limitations for details.

The implementation is a port from the sibling project lada-ex, branch `feat/fp8-recon` (`lada/models/basicvsrpp/fp8_recon.py`, AGPL-3.0).
The stage lada-ex calls "reconstruction" is the same sub-network as jasna's upsample sub-engine.

## Background and goal

BasicVSR++, the main restoration model, runs as a set of TensorRT FP16 sub-engines.
The upsample sub-engine (11 reconstruction convolutions plus a 5-layer tail that upscales 4x via pixel shuffle) has a dynamic profile over clip length 1 to `max_clip_size`, and because of that **allocates an internal arena of about 2.2 GB at load time**.
The allocation happens once at load and is held for the whole run.

cuDNN 9.17+ ships Blackwell-native (sm120) FP8 convolution kernels, measured at 2.1-3.0x over the best FP16 kernels on exactly these shapes.
TensorRT's FP8 is a dead end for this workload.
lada-ex measured FP8 convolutions at 1.01-1.08x on standard TensorRT (10.x / 11.1) and on TensorRT-RTX 1.5 alike, so re-trying via TensorRT precision flags is not worth the effort.

The upsample sub-engine alone is therefore replaced with a cuDNN graph-API FP8 implementation.
There are two goals: the stage itself gets faster (about 1.5x measured), and skipping the TensorRT engine load frees about 2 GB of VRAM.
The VRAM saving is the main practical benefit; as the evaluation section shows, the pipeline is detection-bound, so the speedup does not surface in end-to-end fps.

## Architecture

### Target and injection point

The replacement covers the single upsample-engine call made by `BasicVSRPlusPlusNetSplit`: `(T, 5*mid, 64, 64)` fp16 in, `(T, 3, 256, 256)` fp16 out (`mid` is `generator.mid_channels`, default 64).
The residual addition (`+ lqs`) happens outside the engine, in the caller, so the FP8 backend also returns the pre-residual value.
The lada-ex version carries the addition inside the engine; this is the main interface difference in the port.

Construction and injection happen in `create_split_forward()` (`jasna/restorer/basicvsrpp_sub_engines.py`).

1. Only when `JASNA_FP8_RECON=1`, fp16 mode, and a CUDA device are all present does it attempt to construct `CudnnFP8Upsample` (`jasna/restorer/fp8_upsample.py`).
2. Construction runs before the TensorRT sub-engines load, so the FP8 warmup and torch.compile work finish before the TensorRT arenas are allocated.
3. On success, `load_sub_engines(..., load_upsample=False)` skips loading the TensorRT upsample engine entirely. It is never loaded rather than loaded-then-released, so the arena is never allocated.
4. On any failure (all exceptions are caught), a warning is logged and the TensorRT engine loads as before.

The engine-file existence checks (compilation, preflight, the `load_sub_engines` entry gate) are unchanged.
The TensorRT upsample engine keeps being built and kept on disk as the fallback.

Release goes through `BasicVSRPlusPlusNetSplit.close()`, which calls `release()` when an engine provides it and the TensorRT release path otherwise.
`CudnnFP8Upsample` is implemented as a parameter-less `nn.Module` so that `close()`'s `modules()` walk and attribute clearing work on it unchanged.

### Numerical design

The numerical design carries over the lada-ex validation unchanged.

- **Whole-chain FP8/NHWC**: one fp16-to-FP8 (e4m3) cast at the entry, then FP8 between all convolutions. Bouncing intermediates back to fp16 erases the gain.
- **Three fused epilogues**: `act` (bias, ReLU or LeakyReLU(0.1), scale multiply, FP8 out), `res` (scale multiply, bias, identity add, FP8 out), and `final` (scale multiply, bias, fp16 out). Everything after each convolution is fused into the cuDNN graph.
- **Scaling**: weights use a per-tensor amax scale folded into the epilogue; activations use a fixed scale of 1.0. Activation amax sits at a few units, far below FP8's limit of 448, so no calibration is needed. This relies on ReLU / LeakyReLU commuting with positive scaling (positive homogeneity) and must be revisited if the activation function ever changes.
- Computation uses `intermediate_data_type=FLOAT` / `compute_data_type=FLOAT` with heuristics `[A, FALLBACK]`.

### Implementation notes

- **Manual NHWC pixel shuffle**: torch's native `pixel_shuffle` falls into a slow gather path on channels-last 1-byte tensors, about 4x off the floor. A manual int8-view axis permutation is used instead.
- **T buckets and tail tiles**: clip length T is rounded up to multiples of 10, with one cuDNN graph set per bucket. The tail (the 5 layers working at 128^2/256^2 after the pixel shuffles) is frame-independent and runs in T=10 tiles, keeping the large buffers tile-sized. This is what holds the FP8 backend's resident VRAM at about 0.2 GB, versus the roughly 2.2 GB arena it replaces.
- **Shared buffers and the padded path**: buffers are allocated once at the largest bucket and shared across buckets. Non-bucket-exact T is copied into a persistent padding buffer; the tail garbage is discarded by the output slice (no zeroing needed).
- **torch.compile for the glue**: the entry cast and the pixel shuffle reach the memory-bandwidth floor under inductor. inductor needs triton: pytorch-triton comes with torch on Linux, and Windows uses the `triton-windows` wheel added as a dependency (3.7.x, matching torch 2.12; verified working on a real Windows box). Without triton the glue switches to eager automatically, and a runtime compile failure also degrades permanently to eager with a warning (warmup exercises both glue paths, so failures surface at load time). The cast is compiled with `dynamic=True` to avoid per-bucket-shape recompiles.
- **No CUDA graph capture**: measured as ineffective in lada-ex.

### Making construction failures surface immediately

The largest bucket's graph build runs eagerly inside the constructor, outside the warmup's try/except.
On an environment where cuDNN cannot provide FP8 kernels (old cuDNN, unsupported GPU), the exception surfaces here and the caller falls back to TensorRT.
Putting the build inside warmup would swallow the exception, register the backend anyway, and kill a worker thread on the first real call mid-run (a hole lada-ex actually fell into).

Warmup is best-effort and front-loads every lazy cost: graph builds, inductor compiles, allocator pool growth, and two dummy forwards (the bucket-exact path and the padded path).
Because jasna clip lengths spread freely over 1 to `max_clip_size`, warmup **builds every bucket eagerly**.
lada-ex could warm only the top three buckets since its chunk lengths were fixed at 90/80/70; doing the same in jasna leaks lazy builds (0.1-0.2 s each, about 1 s per run measured) into the restore stage during processing.
Warmup takes 1.9 s with a warm inductor cache (around 5 s on the very first run, which includes the compile).

### Coexisting with nvvfx's cuDNN

Combined with `--secondary-restoration rtx-super-res`, the FP8 backend construction initially died with SIGABRT.
The cause has two halves.

nvvfx (RTX Video Effects) prepends its bundled library directory to `os.environ["LD_LIBRARY_PATH"]` at load time.
That directory holds only a libcudnn.so.9 dispatcher (9.7 series, about 125 KB) without the `libcudnn_graph.so.9.7.*` sub-libraries the dispatcher dlopens at run time.

cudnn-frontend, in turn, scans `LD_LIBRARY_PATH` explicitly at import and CDLLs the first libcudnn.so.9 it finds there, keeping that handle.
So whenever rtx-super-res constructed first, the frontend grabbed nvvfx's crippled dispatcher and aborted while resolving `cudnnCreate`.
This is a Python-level path-search problem, not an ELF load-order problem, so pre-loading the right library does not fix it.

The fix: at `fp8_upsample.py` import time, the lib directory of torch's complete cuDNN is put in front of `LD_LIBRARY_PATH`.
The frontend's scan then finds the working dispatcher (9.20) first, and nvvfx is unaffected since it loads its own copy by absolute path anyway.
After the fix, every evaluation run with rtx-super-res enabled completed normally.

Windows has a separate known issue (cudnn-frontend's cudart shim only probes Linux sonames); the module header sets `CUDNN_FRONTEND_CUDART_LIB_NAME` to torch's bundled `cudart64_*.dll` as a workaround, ported from lada-ex.

## CLI and activation

```bash
jasna --input in.mp4 --output out.mp4 --fp8-recon
```

`--fp8-recon` (`BooleanOptionalAction`, default False) bridges to the environment variable `JASNA_FP8_RECON=1`, which the restorer-construction gate reads.
There is no GUI toggle yet, but launching with the environment variable set enables the same path.

Two debug variables exist:

- `JASNA_FP8_RECON_NOCOMPILE=1`: disable torch.compile for the glue and run it eagerly (environments without triton are eager to begin with).
- `JASNA_FP8_RECON_NOWARM=1`: skip warmup (lazy builds keep everything working).

Activation requires all of: `JASNA_FP8_RECON=1`, fp16 mode (fp32 engine configurations are out of scope), a CUDA device, sm89+, a successful `nvidia-cudnn-frontend` import, and cuDNN actually providing FP8 kernels.
Missing any one of these falls back to the TensorRT engine.

## Evaluation

The evaluation runs at three altitudes: an isolated stage A/B (the direct effect, independent of the bottleneck), pipeline before/after (per-stage timing and bottleneck identification), and output quality plus determinism.

### Isolated stage A/B

`jasna --benchmark --benchmark-filter fp8` compares against the TensorRT FP16 engine in the same process on the same real-activation input (a `(90, 320, 64, 64)` batch captured by running a real clip through the split forward).
Numbers are medians of 50 runs.

| T | TensorRT FP16 | FP8 | speedup |
|---|---|---|---|
| 10 | 1.38 ms | 0.95 ms | 1.45x |
| 30 | 4.19 ms | 2.79 ms | 1.50x |
| 60 | 8.54 ms | 5.46 ms | 1.56x |
| 90 | 12.82 ms | 8.39 ms | 1.53x |

Quality is measured against an eager FP32 reference, per-frame PSNR / SSIM in the float domain after the residual addition.

| path | PSNR (mean / min) | SSIM |
|---|---|---|
| FP8 | 64.0 dB / 63.6 dB | 0.99976 |
| TensorRT FP16 (floor) | 92.0 dB / 91.4 dB | 1.00000 |

VRAM, as driver-level (`mem_get_info`) deltas: the TensorRT upsample engine load takes **2210 MB**, the FP8 backend's resident footprint is **219 MB**, a net saving of **1991 MB**.
FP8 output is bit-identical across two runs (both the bucket-exact and the padded path), and construction plus warmup takes 1.9 s (about 5 s on the very first run, including the inductor compile).

This is consistent with the lada-ex reference measurements (1.62x at T=60, PSNR 67.7 dB).

### Pipeline before/after

Measured over six real clips, two configurations (no secondary, and `--secondary-restoration rtx-super-res`), FP8 on/off: 24 conditions in total (test1 and test2 ran twice per condition, interleaved; the rest once).
Wall-clock covers the whole run; VRAM peak / steady-state come from a 200 ms `nvidia-smi` sampler.

No secondary:

| clip | content | wall (FP16 -> FP8) | e2e fps delta | VRAM peak delta | VRAM steady delta |
|---|---|---|---|---|---|
| test1 | 852x480, 10,661f | 61s -> 63s | -3.6% | -1405 MB | -1445 MB |
| test2 | 1080p, 31,524f | 197s -> 198s | -0.4% | -1556 MB | -1582 MB |
| test3 | 720p, 35,956f | 192s -> 193s | -0.7% | -1384 MB | -1610 MB |
| test4 | 720p, 238,380f | 1418s -> 1408s | +0.7% | -1592 MB | -1578 MB |
| test5 | 4K, 32,105f | 380s -> 382s | -0.5% | -1173 MB | -1268 MB |
| test6 | 4K, 4,754f | 64s -> 67s | -4.2% | -1198 MB | -1274 MB |

With rtx-super-res:

| clip | wall (FP16 -> FP8) | e2e fps delta | VRAM peak delta | VRAM steady delta |
|---|---|---|---|---|
| test1 | 71s -> 73s | -2.6% | -1513 MB | -1512 MB |
| test2 | 232s -> 234s | -0.6% | -1572 MB | -1579 MB |
| test3 | 219s -> 220s | -0.7% | -1668 MB | -1598 MB |
| test4 | 1718s -> 1715s | +0.2% | -1453 MB | -1664 MB |
| test5 | 406s -> 408s | -0.4% | -936 MB | -1330 MB |
| test6 | 70s -> 73s | -4.0% | -1190 MB | -1204 MB |

### Bottleneck analysis: why fps does not move

The per-stage timing (`[timing]` logs) shows the decode-detect stage at 100% busy in all 12 configurations, with the primary restore stage carrying large queue-wait (input starvation).
In a producer/consumer pipeline the bottleneck stage sets the end-to-end fps, so speeding up a non-bottleneck stage cannot move it.
The measured wall-clock deltas on long clips stay within +/-1%, exactly as designed.

The FP8 speedup does not surface in the `primary.restore` totals either (within +/-1%; -5.2 s of 1044.6 s on test4, the largest).
The upsample call accounts for only a few milliseconds per clip forward (8.5 ms -> 5.5 ms at T=60), while the propagation passes and preprocessing dominate the restore stage.
Indeed, the whole split forward (T=60) measures 78-79 ms in the TensorRT configuration and 75.5 ms with FP8, matching the isolated 3.1 ms saving (about 4% at the forward level).
Put differently, this stage replacement speeds up a few percent of the restore stage by 1.5x; it could only show up in fps in a configuration where restore itself is the bottleneck (very light detection or very heavy restore load).

Clips around 60 seconds gain 2-3 s of wall-clock (fps -3 to -4%).
That is the FP8 backend's construction and warmup (about 2 s) failing to amortize over a short run; it disappears on long clips.

VRAM, by contrast, dropped consistently in all 12 configurations: peak by 0.9-1.7 GB, steady-state by 1.2-1.7 GB.
The saving is independent of clip resolution (the arena covers 256^2 crops and the `max_clip_size` profile, not the video size).
Higher resolutions eat VRAM through decode surfaces and frame queues (4K peaks at 10.5-11.0 GB in FP16 in these runs), so the fixed saving matters more, in absolute headroom, the higher the resolution.
On a 16 GB GPU, this ~1.5 GB is the difference before `vram_offloader` starts spilling to RAM and losing throughput, or the room that makes a secondary restorer viable at all.

### Output quality (pipeline output vs the FP16 engine)

FP8 off/on output pairs from identical inputs (no secondary, identical NVENC settings) were compared with per-frame PSNR / SSIM.
Note these numbers measure the *difference* between the FP16 and FP8 outputs, not degradation against ground truth (both approximate the FP32 model output; the isolated stage measures 64 dB against FP32, as shown above).

| clip | PSNR (mean / min) | SSIM (mean / min) |
|---|---|---|
| test1 (480p) | 43.5 dB / 41.1 dB | 0.9825 / 0.9733 |
| test2 (1080p) | 44.6 dB / 41.4 dB | 0.9869 / 0.9773 |
| test3 (720p) | 45.6 dB / 42.8 dB | 0.9890 / 0.9796 |
| test4 (720p) | 45.1 dB / 41.7 dB | 0.9862 / 0.9771 |
| test5 (4K) | 48.7 dB / 45.4 dB | 0.9942 / 0.9880 |
| test6 (4K) | 47.4 dB / 44.4 dB | 0.9932 / 0.9851 |

Against the planned gate (mean >= 45 dB), test1 and test2 fall short at 43.5 / 44.6 dB.
Examining what the difference actually consists of, the quality was judged equivalent, on three grounds.
First, even the worst frame (41.1 dB) is visually indistinguishable, and the difference map spreads over the whole frame at a mean of 1.5 LSB (max 22/255) rather than concentrating in the restored regions; this is the encoder's inter prediction diffusing tiny restored-region differences across the GOP, not restoration-quality divergence showing through.
Second, re-measuring with near-lossless settings (`--encoder-settings cq=1`) barely moves test1 (44.7 dB), so encoder path divergence (mode decisions), not compression strength, dominates.
Third, the numbers improve as the restored regions shrink relative to the frame (47-49 dB at 4K), consistent with that mechanism.
The comparable measurement in lada-ex's Windows verification was 49 dB / SSIM 0.993, the same order.

### Determinism

At the stage level, FP8 output is bit-identical across runs.
For pipeline outputs, the floor (encoder or threading jitter) is established first with an FP16-vs-FP16 run pair, then FP8-vs-FP8 pairs are required to match at least as well.

| pair | test1 (10,661f) | test2 (31,524f) |
|---|---|---|
| FP16 run0 vs run1 (floor) | md5 equal | md5 equal |
| FP8 run0 vs run1 | md5 equal | md5 equal |

The FP16 pipeline is fully deterministic across runs (md5 of the decoded raw frames matches), and the FP8 pipeline matches likewise.
FP8 costs no determinism at all.

## Limitations and caveats

- **The speedup is unverified on sm89 / sm90.** The cuDNN FP8 kernels exist there, but the 1.5x was measured only on sm120 (Blackwell); on Ada / Hopper the advantage over the TensorRT FP16 engine may not hold. The opt-in gate plus automatic fallback keeps this safe.
- **Verified on Windows.** On Windows 11 with the same RTX 5080, the isolated A/B benchmark passes every gate (speedup 1.42-1.56x, PSNR 64.1 dB / min 63.6, SSIM 0.99976, FP8 resident 232 MB / net saving 1976 MB, bit-deterministic on both paths) and a full 1080p pipeline run with `--fp8-recon` completes normally. The inductor-compiled glue works on a real Windows box through the `triton-windows` wheel (about 20 s on the very first run including the compile, 2-3 s once the cache is warm); a compile failure still degrades to eager glue automatically. `import cudnn` / triton and DLL resolution inside a frozen build (PyInstaller / Nuitka) remain an open item; if it fails, the TensorRT fallback keeps runs alive.
- **Startup cost grows by about 2 seconds** (about 5 s on the very first run, including the inductor compile). Relatively visible on 60-second-class clips.
- The `forward` return value is a view of a persistent output buffer, overwritten by the next forward. The single current caller consumes it immediately in the residual addition; new callers must keep this in mind.
- fp32 engine configurations (without `--fp16`) are out of scope; the FP8 gate never activates there.

## Tests

`tests/test_fp8_upsample.py` holds the GPU-free tests (gate and fallback with mocks, pixel-shuffle math verification, bucket rounding, CLI parsing) and the GPU-gated ones (CUDA + sm89+ + cudnn frontend): FP32 parity, bit determinism, and `release()` idempotency.
`tests/test_basicvsrpp_sub_engines.py` gained a test that `load_upsample=False` reduces the load to 5 engines.

The isolated stage A/B reproduces from the in-repo benchmark:

```bash
# isolated stage A/B (gate verdicts plus an FP8_AB_JSON line)
jasna --benchmark --benchmark-filter fp8

# split-forward breakdown (with FP8 enabled the upsample line goes through FP8)
JASNA_FP8_RECON=1 jasna --benchmark --benchmark-filter basicvsrpp
```

The pipeline A/B and the output comparisons were run with a local evaluation harness outside the repository.
The procedure, reproducible with equivalent scripting: run `--fp8-recon` off/on pairs on identical inputs with `--log-level info`, collecting per run the wall-clock, the four `[timing]` lines from stderr, and `nvidia-smi --query-gpu=memory.used -lms 200` samples (peak / steady-state).
Output pairs are decoded frame-aligned with av and compared with per-frame PSNR (all frames), sampled SSIM, and, for determinism, the md5 of the decoded raw frames.
