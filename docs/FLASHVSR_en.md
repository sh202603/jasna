# FlashVSR offline secondary restoration (`+modi`)

`--secondary-restoration flashvsr` upscales each restored 256px mosaic crop to
1024px (4x) with [FlashVSR](https://github.com/OpenImagingLab/FlashVSR)
(one-step streaming diffusion VSR; jasna uses the
[`lihaoyun6/FlashVSR_plus`](https://github.com/lihaoyun6/FlashVSR_plus) fork) to
recover texture realism the primary BasicVSR++ model leaves blurry on large
mosaic regions, close-ups, and 4K sources.

FlashVSR has two modes:

- **`--secondary-restoration flashvsr` (offline 3-phase)** — three processes via
  intermediate files. Works on **12 GB-class GPUs** and with an unpatched FlashVSR
  checkout, and supports staged resume. Most of this document describes this mode.
- **`--secondary-restoration flashvsr-inline` (inline, single pass)** — runs
  FlashVSR inside the normal streaming pipeline with **no intermediate files, no
  disk gate, and no double encode**. Requires a **16 GB card and a FlashVSR
  checkout with the tiny-long patch**. See "Inline mode" at the end.

Why offline 3-phase exists: FlashVSR's tiny mode peaks at **12–16 GB VRAM on its
own**, so it cannot co-reside with jasna's primary pipeline on a 16 GB card.
Splitting the work across processes whose peak VRAM never overlaps in time is what
makes it fit. Inline mode instead uses FlashVSR's **tiny-long** (constant ~11.9 GB,
requires the patch), co-residing with the primary (~1.6 GB under fp8-recon) to run
as a single pass.

## How it works — offline 3-phase

A single `--secondary-restoration flashvsr` command runs three subprocesses in
sequence. Each finishes (and releases all its VRAM at process exit) before the
next starts, so peak VRAM is never live at the same time:

| Phase | Env | ~VRAM | Work |
|-------|-----|-------|------|
| 1 — dump | jasna | ~9 GB | decode + detect + BasicVSR++ primary restoration; serialize every clip's 256px crops + masks + geometry to a **bundle** on disk. blend/encode is throwaway. |
| 2 — FlashVSR 4x | FlashVSR | 12–16 GB | upscale each clip's 256px crops to 1024px, write them back into the bundle. |
| 3 — reblend | jasna | light | re-decode the source, re-assemble the restore results from the bundle, blend the 1024px crops back in, and encode the final output. |

Phase 1 and Phase 3 run as `jasna --flashvsr-phase {dump,reblend}` subprocesses
(dispatched in `jasna/__main__.py` before the multiprocessing guard, mirroring
`--compile-engines`). Phase 2 runs `jasna/restorer/flashvsr_phase2_driver.py`
under the FlashVSR virtualenv's Python — a standalone script with no jasna import.

The **bundle** is a directory of numpy/JSON files (`manifest.json`, one
`clip_<track>_<start>.npz` per clip plus a `_fvsr.npz` written by Phase 2). It is
persistent when you pass `--flashvsr-bundle-dir`, so a run that fails partway can
be resumed from the phase that failed (completed clips are skipped).

The geometry the blend needs (`scale_offsets`) is derived from the restored
frame's actual size at blend time, so FlashVSR's 4x output re-blends with **zero
metadata rewrite**.

## Requirements

FlashVSR is **not** bundled. You provide a checkout of the
[`lihaoyun6/FlashVSR_plus`](https://github.com/lihaoyun6/FlashVSR_plus) fork with
its weights and its own virtualenv, then point jasna at it with `--flashvsr-repo`.

### Setting up the FlashVSR checkout (one-time)

These are the exact steps verified on an RTX 5080 (sm120, 16 GB), Linux, CUDA
13.0 — they produce torch 2.13.0+cu130 / triton 3.7.1:

```bash
# 1. Clone the fork jasna targets. This also brings models/posi_prompt.pth,
#    which is tracked in the repo (not downloaded).
git clone https://github.com/lihaoyun6/FlashVSR_plus
cd FlashVSR_plus

# 2. Create a uv-managed *standalone* Python venv. This is mandatory. FlashVSR's
#    Triton Sparse_SageAttention kernel is JIT-compiled at runtime and needs the
#    Python dev headers (Python.h); a system or conda Python does not ship them and
#    the JIT dies with a "fatal error: Python.h" — uv's managed Python includes them.
uv venv --python 3.13 --python-preference only-managed

# 3. Install FlashVSR's dependencies into that venv from the CUDA wheel index that
#    matches your CUDA (jasna is verified on cu130; use .../whl/cu128 for CUDA 12.8).
uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130

# 4. Weights (~6.5 GB) live under models/FlashVSR-v1.1/. The FIRST run auto-downloads
#    them from HuggingFace, so this step is optional — pre-fetch it if you would
#    rather not download during jasna's Phase 2:
.venv/bin/huggingface-cli download JunhaoZhuang/FlashVSR-v1.1 --local-dir models/FlashVSR-v1.1

# 5. (Recommended) smoke-test the FlashVSR env on its own before wiring jasna in.
#    This exercises the exact tiny / sage / bf16 4x path jasna's Phase 2 uses and
#    triggers the weight download if you skipped step 4:
.venv/bin/python run.py -i ./inputs/example0.mp4 -s 4 -v 11 -m tiny -d cuda:0 -t bf16 -a sage ./_smoke
```

Notes:
- The `sageattention` pip package is **not** required — the fork vendors the
  `sparse_sage` kernel that `-a sage` uses; its optional `sageattention` import is
  guarded.
- After this, `<repo>/models/FlashVSR-v1.1/` holds `diffusion_pytorch_model_streaming_dmd.safetensors`,
  `Wan2.1_VAE.pth`, `LQ_proj_in.ckpt`, `TCDecoder.ckpt`, and `<repo>/models/posi_prompt.pth`
  sits alongside — that is exactly what `--flashvsr-repo` expects.

### Pointing jasna at it

- `--flashvsr-repo <path>` (required): the `FlashVSR_plus` checkout from above.
- `--flashvsr-python <path>` (default `<repo>/.venv/bin/python`): the uv-managed
  venv's Python from step 2.
- `--flashvsr-model-dir <path>` (default `<repo>/models/FlashVSR-v1.1`): the weights.

## Usage

```bash
jasna --input in.mp4 --output out.mkv \
      --secondary-restoration flashvsr \
      --flashvsr-repo ~/FlashVSR_plus \
      --log-level info
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--flashvsr-repo` | (required) | Path to the `FlashVSR_plus` checkout. |
| `--flashvsr-python` | `<repo>/.venv/bin/python` | FlashVSR env Python (uv-managed standalone venv). |
| `--flashvsr-model-dir` | `<repo>/models/FlashVSR-v1.1` | FlashVSR weights directory. |
| `--flashvsr-version` | `11` | Model version (`10` or `11`). |
| `--flashvsr-dtype` | `bf16` | Compute dtype (`fp16` / `bf16`). |
| `--flashvsr-max-clip-frames` | `32` | Caps Phase 1 `--max-clip-size` so each clip fits FlashVSR tiny-mode VRAM. |
| `--flashvsr-unload-dit` / `--no-flashvsr-unload-dit` | on | Offload the FlashVSR DiT before VAE decode (saves VRAM). |
| `--flashvsr-tiled-vae` / `--no-flashvsr-tiled-vae` | on | Tile the FlashVSR VAE decode (saves VRAM). |
| `--flashvsr-bundle-dir` | temp | Persist the intermediate bundle here (enables stage resume). |
| `--flashvsr-keep-bundle` | off | Keep the bundle after completion (implied by `--flashvsr-bundle-dir`). |

FlashVSR is fixed at 4x; there is no `--flashvsr-scale`.

### Why the clip-length cap

Phase 2 uses FlashVSR **tiny** mode, which holds every latent frame in VRAM and
returns a lossless tensor. The pilot measured ~13.5 GB at 21 frames and near-OOM
at 65 frames on a 16 GB card, so jasna caps the primary clip length
(`--flashvsr-max-clip-frames`, default 32) to keep every clip within that budget.
This is why FlashVSR-mode clips are shorter than a normal run's; the crossfade at
clip seams handles the extra boundaries. Raising the cap risks OOM in Phase 2.

## Disk space

The bundle is dominated by Phase 2's **uncompressed 1024px output**: every
restored crop-frame is 1024×1024×3 ≈ **3 MiB**, whereas the whole 256px primary
dump for a clip is only ~3 MiB. So bundle size tracks the number of mosaic
crop-frames and grows with video length:

- Rule of thumb: **~4 MB per mosaic-containing source frame** — roughly **~8 GB
  per minute** of 30 fps footage that is mosaiced throughout (proportionally less
  when only part of the timeline has mosaic).
- Measured: a 6-minute / 10,661-frame clip, mosaic throughout, 510 clips →
  **~46 GB** (1024px output ~45 GB + 256px dump ~1.6 GB).
- Feature-length or heavily-mosaiced videos can therefore need **hundreds of GB**.

Peak usage is the **full** bundle: Phase 2 writes every clip's 1024px crops before
Phase 3 begins, so they all coexist on disk at once.

> ⚠️ **The default bundle lives under the system temp dir (`/tmp`), which on Linux
> is often `tmpfs` (RAM-backed) and only tens of GB.** Writing a large bundle there
> fills `/tmp` / exhausts RAM and the run fails. For anything past a short clip,
> pass `--flashvsr-bundle-dir <path>` pointing at a real disk with room for the
> full bundle — size it as roughly *minutes of mosaic × 8 GB*. This also makes the
> run resumable.

jasna guards this automatically: before Phase 1 it warns if the bundle dir is on
tmpfs and prints free space vs a worst-case estimate; and after Phase 1 — once the
real clip count is known — it computes the exact 1024px output size and **aborts
before the expensive Phase 2** if it won't fit (keeping the bundle so you can point
`--flashvsr-bundle-dir` at a bigger disk and resume).

## Limitations

- **File output only.** Not compatible with `--stream`, folder/image input, or
  `--frame-gen` (run frame generation as a separate pass on the output).
- **Double encode.** Phase 1 encodes a throwaway output so it can run through the
  fully-tested pipeline unchanged; the final encode happens in Phase 3. This adds
  one extra encode pass over a normal run.
- **Not bundled / supporter-independent.** FlashVSR is a third-party model with
  its own license; you supply the checkout, weights, and venv. It is unrelated to
  the jasna supporter models.

## Inline mode (`--secondary-restoration flashvsr-inline`)

Uses the same FlashVSR checkout / weights / venv and the same `--flashvsr-*`
flags (`repo` / `python` / `model-dir` / `version` / `dtype`) as the offline
path, but creates **no intermediate files** and runs FlashVSR as a secondary
restorer inside jasna's normal streaming pipeline.

```bash
jasna --input in.mp4 --output out.mkv \
      --secondary-restoration flashvsr-inline \
      --flashvsr-repo ~/FlashVSR_plus \
      --log-level info
```

### Differences from offline

| | `flashvsr` (offline 3-phase) | `flashvsr-inline` |
|---|---|---|
| Path | 3 processes: dump → FlashVSR → reblend | single streaming pass |
| Intermediate files | 256px + 1024px bundle (tens of GB) | **none** |
| Encodes | 2 (throwaway + final) | 1 |
| FlashVSR mode | tiny (O(T), ~12–16 GB) | **tiny-long (O(1), ~11.9 GB)** |
| VRAM | phases non-concurrent, so effectively tiny alone | **co-resident** with primary (~14.8 GB measured on a 16 GB card) |
| FlashVSR checkout | no patch needed | **requires the tiny-long multi-chunk fix** |
| Staged resume | yes (persistent bundle) | no (single pass) |
| Progress / cancel / GUI | 3-phase flow | same as any secondary |

### Prerequisite: the tiny-long patch

Inline uses **tiny-long** for constant (O(1)) VRAM. FlashVSR_plus's tiny-long has a
known bug that crashes on the second chunk (`8192 vs 4096` error), so a **patched
checkout is required**. jasna checks the checkout at startup and stops with an
explicit error (pointing you to the offline `flashvsr` mode, which uses tiny and
needs no patch) if it is unpatched.

The patch ships at
[`patches/flashvsr_plus_tinylong_multichunk_fix.patch`](../patches/flashvsr_plus_tinylong_multichunk_fix.patch).
Apply it to the FlashVSR_plus checkout:

```bash
cd ~/FlashVSR_plus
git apply /path/to/jasna/patches/flashvsr_plus_tinylong_multichunk_fix.patch
```

All it does is disable two per-chunk cache clears (remove the per-chunk
`LQ_proj_in.clear_cache()` and `TCDecoder.clean_mem()` in
`src/pipelines/flashvsr_tiny_long.py`; the once-per-video reset before the loop
stays).

### Behavior and constraints

- **Forces clip 32 and frame-gen off** (same reasons as offline); `--max-clip-size`
  is rounded down to 32 automatically.
- **Auto-enables fp8-recon** (when unset) to shrink the primary peak ~0.9–1.7 GB so
  it fits the co-residence budget; falls back to TRT if the GPU can't do fp8
  (sm89+ / `--fp16`).
- Synchronous. FlashVSR (~15 crop-fps) is the rate limiter, so mosaic-heavy stretches
  run at that speed (mosaic-free frames stay fast on the primary alone). Because
  FlashVSR dominates wall-clock, lowering `--batch-size` costs almost nothing.
- VRAM, on a **16 GB card with a desktop resident**: ~14.8 GB combined at 480p, but
  **1080p+ runs right at the physical ceiling** (measured ~15.8 GB peak). It stays up
  because the worker's `expandable_segments` allocator and jasna's `vram_offloader`
  (which spills queued frames to system RAM) absorb the pressure — expect
  `expandable_segments: memory mapping failed with OOM` **warnings** (benign; not a
  crash) and heavy offloading at 1080p. For margin at 1080p+, use **`--batch-size 2`**
  (or `1`) and/or disable MPS (frees ~490 MB). Use the offline `flashvsr` mode for
  GPUs with less VRAM, an unpatched checkout, or if 1080p is routinely this tight.
- **On Windows, inline does not fit a 16 GB card** (`expandable_segments` is
  unsupported there, so the worker's reserved VRAM balloons to ~13 GB). See
  [Windows notes](#windows-notes).

### Implementation

- Synchronous `SecondaryRestorer`:
  `jasna/restorer/flashvsr_inline_secondary_restorer.py` (spawns a resident FlashVSR
  venv worker, length-prefixed RGB wire, `close()` to shut down).
- Worker (runs under the FlashVSR venv, no jasna import):
  `jasna/restorer/flashvsr_inline_worker.py` (tiny-long pipe, lossless tensor capture
  by replacing `imageio.get_writer`, next_8n5 padding to absorb small clips and return
  exactly T frames).
- CLI wiring: `jasna/main.py`. Tests: `tests/test_flashvsr_inline.py`.

## Windows notes

Verified on Windows 11 / RTX 5080 16 GB / torch 2.13.0+cu130 (FlashVSR venv). Bottom
line: **on a 16 GB card, only offline (`flashvsr`) is recommended; inline does not
have the VRAM.**

- **PyTorch's `expandable_segments` is unsupported on Windows** (it warns and falls
  back to the default caching allocator). tiny-long's reserved VRAM runs **+1–2 GB**
  above the Linux "flat ~11.9 GB" figure due to fragmentation.
  `backend:cudaMallocAsync` does not help (measured slightly worse). jasna therefore
  does not set `expandable_segments` for the worker on Windows.
- **The WDDM desktop holds ~1 GB** (near zero on headless Linux), leaving
  **~15.2 GB effective** on a 16 GB card.
- Measured peaks (scale 4 / tiny-long / bf16 / sage / 85 frames, reserved):
  - **256px input (jasna's real workload): ~13.0 GB** — Phase 2 has the GPU to
    itself, so offline works on 16 GB Windows.
  - **384px input (the bundled example0 smoke): ~15.1 GB** — razor-thin against the
    effective free VRAM; a browser or IDE holding a few hundred MB tips it into OOM.
    **A smoke-test OOM does not imply jasna's real workload OOMs.**
- **An 85-frame smoke with `-m tiny` (O(T) memory) is expected to OOM on 16 GB
  Windows.** Smoke-test with `-m tiny-long` instead (substitute it in the step-5
  setup command).
- The venv Python lives at `<repo>/.venv/Scripts/python.exe` (the
  `--flashvsr-python` default resolves there on Windows).
- When stdout goes to a pipe (redirection / some GUI launches), FlashVSR's
  block-character startup banner raises `UnicodeEncodeError` under cp932 before
  inference starts. jasna-spawned runs (offline Phase 2 / the inline worker) set
  `PYTHONUTF8=1` automatically; **when invoking run.py by hand with redirected
  output, set `$env:PYTHONUTF8=1` first**.
- run.py does not create the output directory (a missing one fails with
  `FileNotFoundError` after inference) — `mkdir` it beforehand.

## Implementation (offline)

- Orchestrator, bundle format, Phase 1 dump hook, Phase 3 reblend:
  `jasna/restorer/flashvsr_offline.py`.
- Phase 2 driver (FlashVSR venv): `jasna/restorer/flashvsr_phase2_driver.py`.
- Subprocess dispatch: `jasna/__main__.py` (`--flashvsr-phase`).
- CLI wiring / early dispatch: `jasna/main.py`.
- Tests: `tests/test_flashvsr_offline.py`, `tests/test_main.py`.

Reused jasna assets: `BlendBuffer` / `crop_buffer.scale_offsets` (the 1024px
crops re-blend unchanged), `RestorationPipeline.build_secondary_result` (the
`[keep_start:keep_end]` slice), `pipeline_items` (the serialization units), and
`media/backend.make_video_{reader,encoder}` for Phase 3 decode/encode.
