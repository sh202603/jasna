# FlashVSR offline secondary restoration (`+modi`)

`--secondary-restoration flashvsr` upscales each restored 256px mosaic crop to
1024px (4x) with [FlashVSR](https://github.com/OpenImagingLab/FlashVSR)
(one-step streaming diffusion VSR; jasna uses the
[`lihaoyun6/FlashVSR_plus`](https://github.com/lihaoyun6/FlashVSR_plus) fork) to
recover texture realism the primary BasicVSR++ model leaves blurry on large
mosaic regions, close-ups, and 4K sources.

Unlike the other secondary restorers (unet-4x / RTX Super-Res / TVAI), FlashVSR
runs **offline in three separate processes** rather than inline in the streaming
pipeline. It peaks at **12–16 GB VRAM on its own**, so it cannot co-reside with
jasna's ~9 GB primary pipeline on a 16 GB card. Splitting the work across
processes whose peak VRAM never overlaps in time is what makes it fit.

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

## Implementation

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
