# FlashVSR secondary restoration (`+modi`)

`--secondary-restoration flashvsr` / `flashvsr-inline` upscale each restored
256px mosaic crop with [FlashVSR](https://github.com/OpenImagingLab/FlashVSR)
(one-step streaming diffusion VSR; jasna uses the
[`sh202603/FlashVSR_plus`](https://github.com/sh202603/FlashVSR_plus) fork) to
recover texture realism the primary BasicVSR++ model leaves blurry on large
mosaic regions, close-ups, and 4K sources. The crop is processed at 1024px (4x,
the model's native factor) or, with `--flashvsr-scale 2`, at 512px; either way
the blend shrinks it back onto the frame, so the output resolution never
changes. The output crops are always color-corrected against the primary
restoration they came from (see [Color correction](#color-correction)).

FlashVSR has two modes. Both are supported; pick by setup:

| | `flashvsr-inline` (single pass) | `flashvsr` (offline 3-phase) |
|---|---|---|
| Fits | the `basicvsrpp` primary on a 16 GB card: one pass, **no intermediate files, no disk gate, no double encode** | the SeedVR2 primary (the maximum-quality stack), 12 GB-class GPUs, long sources that need staged resume |
| FlashVSR pipeline | tiny-long (VRAM independent of clip length; the recommended fork works as-is, an upstream checkout **requires the tiny-long patch**) | tiny (no patch needed) |
| With `--restoration-model-name seedvr2` | rejected at startup (two resident workers exceed 16 GB) | allowed |

Most of this document describes the offline mode; inline is covered in
"Inline mode" at the end.

Why offline 3-phase exists: FlashVSR's tiny mode peaks at **12–16 GB VRAM on its
own**, so it cannot co-reside with jasna's primary pipeline on a 16 GB card.
Splitting the work across processes whose peak VRAM never overlaps in time is what
makes it fit. Inline mode instead uses FlashVSR's **tiny-long** (constant ~11.9 GB),
co-residing with the primary (~1.6 GB under fp8-recon) to run
as a single pass.

## How it works — offline 3-phase

A single `--secondary-restoration flashvsr` command runs three subprocesses in
sequence. Each finishes (and releases all its VRAM at process exit) before the
next starts, so peak VRAM is never live at the same time:

| Phase | Env | ~VRAM | Work |
|-------|-----|-------|------|
| 1 (dump) | jasna | ~9 GB | decode + detect + BasicVSR++ primary restoration; serialize every clip's 256px crops + masks + geometry to a **bundle** on disk. blend/encode is throwaway. |
| 2 (FlashVSR) | FlashVSR | 12–16 GB | upscale each clip's 256px crops to 1024px (512px at `--flashvsr-scale 2`), color-correct them, write them back into the bundle. |
| 3 (reblend) | jasna | light | re-decode the source, re-assemble the restore results from the bundle, blend the upscaled crops back in, and encode the final output. |

Phase 1 and Phase 3 run as `jasna --flashvsr-phase {dump,reblend}` subprocesses
(dispatched in `jasna/__main__.py` before the multiprocessing guard, mirroring
`--compile-engines`). Phase 2 runs `jasna/restorer/flashvsr_phase2_driver.py`
under the FlashVSR virtualenv's Python — a standalone script with no jasna import.

The **bundle** is a directory of numpy/JSON files (`manifest.json`, one
`clip_<track>_<start>.npz` per clip plus a `_fvsr.npz` written by Phase 2). It is
persistent when you pass `--flashvsr-bundle-dir`, so a run that fails partway can
be resumed from the phase that failed (completed clips are skipped).

The geometry the blend needs (`scale_offsets`) is derived from the restored
frame's actual size at blend time, so FlashVSR's output re-blends with **zero
metadata rewrite** at either scale.

## Requirements

FlashVSR is **not** bundled. You provide a FlashVSR checkout with its weights and
its own virtualenv, then point jasna at it with `--flashvsr-repo`.

Use the fork [`sh202603/FlashVSR_plus`](https://github.com/sh202603/FlashVSR_plus)
(default branch `modi`). It is upstream
[`lihaoyun6/FlashVSR_plus`](https://github.com/lihaoyun6/FlashVSR_plus) plus the
following, and it is what jasna is verified against:

- It includes the tiny-long multi-chunk fix that inline needs (no patch).
- It has the `--flashvsr-accel` speed-up (see [Acceleration](#acceleration---flashvsr-accel)).
- A single `uv sync` installs the dependencies at the versions pinned in `uv.lock`.

An upstream checkout still works, but without acceleration, and inline needs a
patch (see [Using an upstream checkout](#using-an-upstream-checkout)).

### Setting up the FlashVSR checkout (one-time)

`uv.lock` pins the dependencies, so you get torch 2.13.0+cu130 / triton 3.7.1
(triton-windows on Windows) / nvidia-cudnn-frontend 1.29.0:

```bash
# 1. Clone the fork (default branch modi). This also brings models/posi_prompt.pth,
#    which is tracked in the repo (not downloaded).
git clone https://github.com/sh202603/FlashVSR_plus
cd FlashVSR_plus

# 2. Create .venv and install the dependencies in one command. torch / torchvision
#    come from the PyTorch cu130 index, as configured in pyproject.toml.
#    The Python must ship the dev headers (Python.h). This is mandatory: FlashVSR's
#    Triton Sparse_SageAttention kernel is JIT-compiled at runtime against them, and
#    a header-less system or conda Python dies with "fatal error: Python.h" (or,
#    worse, tiny-long silently returns 0 frames). The fork does not pin a Python
#    (requires-python >=3.10, no .python-version), so ask for a uv-managed
#    standalone Python explicitly; a system Python with its -dev package (e.g.
#    python3.13-dev) also works. CAUTION: if uv itself runs inside a snap-confined
#    app (e.g. snap VSCode), its managed Pythons land under a snap revision path and
#    the venv dies on the next snap refresh; prefer an explicit stable interpreter
#    path then:
uv sync --python 3.13 --python-preference only-managed     # or: uv sync --python /usr/bin/python3.13

# 3. Weights (~6.5 GB) live under models/FlashVSR-v1.1/. The FIRST run auto-downloads
#    them from HuggingFace, so this step is optional — pre-fetch it if you would
#    rather not download during a jasna run:
.venv/bin/huggingface-cli download JunhaoZhuang/FlashVSR-v1.1 --local-dir models/FlashVSR-v1.1

# 4. (Recommended) smoke-test the FlashVSR env on its own before wiring jasna in.
#    This runs the tiny-long / sage / bf16 path inline uses, at scale 2, and checks
#    acceleration with --accel. It triggers the weight download if you skipped
#    step 3. run.py does not create its output folder, so create it first:
mkdir -p _smoke
.venv/bin/python run.py -i ./inputs/example0.mp4 -s 2 -v 11 -m tiny-long -d cuda:0 -t bf16 -a sage --accel ./_smoke
```

Notes:
- If step 4 logs `[FlashVSR] accel: enabled fp8_conv_lq, fp8_dit, fused_dit.`,
  acceleration works on this machine. On RTX 30 series and older GPUs it logs
  `... disabled: needs an FP8-capable GPU ...` and runs the standard path (expected).
- The `sageattention` pip package is **not** required — the fork vendors the
  `sparse_sage` kernel that `-a sage` uses; its optional `sageattention` import is
  guarded.
- After this, `<repo>/models/FlashVSR-v1.1/` holds `diffusion_pytorch_model_streaming_dmd.safetensors`,
  `Wan2.1_VAE.pth`, `LQ_proj_in.ckpt`, `TCDecoder.ckpt`, and `<repo>/models/posi_prompt.pth`
  sits alongside — that is exactly what `--flashvsr-repo` expects.
- On Windows the venv Python is `.venv\Scripts\python.exe` (adjust the
  `.venv/bin/...` paths in steps 3 and 4). More in [Windows notes](#windows-notes).
- When updating an existing checkout, run `uv sync` again after `git pull`
  (dependencies such as `nvidia-cudnn-frontend`, used by acceleration, were added).

### Using an upstream checkout

To use upstream [`lihaoyun6/FlashVSR_plus`](https://github.com/lihaoyun6/FlashVSR_plus),
replace steps 1 and 2 with the following (there is no `uv.lock`, so versions are not
pinned):

```bash
git clone https://github.com/lihaoyun6/FlashVSR_plus
cd FlashVSR_plus
uv venv --python 3.13 --python-preference only-managed
uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130   # .../whl/cu128 for CUDA 12.8
```

Such a checkout differs from the fork in two ways:

- Inline (`flashvsr-inline`) needs the tiny-long patch (see
  [Prerequisite: the tiny-long fix](#prerequisite-the-tiny-long-fix)). Offline
  (`flashvsr`) works without it.
- `--flashvsr-accel` is unavailable. jasna warns and runs at standard speed.

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
| `--flashvsr-scale` | `4` | Processing scale for both modes: `4` = model-native 1024px, `2` = 512px (faster, lower VRAM). See [Processing scale](#processing-scale---flashvsr-scale). |
| `--flashvsr-accel` / `--no-flashvsr-accel` | off | Both modes: use the fork's acceleration (FP8 and fused kernels). Needs an RTX 40 series or newer GPU; anything else falls back to the standard path automatically. See [Acceleration](#acceleration---flashvsr-accel). |
| `--flashvsr-max-clip-frames` | `90` | Offline only: cap on Phase 1 `--max-clip-size` (tiny-mode VRAM). Inline uses `--max-clip-size` as-is. |
| `--flashvsr-unload-dit` / `--no-flashvsr-unload-dit` | on | Offload the FlashVSR DiT before VAE decode (saves VRAM). |
| `--flashvsr-tiled-vae` / `--no-flashvsr-tiled-vae` | on | Tile the FlashVSR VAE decode (saves VRAM). |
| `--flashvsr-tiles` | `1` | Inline only: split the DiT inference into horizontal strips (`2`–`4`) to cut peak VRAM. The offline path ignores it. See [Strip tiling](#strip-tiling---flashvsr-tiles). |
| `--flashvsr-bundle-dir` | temp | Persist the intermediate bundle here (enables stage resume). |
| `--flashvsr-keep-bundle` | off | Keep the bundle after completion (implied by `--flashvsr-bundle-dir`). |

Color correction has no flag: it is always on (below).

### Processing scale (`--flashvsr-scale`)

FlashVSR is a 4x model: a 256px crop is bicubic-pre-upscaled by the scale and
the DiT then restores it at that size, so `4` processes at the model-native
1024px. `--flashvsr-scale 2` processes at 512px instead. Because jasna's blend
derives the crop geometry from the restored frame's actual size, both scales
re-blend onto the frame with no other change and the output video resolution is
the same either way.

Scale 2 is an opt-in trade: the model is 4x-trained, so it runs off its training
factor, but it is much cheaper. Measured on the lada-ex implementation this
worker is kept identical with (RTX 5080 16 GB, 480p, `--tensorrt`): scale 2 at
tiles 1 ran about **5x faster** than scale 4 at tiles 2 (96 s vs 449 s end to
end) with the whole-GPU peak about **4 GB lower** (11.3–11.5 GB vs
14.8–15.1 GB), and it passed the same gates (flow-warping error ratio 1.128 vs
the ≤1.2 gate; visual A/B judged clean). The default stays at 4 (the
model-native factor the original quality gates were run at).

jasna's own measurements (RTX 5080 16 GB, Linux, `small-01.mp4` 480p / 4930
frames with mosaic throughout, the default clip 90 / overlap 8; output frame
count = input in every run; wall clock is the whole command, VRAM is the
whole-GPU `nvidia-smi` peak):

| Mode | Scale / tiles | Wall clock | FlashVSR time | Peak VRAM | Notes |
|---|---|---|---|---|---|
| primary only | — | 20 s | — | 3.8 GB | reference (fp8-recon) |
| inline | 4 / 2 | 491 s | 479 s | 13.7 GB | |
| inline | 2 / 1 | 113 s | 100 s | 10.1 GB | **4.8x faster, 3.6 GB lower** than 4 / 2 |
| inline, 1080p (`test-flashvsr-fhd-02`, 4203 f) | 2 / 1 | 156 s | 146 s | 11.3 GB | no offloads, no allocator warnings |
| offline | 4 | 477 s | Phase 2 ~430 s | 13.2 GB | bundle 9.7 GB |
| offline | 2 | 194 s | Phase 2 ~150 s | 7.8 GB | |

Earlier builds capped clips at 32 frames, and FlashVSR time was about twice the
table above (841 s inline 4 / 2, 163 s inline 2 / 1, 864 s offline 4). Against
the 8-frame overlap a 32-frame clip advances only 16 frames, so the DiT frame
count was 1.8x and the call count 3x; the worker itself is unchanged (the A/B
against the pre-port worker matched, and the color correction costs +6–7 %).
Lifting the cap adds only the primary's queue frames in VRAM (+0.1–0.3 GB at
480p, +0.7 GB at 1080p).

### Acceleration (`--flashvsr-accel`)

`--flashvsr-accel` turns on the fork's acceleration (the same as the fork's
`--accel`) in both modes. In the fork's measurements (RTX 5060 Ti 16 GB,
tiny-long, 90 frames) FlashVSR ran 1.37x faster at scale 2 and 1.41x faster at
scale 4, and FlashVSR's own peak allocation dropped by about 1.4 GiB. It is off by
default.

It replaces three parts, each checked at startup:

| Part | What changes |
|---|---|
| `fp8_conv_lq` | LQ projector convolutions in FP8 (cuDNN graph API) |
| `fp8_dit` | DiT linears and FFN in FP8 |
| `fused_dit` | DiT RMSNorm + RoPE and AdaLN as fused Triton kernels |

FP8 for the VAE decoder (TCDecoder) is not included. The TCDecoder produces the
output pixels directly, and FP8's coarse mantissa turns smooth gradients such as
skin into steps, which show up as banding.

**Requirements:** an RTX 40 series or newer GPU (sm89+), `--flashvsr-version 11`
and `--flashvsr-dtype bf16` (both defaults), and the fork's checkout. The FP8
convolutions need cuDNN 9.17 or newer, which the fork's torch (cu130) bundles.

**Automatic fallback:** a part that can't run is dropped by the startup checks
(GPU, dtype, libraries, a trial build, a warmup) and the standard path runs
instead. If every part is dropped, the output is bit-identical to a run without
`--flashvsr-accel`. If a part fails during the run, it is switched back to the
standard path for the rest of the run and the clip is redone once (by the worker
inline, by the Phase 2 driver offline). Out-of-VRAM (OOM) errors do not drop a part.

**How to check it:** the inline worker's stdout is discarded, so the worker sends
its startup decisions to jasna, which logs them. With `--log-level info`:

```text
[flashvsr-inline] [FlashVSR] accel: enabled fp8_conv_lq, fp8_dit, fused_dit.
[flashvsr-inline] worker ready (acceleration: fp8_conv_lq, fp8_dit, fused_dit)
```

A dropped part is logged as a warning with the reason (`... disabled: <reason>;
using the standard path.`), and so is a part switched off during the run. Both
show at `--log-level warning` or more verbose (not at the default `error`).
Offline, the fork's log appears as-is in Phase 2's output.

**Output:** it differs slightly from the standard path (FP8 rounding, amplified by
the sparse attention). In the fork's checks the flow-warping error stayed within
1.07x of the standard path (quality gate: 1.2 or less), and a visual A/B found it
equivalent.

**Measured in jasna:** Windows 11 / RTX 5060 Ti 16 GB, 1080p (`test7-short.mp4`,
6242 frames), inline scale 2 / tiles 1. The two runs were measured back to back;
the whole-GPU peaks include the desktop's ~3.2 GB:

| | Wall clock | Whole-GPU peak |
|---|---|---|
| without `--flashvsr-accel` | 577 s | 12191 MiB |
| with `--flashvsr-accel` | **463 s (1.25x faster)** | **11311 MiB (−880 MiB)** |

Both produced 6242 frames with zero offloads and zero worker retries, and a
visual A/B of the two played side by side found no quality difference.

Scale 4 / tiles 1 (same 1080p source, with acceleration) was measured too. It
completed (1555 s, 6242 frames, zero OOMs, retries and offloads), 1.62x faster than
tiles 2 without acceleration (2519 s). But the whole-GPU peak was 15874 MiB, only
437 MiB below the ceiling, and that was with a light desktop (1.36 GB resident); a
desktop above 2 GB would not fit. So on 16 GB cards, keep `--flashvsr-tiles 2` for
scale 4 even with acceleration. The
wall-clock gain is smaller than the fork's per-clip 1.37x. Part of it is that the
primary pipeline and decode/encode time don't change (without acceleration,
FlashVSR is about 90% of the wall clock), but that alone would give about 1.32x,
so sharing the GPU with the primary likely costs some as well.

Notes:
- The per-part variables (`FLASHVSR_FP8_CONV` / `FLASHVSR_FP8_DIT` /
  `FLASHVSR_FUSED_DIT`) are passed through to the worker. To drop one part for an
  A/B check, pass it per command: `FLASHVSR_FP8_DIT=0 jasna ...`.
  `--no-flashvsr-accel` (the default) removes `FLASHVSR_ACCEL`, so a
  `FLASHVSR_ACCEL=1` left in the shell does not turn acceleration on.
- When resuming offline from a bundle, completed clips are reused. Resuming with a
  different acceleration setting mixes accelerated and standard clips (clips are
  independent, so there are no seams); start a new bundle if you want them uniform.
- With an upstream checkout jasna warns that acceleration is unavailable and runs
  at standard speed.

### Color correction

FlashVSR's generated crops can drift in tone from the primary restoration they
were built from; after the blend that reads as a color mismatch between the
restored region and its surroundings. Both modes therefore always correct each
output crop against the **bicubic-upscaled input crop** (the primary output),
using a wavelet reconstruction: the output keeps FlashVSR's high frequencies
(texture) on the input's low frequencies (local tone). It is applied once per
clip on the whole crop (never per strip), before quantization, by the same
function in both modes.

Measured as the median per-channel |Δmean| inside the pixels the secondary
changed (8-bit, vs the primary-only output of the same clip regime): on
lada-ex, no correction 5.28 → AdaIN 0.98 → **wavelet 0.34** (0.32 at scale 2);
on jasna at scale 2 (480p `small-01`), inline 1.76 → 0.43 → **0.24** and offline
1.61 → **0.24** (the two modes land on the same value, as expected from sharing
the function). Upstream FlashVSR_plus has its own `color_fix`, but jasna does
not use it: its call is wrapped in a bare `except: pass`, so a failure is
indistinguishable from "off".

Windows (RTX 5060 Ti, 1080p, inline scale 2 / tiles 1) confirms both the effect and
its cost. The metric there is a separate implementation — the restored-region mask is
taken from the `none` output, and `none` and `wavelet` are then measured on identical
frames through that same mask — so its absolute values are not comparable with the
table above, but none 5.88 → **wavelet 3.20** (ratio 0.54), with wavelet below none on
**91.1%** of the 135 paired frames. The cost is **+5.3%** of FlashVSR time (wavelet
524.5 s vs none 498.3 s), matching the +6-7% measured on Linux.

There is no CLI flag. For A/B verification only, the environment variable
`JASNA_FLASHVSR_COLOR_FIX=adain|wavelet|none` overrides the method in both modes.
Leaving it set in the shell silently applies it to every later run, so pass it per
command: `JASNA_FLASHVSR_COLOR_FIX=none jasna ...`.

### Clip length

Clip length is set by `--max-clip-size` (default 90), as in any run.

- **Inline**: no cap. The worker's tiny-long is flat in VRAM with respect to
  the clip length, and the primary's cost is the same as without FlashVSR
  ([tuning](tuning.md)).
- **Offline**: Phase 2's tiny mode scales with the clip length in principle, so
  `--flashvsr-max-clip-frames` (default 90) caps it. Measured, the Phase 2 peak
  is flat from 32 to 90 (13.1 → 13.2 GB at scale 4, 7.8 GB unchanged at scale
  2); above 90 is unmeasured. Lower the value if Phase 2 OOMs.

## Disk space

The bundle is dominated by Phase 2's **uncompressed upscaled output**: every
restored crop-frame is 1024×1024×3 ≈ **3 MiB** at the default scale 4 (a quarter
of that, 0.75 MiB, at `--flashvsr-scale 2`), whereas the whole 256px primary
dump for a clip is only ~3 MiB. So bundle size tracks the number of mosaic
crop-frames and grows with video length (figures below are for scale 4):

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
real clip count is known — it computes the exact output size for the selected
scale and **aborts before the expensive Phase 2** if it won't fit (keeping the
bundle so you can point `--flashvsr-bundle-dir` at a bigger disk and resume).

## Limitations

- **File output only.** Not compatible with `--stream`, folder/image input, or
  `--frame-gen` (run frame generation as a separate pass on the output).
- **No fps retargeting, smart rendering, or VR.** `--retarget-high-fps` (the
  Phase 1 frame stride would misalign the Phase 3 reblend indices), `--segments`,
  and VR processing (`--vr-mode sbs`/`sbs-fisheye`, or `auto` when it detects VR
  content — Phase 3 has no VR projector) are rejected at startup.
- **Double encode.** Phase 1 encodes a throwaway output so it can run through the
  fully-tested pipeline unchanged; the final encode happens in Phase 3. This adds
  one extra encode pass over a normal run.
- **Not bundled / supporter-independent.** FlashVSR is a third-party model with
  its own license; you supply the checkout, weights, and venv. It is unrelated to
  the jasna supporter models.

## Inline mode (`--secondary-restoration flashvsr-inline`)

Uses the same FlashVSR checkout / weights / venv and the same `--flashvsr-*`
flags (`repo` / `python` / `model-dir` / `version` / `dtype` / `scale`) as the
offline path, but creates **no intermediate files** and runs FlashVSR as a
secondary restorer inside jasna's normal streaming pipeline. It is the mode for
the `basicvsrpp` primary on a 16 GB card; it cannot be combined with the SeedVR2
primary (two resident workers exceed 16 GB; use the offline mode for that stack).

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
| Intermediate files | 256px + upscaled bundle (tens of GB) | **none** |
| Encodes | 2 (throwaway + final) | 1 |
| FlashVSR mode | tiny (O(T), ~12–16 GB) | **tiny-long (O(1), ~11.9 GB)** |
| VRAM | phases non-concurrent, so effectively tiny alone | **co-resident** with primary (scale 4: ~13.7 GB at tiles 2, untiled hits the 16 GB ceiling; scale 2: ~10–11 GB) |
| FlashVSR checkout | no patch needed | the recommended fork works as-is; an upstream checkout **requires the tiny-long multi-chunk fix** |
| Staged resume | yes (persistent bundle) | no (single pass) |
| Progress / cancel / GUI | 3-phase flow | same as any secondary |

### Prerequisite: the tiny-long fix

Inline uses **tiny-long** for constant (O(1)) VRAM. Upstream FlashVSR_plus's
tiny-long has a known bug that crashes on the second chunk (`8192 vs 4096` error),
so a **checkout with the fix is required**. The recommended fork
([`sh202603/FlashVSR_plus`](https://github.com/sh202603/FlashVSR_plus)) includes it,
so there is nothing to do. jasna checks the checkout at startup and, if the fix is
missing, stops with an explicit error pointing you to the fork, the patch, and the
offline `flashvsr` mode (which uses tiny and needs no patch).

For an upstream checkout, apply the bundled patch
[`patches/flashvsr_plus_tinylong_multichunk_fix.patch`](../../patches/flashvsr_plus_tinylong_multichunk_fix.patch):

```bash
cd ~/FlashVSR_plus
git apply /path/to/jasna/patches/flashvsr_plus_tinylong_multichunk_fix.patch
```

All it does is disable two per-chunk cache clears (remove the per-chunk
`LQ_proj_in.clear_cache()` and `TCDecoder.clean_mem()` in
`src/pipelines/flashvsr_tiny_long.py`; the once-per-video reset before the loop
stays).

### Behavior and constraints

- **Forces frame-gen off** (same reason as offline).
- **Auto-enables fp8-recon** (when unset) to shrink the primary peak ~0.9–1.7 GB so
  it fits the co-residence budget; falls back to TRT if the GPU can't do fp8
  (sm89+ / `--fp16`).
- Synchronous. FlashVSR (~15 crop-fps) is the rate limiter, so mosaic-heavy stretches
  run at that speed (mosaic-free frames stay fast on the primary alone). Because
  FlashVSR dominates wall-clock, lowering `--batch-size` costs almost nothing.
- VRAM, on a **16 GB card with a desktop resident**, at the default scale 4:
  **do not run untiled on 16 GB**. Both 480p and 1080p sit at the physical
  ceiling (measured ~15.8 GB) with `expandable_segments: memory mapping failed
  with OOM` warnings and offloading, and the worker can genuinely OOM mid-clip
  (32 of 57 clips at 1080p / clip 90). The worker retries that clip once and
  stops if it fails again (earlier builds filled the rest of a failed clip by
  repeating its last frame, which showed up as ghosting). Use
  **`--flashvsr-tiles 2`** (next section; 13.7 GB at 480p, 14.7 GB at 1080p,
  zero offloads). Use the offline `flashvsr` mode for GPUs with less VRAM or
  an unpatched upstream checkout. **`--flashvsr-scale 2` changes the picture**: untiled,
  it peaks at 10.1 GB at 480p and 11.3 GB at 1080p on Linux (clip 90; zero
  offloads, zero allocator warnings), so it needs neither tiling nor the ceiling
  tricks; see
  [Processing scale](#processing-scale---flashvsr-scale) for the full table.
- **The missing `expandable_segments` on Windows only costs VRAM at scale 4.** There
  the worker's reserved VRAM balloons to ~13 GB, so untiled inline runs pinned to the
  physical ceiling (it completes, but with almost no headroom) and 1080p wants
  **`--flashvsr-tiles 2`**. Scale 2 takes no fragmentation hit — Windows measures
  *below* Linux (10.8 GB at 1080p, 5.5 GB of headroom) — so **tiles 1 is fine**.
  Measurements: [Windows notes](#windows-notes).

### Strip tiling (`--flashvsr-tiles`)

An inline-only VRAM lever. Each 256px crop is split along the height only, into
full-width horizontal strips; tiny-long runs per strip and the strips are
feather-blended back together. The DiT's token-activation memory (most of all the
block-sparse draft's attention mask) shrinks with the square of the tile area, so
peak VRAM drops for a modest compute increase. The offline path (`flashvsr`)
ignores this flag.

| `--flashvsr-tiles` | Strips | Attn mask (vs full) | Compute (vs full) |
|---|---|---|---|
| `1` (default) | none (single shot) | 1.0 | 1.0 |
| `2` | 2 (256w×160h each) | 0.39x | ~1.25x |
| `3` | 3 (256w×128h each) | 0.25x | ~1.5x |
| `4` | 4 (256w×96h each) | 0.14x | ~1.5x |

Fewer strips are both faster and better (less overlap compute, wider spatial
context per strip), so pick the **smallest count that fits your VRAM**: `2` if it
fits, `3` when still pinned at the ceiling or OOMing, `4` as the last step.

The table is for scale 4, where the strip height snaps to a 32-multiple (so the
4x-upscaled strip is the 128-multiple the DiT needs). At `--flashvsr-scale 2`
the snap is 64, so the strips round up larger (tiles `2` = 2 × 256w×192h with a
128 px overlap, `3` = 3 × 128h, `4` = 4 × 128h) and the overlap compute grows;
scale 2 rarely needs tiling in the first place, since its untiled peak already
sits well below the ceiling.

Quality: strip boundaries are feather-blended; hardware verification (Windows /
RTX 5080, same-frame comparison against tiles 1) found no banding, no steps, and
no per-strip color shift — differences stay within the diffusion model's
stochastic texture variation. As with the untiled path, the outermost 1px of the
output has zero weight (pre-existing behavior inherited from upstream run.py;
harmless in practice since the blend feathers crop borders).

### Implementation

- Synchronous `SecondaryRestorer`:
  `jasna/restorer/flashvsr_inline_secondary_restorer.py` (spawns a resident FlashVSR
  venv worker, length-prefixed uint8 BGR wire with the RGB flip on this side,
  `close()` to shut down).
- Worker (runs under the FlashVSR venv, no jasna import):
  `jasna/restorer/flashvsr_inline_worker.py` (tiny-long pipe, lossless tensor capture
  by replacing `imageio.get_writer`, next_8n5 padding to absorb small clips and return
  exactly T frames; the strip split, feather blend, the color correction and the
  relay of the acceleration decisions also live here). The file is kept **byte-identical with lada-ex's
  `flashvsr_worker.py`** (same policy as the SeedVR2 worker), which is why the wire
  is lada-native BGR; a FlashVSR_plus breakage is fixed once and diff-copied.
- CLI wiring: `jasna/main.py`. Tests: `tests/test_flashvsr_inline.py`.

## Windows notes

Two verification rigs: the original scale-4/tiling work on **Windows 11 / RTX 5080
16 GB**, and the scale-2 and color-correction work on **Windows 11 / RTX 5060 Ti
16 GB / driver 616.92** (both on torch 2.13.0+cu130 in the FlashVSR venv). The GPUs
differ, so **wall clock is not comparable across the two**; VRAM peaks are, because
the allocation sizes do not depend on the GPU model.

Bottom line: **at 1080p on a 16 GB card, `--flashvsr-scale 2 --flashvsr-tiles 1` is
the first choice** (10.8 GB peak, 5.5 GB below the ceiling). If you need scale 4, use
offline (`flashvsr`) or inline with `--flashvsr-tiles 2`; untiled inline at scale 4
completes, but with almost no headroom.

- **PyTorch's `expandable_segments` is unsupported on Windows** (it warns and falls
  back to the default caching allocator). `backend:cudaMallocAsync` does not help
  (measured slightly worse). jasna therefore does not set `expandable_segments` for
  the worker on Windows.
- **The fragmentation penalty is scale-dependent and only shows at scale 4.**
  Comparing against Linux with the desktop residency subtracted:

  | Configuration | Linux | Windows | Delta |
  |---|---|---|---|
  | inline scale 2 / tiles 1 (1080p) | 9642 MiB | 8925 MiB | **-717** |
  | offline Phase 2 scale 2 (480p) | 6070 MiB | 6418 MiB | +348 |
  | offline Phase 2 scale 4 (480p) | 11494 MiB | 13040 MiB | **+1546** |

  Scale 4 processes at 1024px, where fragmentation bites and tiny-long's reserved
  VRAM runs **+1–2 GB** above the Linux "flat ~11.9 GB" figure. Scale 2 processes at
  512px and takes no such hit — Windows is actually lower. **The "Windows costs
  +1–2 GB" rule of thumb does not apply to scale 2.**
- **The WDDM desktop holds ~1 GB** (near zero on headless Linux), leaving
  **~15.2 GB effective** on a 16 GB card. A real desktop with a browser and IDE open
  can idle above ~2 GB.
- Full-pipeline inline measurements (full-length clips, whole-GPU nvidia-smi
  peak, real desktop idling at ~2.1 GB):

  | `--flashvsr-tiles` | 1080p peak | Wall clock (vs tiles 1) |
  |---|---|---|
  | `1` | 15918 MiB | 1.00x |
  | `2` | **14222 MiB** | 1.25x |
  | `3` | 12490 MiB | 1.46x |
  | `4` | 11514 MiB | 1.46x |

  Tiles `1` completed at both 480p and 1080p (zero offloads, zero OOM warnings), but
  the peak sits within 400 MiB of the physical ceiling (16303 MiB) — a resident app
  opening a few tabs can tip it into OOM. **For regular 1080p use at scale 4, run
  `--flashvsr-tiles 2`** (~2 GB headroom, +25% wall clock). No strip-boundary seams
  (banding, color shift) were detected in these runs either.
- Scale 2 vs scale 4 measurements (RTX 5060 Ti 16 GB, 1870 MiB desktop residency,
  whole-GPU peak; 480p = 852x480 / 10661 frames, 1080p = 1920x1080 / 6242 frames):

  | Mode | scale / tiles | Resolution | Peak | Headroom | Wall clock |
  |---|---|---|---|---|---|
  | inline | 2 / 1 | 480p | 10237 MiB | 6074 MiB | 824 s |
  | inline | 2 / 1 | 1080p | **10795 MiB** | **5516 MiB** | 575 s |
  | inline | 4 / 2 | 1080p | 15148 MiB | 1163 MiB | 2519 s |
  | offline | 2 | 480p | 8288 MiB | 8023 MiB | 1220 s |
  | offline | 4 | 480p | 14910 MiB | 1401 MiB | 3174 s |

  Every run: zero offloads, zero worker clip retries, output frame count equal to the
  input. Scale 4 / tiles 2 costs 4.38x the wall clock of scale 2 / tiles 1, matching
  the 4.35x measured on Linux. **The scale-4 figures (inline 15148 MiB, offline
  14910 MiB) include 1.9 GB of desktop residency**, so a machine idling at 2.5 GB is
  left under 500 MiB of headroom — running scale 4 on 16 GB assumes you manage what
  else holds VRAM.
- Offline bundle sizes land where the estimate says (480p / 10661 frames): scale 2
  estimated 7.0 GiB and wrote 8.1 GiB; scale 4 estimated 28.1 GiB and wrote 30 GiB
  (29 GiB of 1024px output plus a 1000 MB 256px dump). Both fit "estimate plus the
  dump". Phase 2 raises no `UnicodeEncodeError`.
- Measured peaks (scale 4 / tiny-long / bf16 / sage / 85 frames, reserved):
  - **256px input (jasna's real workload): ~13.0 GB** — Phase 2 has the GPU to
    itself, so offline works on 16 GB Windows.
  - **384px input (the bundled example0 smoke): ~15.1 GB** — razor-thin against the
    effective free VRAM; a browser or IDE holding a few hundred MB tips it into OOM.
    **A smoke-test OOM does not imply jasna's real workload OOMs.** This is why the
    setup's step-4 smoke test runs at scale 2.
- **An 85-frame smoke with `-m tiny` (O(T) memory) is expected to OOM on 16 GB
  Windows.** Smoke-test with `-m tiny-long`, as setup step 4 does.
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

Reused jasna assets: `BlendBuffer` / `crop_buffer.scale_offsets` (the upscaled
crops re-blend unchanged at either scale), `RestorationPipeline.build_secondary_result` (the
`[keep_start:keep_end]` slice), `pipeline_items` (the serialization units), and
`media/backend.make_video_{reader,encoder}` for Phase 3 decode/encode.
