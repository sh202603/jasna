# FlashVSR secondary restoration (`+modi`)

`--secondary-restoration flashvsr` / `flashvsr-inline` upscale each restored
256px mosaic crop with [FlashVSR](https://github.com/OpenImagingLab/FlashVSR)
(one-step streaming diffusion VSR; jasna uses the
[`lihaoyun6/FlashVSR_plus`](https://github.com/lihaoyun6/FlashVSR_plus) fork) to
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
| FlashVSR pipeline | tiny-long (VRAM independent of clip length; **requires the tiny-long patch**) | tiny (no patch needed) |
| With `--restoration-model-name seedvr2` | rejected at startup (two resident workers exceed 16 GB) | allowed |

Most of this document describes the offline mode; inline is covered in
"Inline mode" at the end.

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

# 2. Create the venv from a Python that ships the dev headers (Python.h). This is
#    mandatory: FlashVSR's Triton Sparse_SageAttention kernel is JIT-compiled at
#    runtime against them, and a header-less system or conda Python dies with
#    "fatal error: Python.h" (or, worse, tiny-long silently returns 0 frames).
#    Either a uv-managed standalone Python or a system Python with its -dev
#    package (e.g. python3.13-dev) works. CAUTION: if uv itself runs inside a
#    snap-confined app (e.g. snap VSCode), its managed Pythons land under a snap
#    revision path and the venv dies on the next snap refresh; prefer an explicit
#    stable interpreter path then:
uv venv --python 3.13 --python-preference only-managed     # or: uv venv --python /usr/bin/python3.13

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
| `--flashvsr-scale` | `4` | Processing scale for both modes: `4` = model-native 1024px, `2` = 512px (faster, lower VRAM). See [Processing scale](#processing-scale---flashvsr-scale). |
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

There is no CLI flag. For A/B verification only, the environment variable
`JASNA_FLASHVSR_COLOR_FIX=adain|wavelet|none` overrides the method in both modes.

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
[`patches/flashvsr_plus_tinylong_multichunk_fix.patch`](../../patches/flashvsr_plus_tinylong_multichunk_fix.patch).
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
  an unpatched checkout. **`--flashvsr-scale 2` changes the picture**: untiled,
  it peaks at 10.1 GB at 480p and 11.3 GB at 1080p on Linux (clip 90; zero
  offloads, zero allocator warnings), so it needs neither tiling nor the ceiling
  tricks; see
  [Processing scale](#processing-scale---flashvsr-scale) for the full table.
- **On Windows, `expandable_segments` is unavailable and the worker's reserved VRAM
  balloons to ~13 GB**, so untiled inline runs pinned to the physical ceiling (it
  completes, but with almost no headroom). At 1080p, use **`--flashvsr-tiles 2`**.
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
  exactly T frames; the strip split, feather blend and the color correction also
  live here). The file is kept **byte-identical with lada-ex's
  `flashvsr_worker.py`** (same policy as the SeedVR2 worker), which is why the wire
  is lada-native BGR; a FlashVSR_plus breakage is fixed once and diff-copied.
- CLI wiring: `jasna/main.py`. Tests: `tests/test_flashvsr_inline.py`.

## Windows notes

Verified on Windows 11 / RTX 5080 16 GB / torch 2.13.0+cu130 (FlashVSR venv). Bottom
line: **on a 16 GB card, use offline (`flashvsr`), or inline with `--flashvsr-tiles`
(`2` recommended at 1080p). Untiled inline completes, but with almost no headroom.**

- **PyTorch's `expandable_segments` is unsupported on Windows** (it warns and falls
  back to the default caching allocator). tiny-long's reserved VRAM runs **+1–2 GB**
  above the Linux "flat ~11.9 GB" figure due to fragmentation.
  `backend:cudaMallocAsync` does not help (measured slightly worse). jasna therefore
  does not set `expandable_segments` for the worker on Windows.
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
  opening a few tabs can tip it into OOM. **For regular 1080p use, run
  `--flashvsr-tiles 2`** (~2 GB headroom, +25% wall clock). No strip-boundary seams
  (banding, color shift) were detected in these runs either.
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

Reused jasna assets: `BlendBuffer` / `crop_buffer.scale_offsets` (the upscaled
crops re-blend unchanged at either scale), `RestorationPipeline.build_secondary_result` (the
`[keep_start:keep_end]` slice), `pipeline_items` (the serialization units), and
`media/backend.make_video_{reader,encoder}` for Phase 3 decode/encode.
