# Change summary: `modi` (`v0.8.0+modi`) vs upstream main

A summary of what this branch adds on top of upstream (`main`). `modi` is rebased onto upstream main `abc5d6a` (`v0.8.0`).

- Branch: `modi` (fork `https://github.com/sh202603/jasna.git`)
- Version: `0.8.0+modi` (based on upstream main `abc5d6a` = `v0.8.0`)

> **New with the `v0.8.0` base** (merged 2026-07-18): upstream moved the media layer wholesale to **PyAV (NVDEC/NVENC)** (removing `python_vali` / `PyNvVideoCodec` from the runtime and **dropping the mkvmerge dependency**), and implemented **AV1/H264 encoding**, **BT.601/709/2020 x full/limited x 8/10-bit** colorspace/bit-depth handling, **software-decode fallback**, and **anamorphic (SAR) support**. It also added the **segment editor + smart rendering** (`--segments`), **restoration preview playback**, the **mask suggestion editor** (uploads frame+mask, encrypted, to upstream's Cloudflare Worker — only on an explicit Submit), **VR180 restoration** (`--vr-mode`), **high-fps retargeting** (`--retarget-high-fps`, 60->30 decimation), and many Linux GUI fixes. Driver requirement: Windows 610+ / Linux 580+; dependency `av>=18,<19` (the GPU path needs the current_ctx API slated for PyAV 18.1.0 — until it is published, use a wheel built from PyAV upstream main `61e4aa8`).
>
> Rebase policy: **upstream wins on overlapping features.** This fork's "flexible output" (old §2) was dropped in favor of the upstream implementation (the `--bit-depth` flag is gone; see §2), and the torchcodec backend semantics were redefined against the new media layer (§7).
>
> **Verification status**: post-rebase verification so far covers the GPU-less pytest suite (1523 passed; remaining failures are CUDA-required / missing-protection / headless-environment only) and CLI smoke checks. **GPU runs (native/torchcodec backends, frame-gen, FP8, FlashVSR, TRT engine recompilation) are still pending** for the next GPU session.

> **Base bumped to `v0.7.2`.** The per-commit reconciliation notes below were written for the original `v0.6.2`-era rebase; those upstream fixes (decoder stream-sync, separable conv, validate-model-name, onnx export, trt load) persist in the current base (upstream force-pushed `main` at the v0.7.0 release, changing SHAs but not the features), so the convergence still holds. The `v0.7.1` base already carried upstream's **supporter models** (SD 1.5 image restoration, unet-4x) and model-encryption/Nuitka plumbing; this fork carries that code **inert** (it cannot be decrypted/run from public source; see the scope note in the README). The fork's build moved with upstream from PyInstaller to Nuitka; packaging tooling is private, so the public path is running from source (see `docs/BUILDING_*`).
>
> **New in the `v0.7.2` base** (absorbed 2026-06-18): upstream's **post-export actions** (`--post-export-action shutdown|command`, `--post-export-command`), a **detection-model registry / model factory** (`jasna/mosaic/detection_registry.py`), **output-pattern** filename templates (`--output-pattern "{original}_restored.mp4"`), CLI/GUI folder-processing improvements (combined image+video `[i/N]` counter), startup-time optimizations (lazy `restorer/__init__.py` imports), and GUI/Windows fixes. None conflict with the inert supporter-model carriage above, and the supporter `jasna/protection` submodule pointer is followed (`827c6cb → 1506724`, still empty in the public fork). modi's own CLI per-file folder banner is now **absorbed by upstream** (`2f3a87e`, with a better combined counter) and dropped on this rebase (see §6).

> Changes that upstream has since absorbed are dropped on this rebase: the **separable box blur** (§3, upstream `f8c4048`), the **torch 2.12 stack**, and (new in `v0.6.2`) the **decoder stream-sync that fixes the issue #2 stutter** (`cf33bcf` "sync decoder cuda stream", §5). The branch's own decoder sync barriers and the opt-in `--decoder nvdec` backend are therefore **removed as redundant**; only the separate primary-restore buffer-hold fix is retained (§5). **BT.601 colorspace** was also implemented independently upstream and is reconciled (§2). Upstream `v0.6.2` also **removed the HAGS check** and **added `onnx` as a dependency**.
>
> The three post-`v0.6.2` upstream commits (absorbed 2026-06-09) are reconciled the same way: `b55b501` "validate model name" lands as-is (file untouched by modi). `5b3ca34` "ultralytics onnx export fix" (save/restore of `CUDA_VISIBLE_DEVICES`) converged with this branch's YOLO CPU-export fix (§1): upstream's save/restore is adopted while the frozen-build `half=False, device="cpu"` part stays. `6545b78` "linux trt load fix" (preload pip `tensorrt_libs` before nvvfx) converged with this branch's existing fix (§1): upstream's `_preload_tensorrt_runtime` name is adopted, the body keeps this branch's defensive version (`find_spec` + try/except; no-op on Windows, which is handled via DLL load order). Upstream's new tests for both files pass unchanged against the modi implementations.

The changes split into the following layers.

---

## 1. Foundation (build/runtime changes)

Build-environment and runtime updates over upstream main. All are **applied directly to the branch source**, so you can build straight after cloning (the only exception is the mmengine patch applied to the venv, see below).

- **Build/runtime improvements**
  `model_weights/` auto-resolution (new `jasna/model_weights_resolver.py`). Upstream also added a frozen-binary `engine_paths.model_weights_dir()`; its call sites are routed through this resolver to unify them (keeping the superset with env override, package-parent fallback, and logging). Plus TRT benchmark API update, Windows-only DLL load helpers, PyInstaller spec tweaks, etc.
- **YOLO detection export fix**: export the detection ONNX on CPU so a frozen (PyInstaller) binary does not hit CUDA error 100: ultralytics' GPU export initializes torch CUDA before TensorRT and breaks the init order; `CUDA_VISIBLE_DEVICES` is restored afterward. The engine is still built fp16. (`onnx` itself is now an upstream dependency.)
- **Linux GUI / RTX-VSR fixes**
  Fix RTX Super-Res TensorRT version clash by pre-loading `libnvinfer.so.10`. The old blank-modal-dialog workaround (deferred `grab_set`) was dropped: upstream v0.8.0's `wait_visibility()`-based X11 handling plus its Linux GUI fixes supersede it.
- **Build guides**: `docs/BUILDING_{LINUX,WINDOWS}_{ja,en}.md`. On this branch the changes are pre-applied to the source, so you can build straight after cloning; each guide's appendix documents the change breakdown for reference.

> The mmengine patch `patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff` applies to the venv package and is applied separately during the build (each guide's §8.1); it is the only bundled patch.

---

## 2. Flexible output formats (fully absorbed by upstream v0.8.0)

> **No longer a delta of this branch.** In the v0.7.2 era, modi implemented AV1 / 8-10-bit / BT.601-2020 output on its own (`--codec` / `--bit-depth`, the generic `chw_rgb_to_surface` converter, mkvmerge+ffmpeg remux). Upstream v0.8.0's PyAV-based encoder now implements the **same or more**: `--codec {hevc,h264,av1}`, the full BT.601/709/2020 x full/limited x NV12/P010 converter set (`rgb_to_nv12.py` / `rgb_to_p010.py`), inline H.273 color tagging through PyAV, full-range input support, and no mkvmerge. Per the rebase policy (upstream wins), the modi implementation was dropped. Consequences:

- **The `--bit-depth` flag is gone.** Bit depth follows the codec per upstream's spec (hevc/av1 -> 10-bit P010, h264 -> 8-bit NV12; only `--segments` smart-render fragments follow an 8-bit source via `match_input_bit_depth`).
- modi's `Colorspace` enum / `VideoMetadata.yuv_colorspace` / `chw_rgb_to_surface` / mkvmerge-based remux were removed (upstream represents BT.2020 natively with `av>=18`).
- The old `docs/CODECS_AND_COLORSPACE_{ja,en}.md` and `tests/test_rgb_to_surface.py` were deleted (superseded by upstream's `test_rgb_to_nv12.py` / `test_rgb_to_p010.py` / `test_video_encoder_mux.py`).
- The only modi delta remaining in this area is the **frame-gen output-fps wiring** (see §4).

---

## 3. Performance fix: separable box blur for blend-mask generation (now upstream)

> **No longer a branch-specific delta:** upstream `v0.6.2` carries an equivalent implementation (`f8c4048` "Reapply separable conv"), so the duplicate commit is dropped on the rebase. Kept here as history.

Upstream's `create_blend_mask` (`jasna/tracking/blending.py`) ran two dense `conv2d` box blurs (dilation + falloff) with a large uniform kernel scaled to frame height (61×61 @1080p, 121×121 @4K, larger under super-res), which is `O(K²)` per region per frame. A uniform kernel is separable (`K×K = (1×K) ⊛ (K×1)`), so replacing `_box_blur` with two 1D passes drops the cost to `O(2K)` with bit-identical output. This branch and upstream implemented essentially the same algorithm independently, so the rebase adopts upstream's version.

---

## 4. New feature (modi): frame-rate up-conversion (frame generation)

Raises the output frame rate by inserting AI-interpolated frames: `--frame-gen {none,2x,4x}` (file output only; not for streaming, and rejected together with `--segments` smart rendering, whose copied spans would no longer match the output frame rate).

- **Why a new stage, not a secondary restorer**: the secondary restorers (`unet-4x` / `tvai` / `rtx-super-res`) process 256×256 mosaic crops and never change the frame count. Frame generation operates on full-resolution output frames and *increases* the frame count + PTS, so it is wired in as a thin decorator (`FrameGenWriter`) around the pipeline's `FrameWriter`; the rest of the pipeline and the encoder are untouched.
- **Backend** (`--frame-gen-backend {rife,rtx}`), pluggable via the `FrameGenerator` protocol:
  - `rife` (default): neural interpolation (RIFE), runs today on CUDA. Loads a TorchScript checkpoint (recommended, self-contained) or a state_dict into the vendored RIFE 4.6 `IFNet` (`jasna/models/rife/`). Weights via `--frame-gen-model-path` or `model_weights/rife.pth`.
  - `rtx`: **NVIDIA RTX Video Frame Generation**, announced as a Python wheel + ComfyUI node alongside RTX Spark, but not yet shipped in `nvidia-vfx` (1.2.0 exposes only `VideoSuperRes`). The adapter (`jasna/framegen/rtx_frame_generator.py`) probes for the future effect and raises a clear error until it is released; activating it is then a one-line inference call.
- **PTS math** (`FrameGenWriter`): for each consecutive pair, emit the real frame then `M-1` interpolated frames at `pts_k = prev_pts + round((curr_pts - prev_pts) * k / M)`. Output timing is PTS-driven, so inserting timestamps is what produces the 2x/4x rate; audio keeps the original timecodes, so duration and sync are preserved. Total frames: `(N-1)*M + 1`. Non-monotonic PTS intervals skip interpolation. Encoder fps/GOP are corrected by passing **source fps x multiplier** into the upstream v0.8.0 encoder's `output_fps` parameter (the old `output_fps_multiplier` parameter is gone; the multiplier also applies on top of the `--retarget-high-fps` decimated rate).
- CLI (`jasna/main.py`) and GUI (`jasna/gui/`) gain `--frame-gen` / `--frame-gen-backend`. The GUI encoding settings also expose the RIFE weights path (the `--frame-gen-model-path` equivalent).
- Standalone `jasna-framegen` CLI (`jasna/framegen_cli.py`, new `console_script`): applies **only** frame generation (2x/4x) to an already-restored video, with no detection and no restoration. A thin driver reusing the same NVDEC/NVENC path and `FrameGenWriter`; imports nothing from `jasna.pipeline` or `jasna.protection` (guarded by `tests/test_framegen_cli_protection_free.py`). Enables a two-pass workflow (restore with the official binary → up-convert here). Supports folder input/output with `--output-pattern` ({original} template), reusing `media_files.classify_folder` / `folder_output_path`; videos only (images skipped), one shared generator across the batch. Tests: `tests/test_framegen_cli_{driver,device,folder,folder_device,protection_free}.py`.
- Converter: `make_rife_torchscript.py` (Practical-RIFE -> TorchScript; delegates to `Model.inference` so the per-version `scale_list` is correct; fp16 trace is the default on CUDA with automatic fp32 fallback; verified with RIFE 4.25). Procedure: `docs/FRAME_GENERATION_{en,ja}.md`.
- Tests: `tests/test_frame_gen_writer.py` (PTS/count, GPU-free) and `tests/test_rife_frame_generator.py` (padding + interpolate shapes + TorchScript load + fp16 default/fp32 fallback, CUDA-gated).
- Hardware-verified (RTX, Windows, run from source): a 30 fps / 10661-frame input -> `--frame-gen 2x` gives 60 fps / 21321 frames (= `(N-1)*2+1`) with `ffprobe` confirming unchanged duration and audio sync.

### Notes / limitations
- v1 runs RIFE in PyTorch at full resolution on the blend-encode thread (TensorRT and a dedicated stage are possible future work).
- RIFE defaults to fp16 (following `--fp16`). The bundled IFNet builds the warp grid in the flow's dtype, satisfying `grid_sample`'s dtype-match requirement; external TorchScript checkpoints with a baked-in float32 grid are detected by an init-time probe and fall back to fp32 automatically. Measured end-to-end (RTX 5060 Ti, 1080p, `--frame-gen 2x`, lada-yolo-v4): 16.5 fps fp32 → **31.4 fps fp16 (~1.9x)**; fp16 vs fp32 outputs average ~50 dB PSNR (visually identical).
- RIFE weights are not bundled; supply a checkpoint. Verify the license of any weights (Practical-RIFE training weights carry non-commercial terms).

---

## 5. Bug fix: Linux mosaic-region "chunk movement" artifact (issue #2 / upstream #158)

On Linux, processed videos showed the de-mosaic (mosaic-removal) region stuttering and drifting out of sync with its surroundings ("chunk movement"). It is independent of detector / secondary restoration / RTX Super-Res / MPS / VRAM settings and also existed upstream (reported as #158).

> **Upstream `v0.6.2` fixes the primary cause.** The headline `v0.6.2` change `cf33bcf` ("sync decoder cuda stream") re-introduces a decoder stream-sync, a CUDA `ExternalStream` built on a driver-level blocking stream plus `synchronize()` barriers around the decode→convert→copy handoffs. This is the same decoder surface-reuse race the branch previously fixed with its own `torch.cuda.synchronize()` barriers, so **the branch's decoder fix and the opt-in `--decoder nvdec` backend are dropped as redundant**; `v0.6.2` resolves the stutter floor on the default path.

### Retained on this branch: primary-restore buffer-hold race
A second, independent site is **not** addressed upstream and is kept here:

- `jasna/restorer/restoration_pipeline.py`: a view into the TRT runner's persistent output buffer was held and copied out to uint8 later on a different thread, so the next clip's inference could overwrite it first, leaving the restored region with another clip's data. Fixed with `primary_raw = self.restorer.raw_process(...).clone()` (copies out on the primary stream before the next inference reuses the buffer).

This clone closes the restore-side race that the decoder stream-sync alone does not cover.

---

## 6. Folder-batch processing (modi)

Upstream v0.7.0 added folder input (`--input <dir> --output <dir>`, which processes images then videos). One frame-generation fix remains on top, scoped to folder batches:

> **Per-file progress banner: now upstream.** modi previously printed `[i/N] Processing <in> -> <out>` before each video in a folder batch (`d5c801f`). Upstream `v0.7.2` added the identical banner (`2f3a87e`) and extended it to a combined image+video `[current/total]` counter, so the modi commit is **dropped on this rebase** as redundant. Kept here as history.

- **Frame generation across a batch**: the RIFE generator is built once and shared across every video in the batch, but `FrameGenWriter.close()` used to close that *borrowed* generator after the first video (freeing the model → `_model = None`), so the second video crashed with `'NoneType' object is not callable`. The writer now leaves the generator alone; whoever built it (the CLI folder loop / the GUI job runner) owns its lifecycle and closes it once after the batch, mirroring how the pipeline treats the borrowed restoration pipeline. Verified: a 2-video folder with `--frame-gen 2x` now processes both files, each `(N-1)*2+1` frames.

## 7. New feature (modi): torchcodec backend (experimental, with native = PyAV fallback)

An experimental `torchcodec>=0.15.0` backend usable instead of native (PyAV NVDEC/NVENC since v0.8.0) — optional dependency, off by default. The default stays `native` (current behavior); select via `--video-backend {native,auto,torchcodec}` plus per-side `--decode-backend`/`--encode-backend {inherit,...}` overrides.

**Semantics changed with the v0.8.0 rebase**: the upstream native encoder now always outputs 10-bit (P010) for HEVC/AV1, so output parity with the 8-bit-nv12-only torchcodec encoder no longer holds. Therefore:

- **decode**: as before, `auto` prefers torchcodec and falls back to native on failure. The `--retarget-high-fps` frame stride is native-reader-only (forcing torchcodec with it errors; auto goes native).
- **encode**: **never selected by `auto`** (to avoid silently changing the output bit depth). It runs only when forced via `--encode-backend torchcodec`, and only for **8-bit sources + HEVC/AV1 + mappable NVENC settings** (output is 8-bit; streaming / `--segments` / frame-gen / fps changes are rejected). Colorspace (BT.601/709/2020) is tagged by the encoder's built-in ffmpeg copy-remux using the H.273 code points (plus the `hevc_metadata` BSF VUI rewrite for HEVC); the old mkvmerge-era helper dependency is gone.
- The **observable backend** from b2471d5 (startup log shows `[decode: ..., encode: ...]`; readers/encoders carry a `backend` attribute) and the **dedicated encode worker thread** are kept.

The NVENC settings mapping (cq/qmin/qmax/gop/lookahead/temporalaq/aq/nonrefp/maxbitrate/vbvbufsize and preset -> `extra_options`) is unchanged. Implemented in `jasna/media/backend.py` (selection layer), `torchcodec_decoder.py`, and `torchcodec_encoder.py`. The GUI exposes this as a "Video Backend" dropdown in the encoding settings (applied to both decode and encode). See `docs/TORCHCODEC_BACKEND_{ja,en}.md` for the design and capability matrix (this section is authoritative for the v0.8.0 semantics changes).

## 8. New feature (modi): cuDNN FP8 restoration backend (experimental, with TensorRT fallback)

An experimental cuDNN graph-API FP8 backend for the BasicVSR++ **upsample** sub-engine, ported from lada-ex `feat/fp8-recon` (`lada/models/basicvsrpp/fp8_recon.py`, AGPL-3.0; the stage lada calls "reconstruction" is the same sub-network). Opt-in via `--fp8-recon` (bridged to `JASNA_FP8_RECON=1` so GUI/subprocess paths can also enable it; the GUI exposes it as an "FP8 Restoration (experimental)" toggle in the advanced settings); default off = unchanged behavior. Requires an FP8-capable GPU (sm89+), fp16 mode, and the new `nvidia-cudnn-frontend` dependency (cuDNN runtime >= 9.17 comes with the torch cu130 wheel; win32 additionally gets `triton-windows` for the compiled glue, with a permanent eager fallback if inductor fails); any construction failure logs a warning and falls back to the TensorRT engine.

When active, the TensorRT upsample engine is **not loaded at all** (`load_sub_engines(..., load_upsample=False)`), so its load-time arena (measured 2210 MB with the default b90 profile) is never allocated; the engine file is still built and kept as the fallback. Measured on RTX 5080 (sm120): stage latency 1.45-1.56x vs the TRT FP16 engine (T=60: 8.54 -> 5.46 ms), FP8 resident footprint ~220 MB (net -1991 MB), stage output PSNR 64 dB vs the FP32 reference; pipeline-level, VRAM peak drops 0.9-1.7 GB across 480p-4K clips while e2e fps stays flat (the pipeline is detection-bound in every measured config), output is visually indistinguishable from the FP16 engine (SSIM 0.983-0.993) and bit-deterministic across runs (md5-equal, same as FP16). Two integration issues were found and fixed: torch's bundled cuDNN lib dir is now put in front of `LD_LIBRARY_PATH` before `import cudnn` (nvvfx prepends its own dir holding a bare 9.7 dispatcher without sub-libraries, which cudnn-frontend would otherwise pick and abort on `cudnnCreate` whenever rtx-super-res was constructed first), and warmup eagerly builds **all** T-buckets (jasna clip lengths vary freely over [1, max_clip_size], unlike lada's fixed chunk lengths, so lazy builds would land inside the restore stage). Implemented in `jasna/restorer/fp8_upsample.py` plus the injection/fallback/close changes in `basicvsrpp_sub_engines.py`; A/B gate benchmark via `jasna --benchmark --benchmark-filter fp8` (the pipeline A/B and output comparisons used local harness scripts kept outside the repo). Verified on Windows as well (Windows 11, same RTX 5080): the A/B benchmark passes every gate with matching numbers and a full 1080p pipeline run completes, with the glue compiled through `triton-windows`. Windows frozen-build packaging of `cudnn` remains an open item (falls back to TRT if it fails). Design + full measurements: `docs/FP8_RECON_{ja,en}.md`.

---

## 9. New feature (modi): FlashVSR secondary restoration (experimental, offline 3-phase + inline)

A secondary restorer that upscales each 256px primary crop to 1024px (4x) with the diffusion one-step VSR model [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) (fork [`lihaoyun6/FlashVSR_plus`](https://github.com/lihaoyun6/FlashVSR_plus), 4x fixed) and re-blends it, recovering texture the primary BasicVSR++ leaves blurry on large mosaics, close-ups, and 4K. FlashVSR is a third-party model under its own license; you supply the checkout, weights, and a dedicated venv via `--flashvsr-repo` (unrelated to jasna's supporter models, not bundled). The FlashVSR venv must be a **uv-managed standalone Python** (system Python can't JIT FlashVSR's Triton Sparse_SageAttention kernel). There are **two modes**.

- **`--secondary-restoration flashvsr` (offline 3-phase)**: runs `dump → FlashVSR → reblend` as three sequential subprocesses, each releasing all its VRAM at process exit. FlashVSR's tiny mode peaks at 12-16 GB on its own and cannot co-reside with the ~9 GB primary on a 16 GB card, so the work is split so peak VRAM never overlaps in time. It keeps an intermediate bundle (256px + uncompressed 1024px, hundreds of GB on long mosaic-heavy videos) on disk and supports staged resume via `--flashvsr-bundle-dir`. **Works on 12 GB-class GPUs.** Encodes twice (Phase 1 throwaway + Phase 3 final). Implemented in `jasna/restorer/flashvsr_offline.py` + `flashvsr_phase2_driver.py`, subprocess dispatch in `jasna/__main__.py` (`--flashvsr-phase`).

- **`--secondary-restoration flashvsr-inline` (inline, single pass)**: runs FlashVSR as a secondary restorer inside the normal streaming pipeline, with **no intermediate files, disk gate, or double encode**. It uses FlashVSR's **tiny-long** pipeline (O(1), constant ~11.9 GB regardless of clip length), co-residing with the primary (shrunk to ~1.6 GB by fp8-recon, §8) to run as a single pass. A synchronous `SecondaryRestorer` spawns a resident FlashVSR-venv worker and exchanges 256px→1024px crops over a length-prefixed RGB protocol (lossless tensor capture by replacing `imageio.get_writer`; next_8n5 input padding so small clips still return exactly T frames). It forces the same constraints as offline (clip 32, frame-gen off) and **auto-enables fp8-recon** for the co-residence budget. As a VRAM-ceiling lever it offers **`--flashvsr-tiles {1..4}`** (strip tiled-dit, inline only: the DiT runs per horizontal strip and the strips are feather-blended; `2` cuts the attention mask to 0.39x for ~1.25x wall clock). Requires a **16 GB card and a patched checkout**. Implemented in `jasna/restorer/flashvsr_inline_secondary_restorer.py` + `flashvsr_inline_worker.py`.

**tiny-long patch (bundled)**: the tiny-long pipeline inline relies on has a known bug that crashes on the second chunk (`8192 vs 4096` error); the fix disables two per-chunk cache clears (per-chunk `LQ_proj_in.clear_cache()` / `TCDecoder.clean_mem()`; the once-per-video reset before the loop stays). The `lihaoyun6/FlashVSR_plus` fork is **inactive**, so rather than upstream it the patch ships at [`patches/flashvsr_plus_tinylong_multichunk_fix.patch`](../patches/flashvsr_plus_tinylong_multichunk_fix.patch) (you `git apply` it to the checkout). The restorer checks the checkout is patched at startup and stops with an explicit error (pointing to the offline mode, which uses tiny and needs no patch) otherwise. This makes two bundled patches total: the mmengine one (§1) and this one.

**Constraints**: both modes are file-output only (no `--stream` / image input) and incompatible with `--frame-gen` (run frame generation as a separate pass). Offline additionally rejects the v0.8.0 features `--retarget-high-fps` (the Phase 1 frame stride would misalign the Phase 3 reblend indices), `--segments`, and VR processing (`--vr-mode sbs`/`sbs-fisheye`, or `auto` when it detects VR content) at startup. Inline is rate-limited by FlashVSR (~15 crop-fps), so mosaic-heavy stretches run at that speed.

**Verification**: unit tests `tests/test_flashvsr_{offline,inline}.py` + `test_main.py` (no-GPU CI, zero full-suite regressions). On hardware (RTX 5080 sm120): both modes complete E2E with output frame count = input. On Linux, inline measured a combined VRAM peak of 14780 MiB at 852x480, with an A/B (vs primary-only) confirming only the mosaic regions change; 1080p rises to the physical ceiling (~15.8 GB) where jasna's `vram_offloader` (spills queued frames to system RAM) and the worker's `expandable_segments` absorb the pressure and it still completes. On Windows (same card, no `expandable_segments`), full-length 480p/1080p E2E verified: 1080p peaks at 15.9 GB whole-GPU with tiles `1` (<0.4 GB below the ceiling) vs 14.2 GB at +25% wall clock with `--flashvsr-tiles 2`, with no strip-boundary seams (banding, color shift) detected — tiles `2` is the recommended 1080p setting on Windows 16 GB. Design + full measurements: `docs/FLASHVSR_{ja,en}.md`.
