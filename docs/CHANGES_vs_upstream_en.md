# Change summary: `modi` (`v0.7.2+modi`) vs upstream main

A summary of what this branch adds on top of upstream (`main`). `modi` is rebased onto upstream main `278ab09` (`v0.7.2`).

- Branch: `modi` (fork `https://github.com/sh202603/jasna.git`)
- Version: `0.7.2+modi` (based on upstream main `278ab09` = `v0.7.2`)
- Scope: consolidated to 6 commits over upstream main (`git diff --shortstat upstream/main..modi`).

> **Base bumped to `v0.7.2`.** The per-commit reconciliation notes below were written for the original `v0.6.2`-era rebase; those upstream fixes (decoder stream-sync, separable conv, validate-model-name, onnx export, trt load) persist in the current base (upstream force-pushed `main` at the v0.7.0 release, changing SHAs but not the features), so the convergence still holds. The `v0.7.1` base already carried upstream's **supporter models** (SD 1.5 image restoration, unet-4x) and model-encryption/Nuitka plumbing; this fork carries that code **inert** (it cannot be decrypted/run from public source — see the scope note in the README). The fork's build moved with upstream from PyInstaller to Nuitka; packaging tooling is private, so the public path is running from source (see `docs/BUILDING_*`).
>
> **New in the `v0.7.2` base** (absorbed 2026-06-18): upstream's **post-export actions** (`--post-export-action shutdown|command`, `--post-export-command`), a **detection-model registry / model factory** (`jasna/mosaic/detection_registry.py`), **output-pattern** filename templates (`--output-pattern "{original}_restored.mp4"`), CLI/GUI folder-processing improvements (combined image+video `[i/N]` counter), startup-time optimizations (lazy `restorer/__init__.py` imports), and GUI/Windows fixes. None conflict with the inert supporter-model carriage above, and the supporter `jasna/protection` submodule pointer is followed (`827c6cb → 1506724`, still empty in the public fork). modi's own CLI per-file folder banner is now **absorbed by upstream** (`2f3a87e`, with a better combined counter) and dropped on this rebase (see §6).

> Changes that upstream has since absorbed are dropped on this rebase: the **separable box blur** (§3, upstream `f8c4048`), the **torch 2.12 stack**, and — new in `v0.6.2` — the **decoder stream-sync that fixes the issue #2 stutter** (`cf33bcf` "sync decoder cuda stream", §5). The branch's own decoder sync barriers and the opt-in `--decoder nvdec` backend are therefore **removed as redundant**; only the separate primary-restore buffer-hold fix is retained (§5). **BT.601 colorspace** was also implemented independently upstream and is reconciled (§2). Upstream `v0.6.2` also **removed the HAGS check** and **added `onnx` as a dependency**.
>
> The three post-`v0.6.2` upstream commits (absorbed 2026-06-09) are reconciled the same way: `b55b501` "validate model name" lands as-is (file untouched by modi). `5b3ca34` "ultralytics onnx export fix" (save/restore of `CUDA_VISIBLE_DEVICES`) converged with this branch's YOLO CPU-export fix (§1) — upstream's save/restore is adopted while the frozen-build `half=False, device="cpu"` part stays. `6545b78` "linux trt load fix" (preload pip `tensorrt_libs` before nvvfx) converged with this branch's existing fix (§1) — upstream's `_preload_tensorrt_runtime` name is adopted, the body keeps this branch's defensive version (`find_spec` + try/except; no-op on Windows, which is handled via DLL load order). Upstream's new tests for both files pass unchanged against the modi implementations.

The changes split into the following layers.

---

## 1. Foundation (build/runtime changes)

Build-environment and runtime updates over upstream main. All are **applied directly to the branch source**, so you can build straight after cloning (the only exception is the mmengine patch applied to the venv, see below).

- **Build/runtime improvements**
  `model_weights/` auto-resolution (new `jasna/model_weights_resolver.py`). Upstream also added a frozen-binary `engine_paths.model_weights_dir()`; its call sites are routed through this resolver to unify them (keeping the superset with env override, package-parent fallback, and logging). Plus TRT benchmark API update, Windows-only DLL load helpers, PyInstaller spec tweaks, etc.
- **YOLO detection export fix**: export the detection ONNX on CPU so a frozen (PyInstaller) binary does not hit CUDA error 100 — ultralytics' GPU export initializes torch CUDA before TensorRT and breaks the init order; `CUDA_VISIBLE_DEVICES` is restored afterward. The engine is still built fp16. (`onnx` itself is now an upstream dependency.)
- **Linux GUI / RTX-VSR fixes**
  Fix blank modal dialogs; fix RTX Super-Res TensorRT version clash by pre-loading `libnvinfer.so.10`.
- **Build guides**: `docs/BUILDING_{LINUX,WINDOWS}_{ja,en}.md`. On this branch the changes are pre-applied to the source, so you can build straight after cloning; each guide's appendix documents the change breakdown for reference.

> The mmengine patch `patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff` applies to the venv package and is applied separately during the build (each guide's §8.1) — it is the only bundled patch.

---

## 2. New feature (modi): flexible output formats

Extends the GPU encode output — previously fixed to HEVC / 10-bit (P010) / BT.709 — to match lada-ex's flexibility.

> Upstream independently added **BT.601 support** (a narrow `chw_rgb_to_p010_bt601_limited` plus an HEVC VUI rewrite) and a **`.cube` LUT** feature. The rebase keeps this branch's generic superset (`chw_rgb_to_surface`, BT.601/709/2020 × NV12/P010) while **absorbing upstream's HEVC VUI bitstream rewrite (`-bsf:v hevc_metadata`, not applied for AV1) and the `.cube` LUT** so both coexist.

- **AV1 output**: `--codec av1` (file output only; not for streaming; NVENC AV1 has B-frames disabled).
- **8-bit (NV12) / 10-bit (P010) output**: `--bit-depth {auto,8,10}`. Default `auto` follows the source (10-bit → P010, otherwise → NV12).
- **BT.601 / BT.709 / BT.2020 colorspace preservation**: selects the limited-range RGB→YUV matrix from the source colorspace and tags the output container with matrix + primaries + transfer (BT.2020 → `bt2020nc` / `bt2020` / `bt2020-10`).

### Implementation highlights
- `jasna/media/rgb_to_p010.py`: generalized `chw_rgb_to_surface(frame, colorspace, bit_depth)` (NV12/P010 × BT.601/709/2020).
- `jasna/media/__init__.py`: `Colorspace` enum and `VideoMetadata.yuv_colorspace` (carries BT.2020, which av's Colorspace cannot represent); extended ffprobe colorspace detection.
- `jasna/media/video_encoder.py`: derive `fmt`/`profile`/B-frames/temp extension (`.hevc`/`.obu`) from codec + bit depth; remux tags matrix/primaries/transfer correctly. The output container is chosen by the output extension (`.mkv` Matroska, `.mp4`/`.mov` MP4/MOV, AV1-in-MP4 supported): mkvmerge builds an intermediate, then ffmpeg does the final remux (audio mux, `-c:v copy`, `+faststart` for `.mp4`/`.mov`).
- `jasna/pipeline.py` / `jasna/streaming_pipeline.py`: change the early guard from "reject non-BT.709" to "reject full range only" (BT.601/709/2020 all allowed). Full range is detected when ffprobe's `color_range` is `pc` or `jpeg` (ffprobe reports `pc`, so the `pc` token must not be missed -- fixed).
- CLI (`jasna/main.py`) and GUI (`jasna/gui/`) gain `--codec` / `--bit-depth`.
- Docs: `docs/CODECS_AND_COLORSPACE_{ja,en}.md`, plus a one-line note in each README.
- Tests: new `tests/test_rgb_to_surface.py` and added AV1/NV12/BT.601/BT.2020 + validation cases.

### Hardware verification (ffmpeg 8 / mkvmerge v97)
HEVC 8/10-bit (709), BT.601, BT.2020, AV1 (8/10-bit), and `--bit-depth` overrides (both directions) all verified -- output codec/pix_fmt/color tags checked with ffprobe, decode OK. Additionally verified: **10-bit BT.601/709/2020 to P010**, the **`Main 10` profile for 10-bit HEVC**, **AV1-in-MP4 output to `.mp4`**, and **rejection of full-range (`pc`) input**. Unit tests all pass (added `test_color_range_pc`, a regression test for full-range `pc` detection). The matrix above (`--codec` × `--bit-depth` × BT.601/709/2020, both `--bit-depth` overrides, and full-range rejection) is confirmed on **both Windows and Linux** (RTX 5060 Ti) — via the **CLI, GUI continuous processing, and the frozen (PyInstaller) binary** (incl. lada-yolo detection + RTX Super-Res, and BT.2020 with no visible color breakage in VLC).

### Known limitations
- AV1 is file output only (no streaming).
- HDR transfer characteristics (PQ/HLG) are not preserved (matrix + primaries only).
- Full-range (JPEG/PC range) input is not supported (rejected on detection).
- AV1 muxing goes OBU -> mkvmerge (intermediate) -> ffmpeg remux (final); older mkvmerge may need IVF wrapping or a direct ffmpeg mux.

---

## 3. Performance fix: separable box blur for blend-mask generation (now upstream)

> **No longer a branch-specific delta:** upstream `v0.6.2` carries an equivalent implementation (`f8c4048` "Reapply separable conv"), so the duplicate commit is dropped on the rebase. Kept here as history.

Upstream's `create_blend_mask` (`jasna/tracking/blending.py`) ran two dense `conv2d` box blurs (dilation + falloff) with a large uniform kernel scaled to frame height (61×61 @1080p, 121×121 @4K, larger under super-res) — `O(K²)` per region per frame. A uniform kernel is separable (`K×K = (1×K) ⊛ (K×1)`), so replacing `_box_blur` with two 1D passes drops the cost to `O(2K)` with bit-identical output. This branch and upstream implemented essentially the same algorithm independently, so the rebase adopts upstream's version.

---

## 4. New feature (modi): frame-rate up-conversion (frame generation)

Raises the output frame rate by inserting AI-interpolated frames: `--frame-gen {none,2x,4x}` (file output only; not for streaming, like AV1).

- **Why a new stage, not a secondary restorer**: the secondary restorers (`unet-4x` / `tvai` / `rtx-super-res`) process 256×256 mosaic crops and never change the frame count. Frame generation operates on full-resolution output frames and *increases* the frame count + PTS, so it is wired in as a thin decorator (`FrameGenWriter`) around the pipeline's `FrameWriter` — the rest of the pipeline and the encoder are untouched.
- **Backend** (`--frame-gen-backend {rife,rtx}`), pluggable via the `FrameGenerator` protocol:
  - `rife` (default): neural interpolation (RIFE), runs today on CUDA. Loads a TorchScript checkpoint (recommended, self-contained) or a state_dict into the vendored RIFE 4.6 `IFNet` (`jasna/models/rife/`). Weights via `--frame-gen-model-path` or `model_weights/rife.pth`.
  - `rtx`: **NVIDIA RTX Video Frame Generation** — announced as a Python wheel + ComfyUI node alongside RTX Spark, but not yet shipped in `nvidia-vfx` (1.2.0 exposes only `VideoSuperRes`). The adapter (`jasna/framegen/rtx_frame_generator.py`) probes for the future effect and raises a clear error until it is released; activating it is then a one-line inference call.
- **PTS math** (`FrameGenWriter`): for each consecutive pair, emit the real frame then `M-1` interpolated frames at `pts_k = prev_pts + round((curr_pts - prev_pts) * k / M)`. Output timing is PTS-driven (mkvmerge timecodes), so inserting timestamps is what produces the 2x/4x rate; audio keeps the original timecodes, so duration and sync are preserved. Total frames: `(N-1)*M + 1`. Non-monotonic PTS intervals skip interpolation. NVENC `fps`/`gop` are scaled by the multiplier for correct GOP/rate-control (`NvidiaVideoEncoder(output_fps_multiplier=...)`).
- CLI (`jasna/main.py`) and GUI (`jasna/gui/`) gain `--frame-gen` / `--frame-gen-backend`.
- Converter: `make_rife_torchscript.py` (Practical-RIFE -> TorchScript; delegates to `Model.inference` so the per-version `scale_list` is correct; fp16 trace is the default on CUDA with automatic fp32 fallback; verified with RIFE 4.25). Procedure: `docs/FRAME_GENERATION_{en,ja}.md`.
- Tests: `tests/test_frame_gen_writer.py` (PTS/count, GPU-free) and `tests/test_rife_frame_generator.py` (padding + interpolate shapes + TorchScript load + fp16 default/fp32 fallback, CUDA-gated).
- Hardware-verified (RTX, Windows, run from source): a 30 fps / 10661-frame input -> `--frame-gen 2x` gives 60 fps / 21321 frames (= `(N-1)*2+1`) with `ffprobe` confirming unchanged duration and audio sync.

### Notes / limitations
- v1 runs RIFE in PyTorch at full resolution on the blend-encode thread (TensorRT and a dedicated stage are possible future work).
- RIFE defaults to fp16 (following `--fp16`). The bundled IFNet builds the warp grid in the flow's dtype, satisfying `grid_sample`'s dtype-match requirement; external TorchScript checkpoints with a baked-in float32 grid are detected by an init-time probe and fall back to fp32 automatically. Measured end-to-end (RTX 5060 Ti, 1080p, `--frame-gen 2x`, lada-yolo-v4): 16.5 fps fp32 → **31.4 fps fp16 (~1.9x)**; fp16 vs fp32 outputs average ~50 dB PSNR (visually identical).
- RIFE weights are not bundled — supply a checkpoint. Verify the license of any weights (Practical-RIFE training weights carry non-commercial terms).

---

## 5. Bug fix: Linux mosaic-region "chunk movement" artifact (issue #2 / upstream #158)

On Linux, processed videos showed the de-mosaic (mosaic-removal) region stuttering and drifting out of sync with its surroundings ("chunk movement"). It is independent of detector / secondary restoration / RTX Super-Res / MPS / VRAM settings and also existed upstream (reported as #158).

> **Upstream `v0.6.2` fixes the primary cause.** The headline `v0.6.2` change `cf33bcf` ("sync decoder cuda stream") re-introduces a decoder stream-sync — a CUDA `ExternalStream` built on a driver-level blocking stream plus `synchronize()` barriers around the decode→convert→copy handoffs. This is the same decoder surface-reuse race the branch previously fixed with its own `torch.cuda.synchronize()` barriers, so **the branch's decoder fix and the opt-in `--decoder nvdec` backend are dropped as redundant** — `v0.6.2` resolves the stutter floor on the default path.

### Retained on this branch: primary-restore buffer-hold race
A second, independent site is **not** addressed upstream and is kept here:

- `jasna/restorer/restoration_pipeline.py`: a view into the TRT runner's persistent output buffer was held and copied out to uint8 later on a different thread, so the next clip's inference could overwrite it first, leaving the restored region with another clip's data. Fixed with `primary_raw = self.restorer.raw_process(...).clone()` (copies out on the primary stream before the next inference reuses the buffer).

This clone closes the restore-side race that the decoder stream-sync alone does not cover.

---

## 6. Folder-batch processing (modi)

Upstream v0.7.0 added folder input (`--input <dir> --output <dir>`, which processes images then videos). One frame-generation fix remains on top, scoped to folder batches:

> **Per-file progress banner — now upstream.** modi previously printed `[i/N] Processing <in> -> <out>` before each video in a folder batch (`d5c801f`). Upstream `v0.7.2` added the identical banner (`2f3a87e`) and extended it to a combined image+video `[current/total]` counter, so the modi commit is **dropped on this rebase** as redundant. Kept here as history.

- **Frame generation across a batch**: the RIFE generator is built once and shared across every video in the batch, but `FrameGenWriter.close()` used to close that *borrowed* generator after the first video (freeing the model → `_model = None`), so the second video crashed with `'NoneType' object is not callable`. The writer now leaves the generator alone; whoever built it (the CLI folder loop / the GUI job runner) owns its lifecycle and closes it once after the batch — mirroring how the pipeline treats the borrowed restoration pipeline. Verified: a 2-video folder with `--frame-gen 2x` now processes both files, each `(N-1)*2+1` frames.
