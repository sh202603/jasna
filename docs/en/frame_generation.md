# Frame generation (frame-rate up-conversion)

`--frame-gen {none,2x,4x}` raises the output frame rate by 2x/4x (file output only; not for `--stream`).
It inserts AI-interpolated frames between the source frames with new PTS placed between the originals.
Audio keeps the original timecodes, so duration and sync are preserved.

Backend: `--frame-gen-backend {rife,rtx}`
- `rife` (default): RIFE neural interpolation in PyTorch. Runs on every supported GPU; supply weights separately (below).
- `rtx`: NVIDIA RTX Video Frame Generation through `nvidia-vfx` 0.2.0.0+. No weights to supply, about 10x faster
  than RIFE and slightly higher interpolation accuracy, but **RTX 40 series (Ada) or newer only**. See
  [RTX Video Frame Generation backend](#rtx-video-frame-generation-backend-rtx).

The RIFE backend auto-detects two checkpoint formats. **TorchScript is recommended** (it bundles the
architecture + weights and is guaranteed to run). You can also drop a raw `flownet.pkl` state_dict in,
but that depends on key-compatibility with the vendored IFNet and is not guaranteed.

---

## Creating a TorchScript checkpoint (recommended)

### 1. Get Practical-RIFE and its weights (one time)

```powershell
git clone https://github.com/hzwer/Practical-RIFE
```

**The weights and model code are NOT in the git clone.** From the README's model list, manually download a
**RIFE 4.x model package** (Google Drive / Baidu), and put the unzipped `*.py` (the IFNet implementation) and
`flownet.pkl` into `<repo>\train_log\` (README: "Download a model ... and put *.py and flownet.pkl on
train_log/"). You should end up with `train_log\` containing `RIFE_HDv3.py`, `IFNet_HDv3.py`, `flownet.pkl`.

> Version: upstream currently recommends **v4.25** (**verified working with v4.25**). The converter delegates
> to each version's `Model.inference` (so the per-version `scale_list` is correct) and auto-detects the
> timestep convention (scalar vs. full-resolution map), so other 4.x builds usually work too. The next step's
> `--validate` confirms actual compatibility.

### 2. Run the converter (use jasna's venv so torch matches the runtime)

The script lives at `scripts/make_rife_torchscript.py` in the repo.

```powershell
.\.venv\Scripts\python.exe scripts/make_rife_torchscript.py `
    --rife-repo C:\path\to\Practical-RIFE `
    --output model_weights\rife.pth `
    --validate
```

- **fp16 is the default** (on CUDA; `--no-fp16` traces fp32, CPU always traces fp32). This matches the
  backend's fp16 default, and an fp16-traced module is the more portable artifact: dtype promotion lets
  it run under an fp32 pipeline too. An fp32-traced module instead bakes a float32 warp grid into the
  graph, so under an fp16 pipeline it triggers the backend's automatic fp32 fallback (it works, but
  without the fp16 benefit). If the fp16 trace fails or yields non-finite output, the script falls back
  to fp32 on its own.
- `--validate` reloads the saved module at a **different resolution** and checks shape/range/midpoint
  blend (confirms it generalizes).
- `--size` (default 256) is the trace resolution; RIFE uses scale-relative interpolation and
  runtime-built warp grids, so it generalizes to other sizes.
- Default output `model_weights\rife.pth` is exactly where the backend looks. To put it elsewhere,
  pass `--frame-gen-model-path <path>` to jasna.

### 3. Run

```powershell
.\.venv\Scripts\python.exe -m jasna --input in.mp4 --output out2x.mkv --frame-gen 2x
.\.venv\Scripts\python.exe -m jasna --input in.mp4 --output out4x.mkv --frame-gen 4x
```

### 4. Verify

```powershell
ffprobe out2x.mkv
```

- `nb_frames` / `avg_frame_rate` ~2x (or ~4x for 4x)
- `Duration` unchanged
- audio in sync

---

## RTX Video Frame Generation backend (`rtx`)

`--frame-gen-backend rtx` runs NVIDIA's Video Frame Generation effect from the RTX Video Effects SDK
(`nvidia-vfx` 0.2.0.0 = VFX SDK 1.3.0; the pinned version in `pyproject.toml`). The model is bundled in the
SDK, so there is nothing to download or convert.

**Requirements**

- GPU: **Ada (RTX 40 series) or newer** (compute capability 8.9+). Also Hopper on Linux. Turing (RTX 20) and
  Ampere (RTX 30) are not supported by the SDK; jasna checks the compute capability up front and tells you to
  use `rife` instead.
- Driver: Linux **570.190+, 580.82+ or 590.44+**; Windows **570.65+**. Note that jasna's own Linux minimum
  (580) accepts 580.0 to 580.81, where the SDK still refuses to load the effect.
- `nvidia-vfx>=0.2.0.0` in the venv. Older venvs built before this pin have 0.1.0.1, which has no frame
  generation effect: `uv pip install nvidia-vfx==0.2.0.0` (the stub on PyPI fetches the real wheel from
  pypi.nvidia.com).

**Usage**

```bash
jasna --input in.mp4 --output out2x.mkv --frame-gen 2x --frame-gen-backend rtx
jasna --input in.mp4 --output out4x.mkv --frame-gen 4x --frame-gen-backend rtx --frame-gen-rtx-mode high
jasna-framegen --input restored.mkv --output out2x.mkv --factor 2x --backend rtx
```

`--frame-gen-rtx-mode {low,medium,high}` (`--rtx-mode` on `jasna-framegen`) selects the SDK quality mode.
`medium` (default) is the sweet spot; `high` is about 6x slower for at most 0.2 dB. `--frame-gen-model-path`
and `--fp16` are RIFE-only and ignored here. The SDK's automatic shot-change detection stays on, so a hard cut
is not blended across.

**Measured (RTX 5080, nvidia-vfx 0.2.0.0, 2026-10-03)**

Per generated frame, synthetic motion pair:

| Resolution | `low` | `medium` | `high` | RIFE 4.25 fp16 |
| --- | ---: | ---: | ---: | ---: |
| 1920x1080 | 0.61 ms | 1.53 ms | 8.99 ms | 10.1 ms |
| 3840x2160 | 1.56 ms | 2.51 ms | 10.6 ms | 48.3 ms |

Interpolation accuracy on `assets/test_clip1` (predict frame i+1 from i and i+2, PSNR against the real frame; a
harder motion step than the 2x case):

| Method | 1080p | 2160p |
| --- | ---: | ---: |
| RIFE 4.25 fp16 | 42.85 dB | 41.41 dB |
| RTX `low` | 42.50 dB | 41.93 dB |
| RTX `medium` | 43.57 dB | 43.01 dB |
| RTX `high` | 43.74 dB | 42.94 dB |

Extra VRAM after the effect is loaded: 1080p `medium` +0.55 GB (`high` +1.24 GB, RIFE fp16 +0.75 GB); 2160p
`medium` +2.06 GB (RIFE +2.85 GB). End-to-end, `jasna-framegen --factor 2x` on the 300-frame 1080p test clip
takes 2.3 s wall clock with `rtx` against 5.8 s with `rife` (decode, encode and startup included).

The effect accepts any frame size (no multiple-of-64 padding as with RIFE) and is re-loaded in about 15 ms
when the size changes, so a folder batch of mixed resolutions shares one generator.

**Not yet verified**: visual A/B against RIFE (ghosting, edge breakup, false shot-change triggers), the
Windows wheel of 0.2.0.0, and the exact SDK error on Turing/Ampere (jasna's own capability check fires first).

---

## Two-pass workflow: `jasna-framegen` (standalone)

`jasna-framegen` is a separate command that applies **only** frame generation to an
already-restored video, with no mosaic detection and no BasicVSR++ restoration. Use it when:

- You restored a video in a **first pass** (the official jasna binary, or `jasna`
  without `--frame-gen`) and want to add 2x/4x afterwards without re-running the
  expensive restore pass.
- You want to tweak the frame-gen factor / codec and re-encode quickly.
- You want frame generation decoupled from the main pipeline for batch processing.

It reuses the same NVDEC/NVENC + mkvmerge path as the integrated `--frame-gen`, so
audio and color metadata are carried over and timing stays PTS-driven exactly as
above. It needs the same `model_weights/rife.pth` (steps 1-2) and never touches the
protection / supporter code.

```bash
# Pass 1: restore only (no frame-gen). Use a near-lossless intermediate so the
# second encode does not stack a generation loss, e.g. a high-quality cq:
jasna --input in.mp4 --output restored.mkv --encoder-settings cq=16
# (or produce restored.mkv with the official binary)

# Pass 2: frame generation only
jasna-framegen --input restored.mkv --output out2x.mkv --factor 2x
jasna-framegen --input restored.mkv --output out4x.mkv --factor 4x
```

Common options: `--factor {2x,4x}`, `--backend {rife,rtx}`, `--model-path <rife.pth>` (RIFE),
`--rtx-mode {low,medium,high}` (RTX), `--codec {hevc,av1}`, `--encoder-settings <k=v,...>`,
`--device cuda:0`, `--no-fp16` (RIFE). Output quality defaults to jasna's encoder profile
(cq=25); override with `--encoder-settings`. Run `jasna-framegen --help` for the full
list. Verify the same way as step 4 (`ffprobe` → ~2x/4x frame rate, unchanged
duration, audio in sync).

### Folder batch + naming pattern

When `--input` is a folder, `--output` is treated as an output folder and every
video in it is processed with one shared RIFE model (built once, reused). Frame
generation is video-only, so any images in the folder are skipped. `--output-pattern`
controls the output filenames (same semantics as `jasna`): `{original}` is the input
stem; the default is `{original}_out` keeping each input's extension.

```bash
# 2x every video in in_dir/ into out_dir/ (default names: <name>_out.<ext>)
jasna-framegen --input in_dir --output out_dir --factor 2x

# Custom names, e.g. clip.mkv -> clip_2x.mkv
jasna-framegen --input in_dir --output out_dir --factor 2x --output-pattern "{original}_2x.mkv"
```

A folder run prints `[i/N] name -> out` per file and continues past a file with an
unsupported color range; an `--output-pattern` that maps two inputs to the same
output (or onto an input) is rejected up front.

---

## Troubleshooting

- **`RTX Video Frame Generation is not available in the installed nvidia-vfx 0.1.0.1 ...`**: the venv predates
  the 0.2.0.0 pin. `uv pip install nvidia-vfx==0.2.0.0`, or use `rife`.
- **`RTX Video Frame Generation requires an NVIDIA Ada (RTX 40 series) or newer GPU ...`**: the SDK effect
  does not run on Turing/Ampere. Use `--frame-gen-backend rife`.
- **`rtx` fails inside the SDK on a driver between 580.0 and 580.81 (Linux)**: the effect needs 580.82+
  (or 570.190+ / 590.44+). Update the driver.
- **`RIFE weights not found: ...`**: no `model_weights\rife.pth`; create one (steps 1-2) or pass `--frame-gen-model-path`.
- **`RIFE state_dict loaded non-strictly (...)`**: you dropped a raw `flownet.pkl` whose keys do not match
  the vendored IFNet; results will be wrong - switch to the TorchScript method.
- **Converter import error**: check `--rife-repo` points at a Practical-RIFE checkout (with `train_log/`).
  If a different version returns a different `flownet.forward` signature, adjust
  `RifeTorchScriptWrapper.forward` in `scripts/make_rife_torchscript.py`.

## License

`scripts/make_rife_torchscript.py` itself is fine to publish. The **RIFE model code and weights
(`flownet.pkl` / the generated `rife.pth`) come from Practical-RIFE and carry non-commercial terms** -
check the upstream license before redistributing. https://github.com/hzwer/Practical-RIFE

## Implementation notes

- RIFE runs in **fp16 by default** (following the pipeline's `--fp16`; fp32 when `--fp16` is off). The
  bundled IFNet's warp builds its sampling grid in the flow's dtype, so `grid_sample`'s dtype-match
  requirement holds under fp16. External TorchScript checkpoints that bake a float32 grid into their warp
  are detected by a probe inference at init and **fall back to fp32 automatically** (a warning is logged;
  processing continues). Frames round-trip through uint8 either way, so the output path is unchanged.
- Measured speedup (RTX 5060 Ti, 1080p, `--frame-gen 2x`, lada-yolo-v4, end-to-end pipeline):
  16.5 fps with an fp32 checkpoint → **31.4 fps with fp16 (~1.9x)**. The fp16 and fp32 outputs average
  ~50 dB PSNR against each other (visually identical), so there is no quality reason to prefer fp32.
- Interpolation runs at full resolution on the blend-encode thread (v1). TensorRT and a dedicated stage
  are possible future work.
- The `rtx` backend (`jasna/framegen/rtx_frame_generator.py`) wraps nvvfx `VideoFrameGeneration` the same way
  `rtx-super-res` wraps `VideoSuperRes`: CHW uint8 -> contiguous float32 [0,1] -> `run_at_timestep(prev, cur, t,
  stream_ptr=<current torch stream>)` -> clone (the SDK reuses one output buffer) -> uint8. The effect is bound
  to one input size, so it is loaded lazily from the first frame pair and re-loaded on a size change.
