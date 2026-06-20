# Frame generation (frame-rate up-conversion)

`--frame-gen {none,2x,4x}` raises the output frame rate by 2x/4x (file output only; not for `--stream`).
It inserts AI-interpolated frames between the source frames with new PTS placed between the originals.
Audio keeps the original timecodes, so duration and sync are preserved.

Backend: `--frame-gen-backend {rife,rtx}`
- `rife` (default): neural interpolation. **Available now**; supply weights separately.
- `rtx`: NVIDIA RTX Video Frame Generation. The SDK effect is not yet shipped in `nvidia-vfx`, so this
  **currently raises a clear error**; it will activate once NVIDIA releases the effect.

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

The script `make_rife_torchscript.py` lives at the repo root.

```powershell
.\.venv\Scripts\python.exe make_rife_torchscript.py `
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

## Two-pass workflow: `jasna-framegen` (standalone)

`jasna-framegen` is a separate command that applies **only** frame generation to an
already-restored video — no mosaic detection, no BasicVSR++ restoration. Use it when:

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

Common options: `--factor {2x,4x}`, `--backend {rife,rtx}`, `--model-path <rife.pth>`,
`--codec {hevc,av1}`, `--bit-depth {auto,8,10}`, `--encoder-settings <k=v,...>`,
`--device cuda:0`, `--no-fp16`. Output quality defaults to jasna's encoder profile
(cq=25); override with `--encoder-settings`. Run `jasna-framegen --help` for the full
list. Verify the same way as step 4 (`ffprobe` → ~2x/4x frame rate, unchanged
duration, audio in sync).

---

## Troubleshooting

- **`RTX Video Frame Generation is not available ...`**: `--frame-gen-backend rtx` is not shipped; use `rife`.
- **`RIFE weights not found: ...`**: no `model_weights\rife.pth`; create one (steps 1-2) or pass `--frame-gen-model-path`.
- **`RIFE state_dict loaded non-strictly (...)`**: you dropped a raw `flownet.pkl` whose keys do not match
  the vendored IFNet; results will be wrong - switch to the TorchScript method.
- **Converter import error**: check `--rife-repo` points at a Practical-RIFE checkout (with `train_log/`).
  If a different version returns a different `flownet.forward` signature, adjust
  `RifeTorchScriptWrapper.forward` in `make_rife_torchscript.py`.

## License

`make_rife_torchscript.py` itself is fine to publish. The **RIFE model code and weights
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
  ~50 dB PSNR against each other — visually identical, so there is no quality reason to prefer fp32.
- Interpolation runs at full resolution on the blend-encode thread (v1). TensorRT and a dedicated stage
  are possible future work.
