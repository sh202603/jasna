# Restoring Only Parts of a Video

You don't have to restore a whole video. Pick just the scenes that need it —
the rest of the video is copied through untouched, which is much faster and
keeps the original quality everywhere else. The output is still the
full-length video.

## Segment Editor (GUI)

<img width="822" alt="Segment editor" src="https://github.com/user-attachments/assets/c67939a9-37de-46ae-b722-20c6a0933df7" />

Open it with the scissors button on any queued video.

- Preview the video and select frame-accurate ranges. Leave the selection
  empty to restore the whole video.
- **Restore preview** shows the current frame or a short playback with your
  current restoration settings, before you commit to processing.
- Use the visible **− / +** controls or the mouse wheel to zoom into details,
  then drag the preview to pan. **Reset view** appears only after the view has
  been changed.
- When ranges are selected, the export keeps the source video codec — the
  main **Encoding** codec setting does not apply, because unselected parts
  are copied as-is. The editor tells you upfront if a video can't be used
  with segment processing.

### Mosaic scanning

The editor can find mosaic scenes for you:

- Scan every frame or at 0.25–2 second intervals. Scanning runs on the GPU
  and reaches about **2,000 FPS on an RTX 5090**.
- After scanning, adjust the confidence slider to update the amber detected
  ranges, then add them to your purple restoration selection with one click.

The detection model and confidence are remembered per queued video and used
during final processing, so different videos can use different settings.

### A/B model comparison

The **A/B compare…** button below the preview puts two detection/restoration
model combinations side by side on the same frame. When you upgrade a model
or try a fine-tuned checkpoint, you can see the difference in seconds instead
of processing the whole video twice.

- Three views: **Restored** (the restored frame), **Detection** (the
  detection mask overlay with its score), and **Clip** (a short playback,
  synchronized between both panes).
- Pick each side's detection model, confidence, and restoration checkpoint
  (any `.pth` directly under `model_weights`), then press **Run** to process
  both sides in sequence. A side whose selection or time has changed since
  the last run shows a "(stale)" marker.
- Zoom and pan always stay synchronized between the two panes.
- Each side's badge shows the actual execution path: **TRT** means TensorRT
  engines, **PyTorch** means the slower fp32 fallback for a checkpoint
  without compiled engines (minor quality differences vs TRT are expected).
  The **Compile engines** button builds the engines for that checkpoint
  (15-60 min on first run).
- Checkpoints without EMA weights cannot work correctly (they would run with
  random weights), so they are explicitly refused before running.

### Suggesting better masks

When a detection looks wrong, you can help improve future models. Pause on
the frame, click **Suggest better mask**, and outline each mosaic area — the
editor guides you through drawing. If the mosaic fades out with soft or
blurry edges, include that soft region too.

Submitting uploads the frame and your mask **anonymously**. The data is
encrypted on your machine before upload, and the only attached details are
the app version, the detection model name, and the frame resolution — never
file names, timestamps, or anything identifying you.

## CLI: `--segments`

The CLI counterpart of the editor. Times accept seconds or `HH:MM:SS.s`:

```bash
jasna --input input.mp4 --output output.mp4 --segments "10-25,01:10-01:30.5"
```

Jasna restores the listed ranges and copies everything else. A short
transition around the nearest safe video cut points is re-encoded but not
restored — this is required to splice the video back together seamlessly.

What works with segment processing:

- NVIDIA and AMD GPUs.
- One video at a time (no folders, images, or streaming).
- H.264, HEVC, or AV1 input with constant frame rate; MP4, MOV, or MKV output.
- On AMD, H.264 sources with more than three consecutive B-frames are not supported.
- The output codec always matches the input codec.
- Cannot be combined with `--retarget-high-fps`.
- Cannot be combined with `--fmp4`; the output is assembled after processing finishes.

Incompatible input is rejected with a clear error before processing starts.

## Working directory

While assembling segment output, Jasna keeps temporary files in a hidden
folder next to the output video. If you want them on another (faster or
bigger) drive, set the **Working directory** in the GUI's Encoding section,
or use the CLI flag:

```bash
jasna --input input.mp4 --output output.mp4 --segments "10-25" --working-directory /fast/scratch
```
