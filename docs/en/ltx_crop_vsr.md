# Sharpening an LTX output: `--ltx-crop-vsr` (`+modi`)

`--ltx-crop-vsr` takes a video the official release restored with LTX, redraws only
the areas that had mosaics with SwiftVR at x2, and composites them back onto the LTX
output. The output keeps the LTX output's resolution, frame count and timestamps.

This modi build is made from the public source, so it cannot decrypt the LTX weights
(`.enc`) and cannot run the LTX restoration itself. LTX's scan (mosaic tracking, the
cameras and the windows) and compose (the feathered composite) are public, though, so
the work splits: the official release restores with LTX, and modi sharpens its output.
What the sharpening recovers is the blur LTX leaves where it enlarges its 512 px canvas
(768 px with `--ltx-large-canvas`) back to the crop size bicubically. In the user's
visual check (720x480 footage) the sharpness ranked SwiftVR > distill ≈ CAS > plain
LTX, with no objectionable flicker.

## Usage

```
jasna --input <LTX output> --output <output> \
      --ltx-crop-vsr swiftvr --ltx-crop-vsr-source <source video> \
      --swiftvr-repo <SwiftVR checkout> --swiftvr-model-dir <checkpoint> --cq 14
```

| Flag | Meaning |
| --- | --- |
| `--input` | The video the release restored with LTX. The frames, audio, subtitles and chapters of the output come from it. |
| `--ltx-crop-vsr-source` | The video it was restored from (the one with the mosaics). It is scanned for the mosaics and tells the windows LTX did restore; the LTX output has no mosaics left to detect, so it is required. |
| `--ltx-crop-vsr swiftvr` | The full SwiftVR model through the `swiftvr-inline` worker at scale 2. Uses `--swiftvr-repo`, `--swiftvr-python`, `--swiftvr-model-dir` and `--swiftvr-accel` ([swiftvr.md](swiftvr.md)). |
| `--ltx-crop-vsr swiftvr-distill` | The distill student inside the jasna process (x2). Uses `--swiftvr-distill-model`, `--swiftvr-distill-strength` and `--swiftvr-distill-stabilize`. Lighter than SwiftVR, adds more texture and more flicker. |
| `--ltx-large-canvas` | Only if the LTX run restored at 768 px. The refine uses 768 px without looking at the free VRAM. |

Use the detection model (`--detection-model`) the LTX run used; the threshold and
`--batch-size` need not match. The refine computes its cameras from the source on its
own and feeds SwiftVR crops of the LTX output, so a crop or polygon that differs a
little from the LTX run only widens or narrows the sharpened area. LTX outputs made with
`--ltx-auto-split` or `--segments` are accepted too.

### What the LTX run should do

- Write the LTX output at a high quality (low CQ). The refine re-encodes the picture
  outside the masks as well, so keep its `--cq` low too.
- Apply `--lut` and `--sharpen` in the LTX run; the refine refuses them (applied in both
  runs they would be applied twice).
- If the LTX run had `--ltx-large-canvas`, pass it to the refine too unless its log shows
  the "restoring large mosaics at 512 px" warning.
- Prefer hevc (10-bit) output. The refine's default `--codec` is hevc at 10 bit as well,
  so the untouched picture keeps its bit depth.

### Examples

The workflow is two passes: the official release restores with LTX, this build
sharpens that output. Give both passes the same low `--cq` and the same encoder
settings, because the refine re-encodes the whole picture once more and the parts
outside the mosaics should not lose quality on the way.

```bash
# Pass 1, in the official release: LTX restoration at high quality.
jasna --input source.mp4 --output ltx.mkv \
      --restoration-model-name ltx --detection-model rfdetr-v6 \
      --codec hevc --cq 14 --encoder-settings "preset=p7,rc-lookahead=32,qmax=28"

# Pass 2, in this build: sharpen the mosaic areas of the LTX output with SwiftVR.
jasna --input ltx.mkv --output ltx_sharp.mkv \
      --ltx-crop-vsr swiftvr --ltx-crop-vsr-source source.mp4 \
      --detection-model rfdetr-v6 \
      --swiftvr-repo ~/SwiftVR --swiftvr-model-dir ~/SwiftVR/checkpoints \
      --codec hevc --cq 14 --encoder-settings "preset=p7,rc-lookahead=32,qmax=28"
```

The same pass 2 with the distill student instead of SwiftVR (no checkout or venv;
`swiftvr-distill.pt` in `model_weights/`):

```bash
jasna --input ltx.mkv --output ltx_sharp.mkv \
      --ltx-crop-vsr swiftvr-distill --ltx-crop-vsr-source source.mp4 \
      --detection-model rfdetr-v6 --swiftvr-distill-strength 0.75 \
      --codec hevc --cq 14 --encoder-settings "preset=p7,rc-lookahead=32,qmax=28"
```

What the encoding options do here:

- `--cq 14` is far below the HEVC default of 28 (lower = better quality, bigger
  file; the number is passed to NVENC as it is). Pass 1 decides the quality of the
  picture outside the mosaics; pass 2 only has to preserve it. On a low-bitrate
  source the automatic output-size ceiling can cap a very low CQ in pass 1 (see
  [Encoder settings](cli.md#encoder-settings)); pass 2 inherits a high-bitrate
  source, so it is not affected.
- `--encoder-settings "preset=p7,rc-lookahead=32,qmax=28"`: `preset=p7` is NVENC's
  slowest and best preset (default `p5`), `rc-lookahead=32` lets the rate control
  look 32 frames ahead, and `qmax=28` caps the worst quantizer VBR may fall back to
  (default 34). Any key from the [NVENC table](cli.md#nvidia-nvenc-keys-all-codecs)
  works; an unknown key stops the job with the list of accepted keys.
- `--codec hevc` is the default in both passes and keeps the untouched picture at
  10 bit. If the LTX output was written as 8-bit `h264`, pass `--codec h264` to the
  refine as well, so the bit depth does not change.
- `--ltx-fast`, `--restoration-model-name ltx-undistilled` and `--ltx-seed` belong to
  pass 1 only and do not affect pass 2. `--ltx-large-canvas` goes into both passes
  (see above). `--lut` and `--sharpen` go into pass 1 only.

### Refused combinations

`--restoration-model-name` other than basicvsrpp, `--secondary-restoration`, `--denoise`,
`--segments`, `--ltx-auto-split`, `--stream`, `--retarget-high-fps`, `--fmp4`,
`--frame-gen`, VR (`--vr-mode` resolving to anything but off), image and folder input,
`--lut`, `--sharpen`, `--swiftvr-scale 4`, a non-default `--swiftvr-view-window` or
`--swiftvr-distill-view-window`, and `--ltx-large-canvas` together with
`--swiftvr-distill-stabilize`. The GUI does not offer the mode (CLI only, one video at a
time).

## How it works

The implementation is `jasna/ltx/refine.py`. BasicVSR++ is not loaded, and only the
detector's engine is compiled.

1. **Alignment check.** Both videos are demuxed packet by packet (no decoding, a few
   seconds): the frame counts and frame sizes must match, and each frame's time from its
   stream's start, (pts − start_pts) × time_base, must agree within half a frame. Raw
   timestamps are not compared: jasna's encoder drops the source's start offset, and an
   MKV changes the time base.
2. **Scan.** Upstream's `scan_span` runs on the source and yields the same tracks,
   cameras and windows as LTX. The `LTX plan: N frames, C scene cuts, T tracks, W windows`
   log line can be checked against the release's log. The detector is released after the
   scan, before SwiftVR starts.
3. **Sharpen and composite in one pass.** The LTX output and the source are decoded in
   lockstep by two NVDEC readers. For every window, as each frame arrives, the crop is
   cut from the LTX output, shrunk (pixel area) or enlarged (bicubic) to the canvas size
   and reflect-padded. Once a window's frames are all in, it is upscaled x2, and the
   result, paired with the window's `CropFrame` doubled, goes through upstream's
   `composite_frame` onto the LTX frame. The feathered mask, the crossfade between
   windows, the handover between tracks and the untouched picture outside the masks are
   LTX's own. The colour match (ring median × 0.75) pulls SwiftVR's colours toward the
   LTX output.
4. **Windows LTX left alone are skipped.** The mean luma difference between the LTX
   output and the source inside the track polygon is accumulated per window; below the
   threshold (3/255) the window is neither upscaled nor composited, and a warning names
   it. This catches a track the refine finds that LTX did not (a numeric difference
   between the release's and modi's detection engines; the mosaic is still in the LTX
   output and would be sharpened) and a wrong `--ltx-crop-vsr-source` with the same
   frame count. The reverse (a track LTX had and the refine misses) leaves that stretch
   as LTX made it, which is harmless.
5. **Encode.** The normal NVENC path. The writer gets the LTX output's timestamps, so the
   output has the same timestamp sequence.

### Memory

- Held frames: a frame cannot be written until the windows covering it are upscaled. At
  most 121 frames are held whatever the number of tracks (every window covering frame f
  ends by f + 121). They are page-locked uint8 host buffers: about 0.75 GB at 1080p, 3 GB
  at 4K.
- Upscaled windows: the previous window of each track stays for the crossfade. One
  window is about 380 MB at 1024² and 860 MB at 1536² (host).
- VRAM: the SwiftVR worker (FP8, 1024 px), two decoders and the encoder. The distill
  student adds under 1 GB with stabilize 0 (the default); with stabilize > 0 it holds the
  whole window in float, about 4.5 GB at the 512 canvas and about 10 GB at 768, so that
  combination is refused.

## Verification

Linux, RTX 5080 16 GB, 2026-10-16. Footage: two LTX outputs of release v0.11.1 with
their sources (720x480 with 890 frames and 1080p with 899 frames, 30 s each, the LTX
outputs at CQ 14).

- **Skip threshold.** The mean luma difference of the windows LTX restored is 4.6 to
  10.4; outside the mask in the same crops (the re-encode noise) it stays under 1.0. The
  threshold of 3.0 sits between. All 11 windows of both clips count as restored; nothing
  is skipped.
- **Completion and alignment.** Both clips with both backends complete, and the scan's
  plan matches the release's log (2 tracks, 11 windows, 1 scene cut). The output has the
  LTX output's frame count, and the frame times differ by at most 0.5 ms (the MKV time
  base of 1/1000). The clip whose source has a non-zero start_pts passes too.
- **Metrics** (mioh-labs' texture and flicker, against a re-encode through the same
  path): the distill student at strength 0.75 gives texture 119.9% / 121.7% at a flicker
  increase of +11.2% / +8.4% (720x480 / 1080p), within the planned tolerance of the probe
  (118.8% / 118.4%, +14.2% / +9.4%). SwiftVR at full strength gives 114.6% / 120.9% and
  +8.5% / +10.1%. The probe's SwiftVR rows (108.7% / 110.5%) were a 0.75 blend with the
  bicubic upscale, so they do not compare directly; the probe rerun at full strength on
  the 720x480 clip gives 112.8%, within 2 points. Disabling LTX's colour match raises the
  distill flicker by 2 to 3 points (SwiftVR, with its own wavelet colour fix, does not
  move), so the colour match stays.
- **Time and VRAM** (30 s of footage, worker start included): the distill student takes
  23 to 28 s of wall time and adds 2.5 to 2.8 GB of VRAM; SwiftVR takes 50 to 55 s and
  adds 9.8 to 10.8 GB (more than the 7 GB estimated at design time; a 16 GB card, the
  distill student on 12 GB cards). The scan takes 3 s.
- **The user's visual check** (2026-10-16, SwiftVR at full strength, five clips of 404p to
  720p footage, 5 to 60 s): the sharpening is as intended, with no objectionable flicker.
- Pending: the 768 px canvas, footage whose crops are under half the canvas, footage over
  10 minutes, Windows.

## Limitations

- Tracks whose crop is under half the canvas side (small mosaics in 1080p or 4K footage)
  are unverified: a small crop is enlarged a lot before the upscale, which may behave
  differently.
- The upscale runs synchronously inside the decode loop; it does not overlap with
  decoding, compositing or encoding.
- An LTX output made with `--retarget-high-fps` is refused by the frame-count check.

## License

Post-processing an LTX output is within §5 (Output) of the LTX-2.x Community License;
the `.enc` files are never touched. See the upstream documentation for obtaining and
running LTX.
