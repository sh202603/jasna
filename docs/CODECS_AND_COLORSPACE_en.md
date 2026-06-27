# Output codec, bit depth, and colorspace (AV1 / 8-bit NV12 / BT.601 & BT.2020)

A guide to the output-format flexibility in `v0.7.2+modi` (introduced in `v0.7.1+modi`).

## Overview

Jasna's processing path is a "zero-copy" **NVDEC → restoration → NVENC** pipeline that stays on the
GPU end to end, with no CPU round-trips (enforced by `tests/test_no_cpu_tensor_ops.py`). This release
extends the output stage, which was previously fixed to **HEVC / 10-bit (P010) / BT.709**.

| Aspect | Before | This release |
|---|---|---|
| Codec | HEVC only | **HEVC / AV1** |
| Bit depth | Always 10-bit (P010) | **Follows the source (8-bit NV12 / 10-bit P010), or explicit** |
| Colorspace | Forced to BT.709 | **Preserves BT.601 / BT.709 / BT.2020 from the source** |

## CLI options

### `--codec {hevc,av1}`
Output codec. Default `hevc`. `av1` is supported for file output only (see Limitations).

### `--bit-depth {auto,8,10}`
Output bit depth. Default `auto`.
- `auto`: 10-bit (P010) if the source is 10-bit, otherwise 8-bit (NV12).
- `8`: always 8-bit (NV12).
- `10`: always 10-bit (P010).

> **Note:** the restoration pipeline runs in 8-bit RGB internally, so encoding an 8-bit source as
> 10-bit gains no information; it only widens the container. `auto` is recommended.

### Examples

```bash
# 8-bit source → 8-bit HEVC (auto)
jasna --input in_8bit.mp4 --output out.mkv

# AV1 output (bit depth follows the source)
jasna --input in.mp4 --codec av1 --output out_av1.mkv

# Force an 8-bit HEVC output from a 10-bit source
jasna --input in_10bit.mp4 --bit-depth 8 --output out_8bit.mkv

# BT.2020 source → AV1 10-bit, preserving the colorspace
jasna --input in_bt2020.mp4 --codec av1 --bit-depth 10 --output out.mkv
```

## Colorspace handling

The input colorspace (ffprobe's `color_space`) is detected and encoded with the matching
limited-range RGB→YUV coefficients. The output container is tagged with the colorspace metadata
(matrix + primaries + transfer).

| Source color_space | Matrix used | Output metadata (colorspace / primaries / transfer) |
|---|---|---|
| `bt709` (default) and others | BT.709 | `bt709` / `bt709` / `bt709` |
| `smpte170m` / `bt601` / `bt470bg` | BT.601 | `smpte170m` / `smpte170m` / `smpte170m` |
| `bt2020nc` / `bt2020c` | BT.2020 | `bt2020nc` / `bt2020` / `bt2020-10` |

**The output tags are the canonical triple derived from the matrix family, not a field-by-field
copy of the input tags.** Detection looks only at the input `color_space` (matrix), maps it to a
family in the table above, and re-tags with that family's (colorspace, primaries, transfer). So if
the input's primaries / transfer disagree with its matrix, the output is normalized to the canonical
values of the detected family.

> Example: an input with `color_space=smpte170m` (BT.601) but `color_transfer=bt709` (a mismatch
> common in SD content) is detected as BT.601, and the output transfer is normalized to `smpte170m`
> (`bt709` is not carried over). BT.2020 uses `bt2020-10` for transfer because the matrix name
> `bt2020nc` is not a valid `color_trc`, so matrix and transfer must be kept as distinct values.

## Output container

**The final container is chosen by the output file extension.** `.mkv` produces Matroska, while
`.mp4` / `.mov` produce MP4/MOV (**AV1-in-MP4 works**; both HEVC and AV1 `.mp4` output verified on
hardware).

Muxing is a two-stage process:

1. The NVENC elementary stream (HEVC `.hevc` / AV1 `.obu`) is assembled into an intermediate file
   with **mkvmerge** (which applies timecodes).
2. **ffmpeg** does the final remux: it muxes in the audio and tags the color metadata (matrix /
   primaries / transfer / range). The video is `-c:v copy` (no re-encode). For `.mp4` / `.mov`
   outputs it adds `-movflags +faststart`.

## Limitations

- **AV1 is file output only.** AV1 cannot be selected in streaming mode (`--stream`); HEVC is used
  there. `--codec av1 --stream` is rejected.
- **AV1 uses no B-frames.** NVENC AV1 does not support B-frames, so they are disabled internally.
- **HDR transfer characteristics (PQ / HLG) are not handled.** Only the colorspace matrix and
  primaries are preserved; transfer characteristics are not carried over. Full HDR preservation is
  out of scope.
- **Full-range (JPEG / PC range) output is not supported.** The conversion is limited-range
  (MPEG / TV range) only; a full-range input is rejected.
- **AV1 muxing.** The NVENC AV1 elementary stream (OBU) is fed to mkvmerge via a `.obu` temp file.
  Verified working on RTX 5060 Ti (Blackwell) with ffmpeg 8 / mkvmerge v97. Older mkvmerge versions
  that cannot ingest OBU may need the stream wrapped in IVF or muxed directly with ffmpeg.

## GUI

The "Encoding" section of the settings panel adds **Codec** (HEVC / AV1) and **Bit Depth**
(Auto / 8 / 10) dropdowns. The GUI builds the same Pipeline as the CLI, so behavior matches.

## Implementation notes (for developers)

- RGB→surface conversion: `chw_rgb_to_surface(frame, colorspace, bit_depth)` in
  `jasna/media/rgb_to_p010.py`. It builds a limited-range matrix from (Kr, Kb) and returns NV12
  (uint8) for 8-bit or P010 (int16, value in the high 10 bits) for 10-bit.
- Colorspace detection/preservation: the `Colorspace` enum and `VideoMetadata.yuv_colorspace` in
  `jasna/media/__init__.py` (av's `Colorspace` enum cannot represent BT.2020, so it is tracked
  separately).
- Encoder: `jasna/media/video_encoder.py` derives `fmt` (P010/NV12), `profile`, B-frame count, and
  the temp-file extension (`.hevc`/`.obu`) from codec + bit depth.
- Full-range detection: in `jasna/media/__init__.py`, a ffprobe `color_range` of `pc` or `jpeg` is
  treated as full range (`AvColorRange.JPEG`) and rejected at the start of the pipeline; `tv` /
  absent / `unknown` map to limited (MPEG). ffprobe reports full range as `pc` (not `jpeg`), so it
  is important not to miss the `pc` token.
