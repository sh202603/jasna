# `--fmp4`: MP4 output playable while processing runs

`--fmp4` lets you check the result of a file export while the run is still going.
When enabled, `.mp4` / `.mov` output is written as a **fragmented MP4** (fMP4), so the file can be opened and played mid-run.
If processing is interrupted, the file stays playable up to that point.
The default is off, in which case behavior is byte-identical to before.

Up to v0.7.x a raw .hevc stream was written to the output folder during processing, and opening it showed how far the run had got.
The v0.8.0 media-layer rewrite moved to writing the final file directly via PyAV, which removed that check.
This option replaces it.

## Usage

```bash
jasna --input in.mp4 --output out.mp4 --fmp4
```

Open the in-progress output file in a player such as mpv to review the result so far, audio included.

In the GUI, the "Fragmented MP4" toggle in the encoding settings does the same thing.
There is no behavioral difference between the CLI and the GUI.

## How it works

### Why a normal MP4 can't be played while it is being written

Playing an MP4 requires the **moov atom** — the codec initialization data plus an index of every sample's timestamp and byte offset.
On a normal write the muxer accumulates that index in memory and writes it exactly once, when the container is closed.
A partially written file therefore has no index, and a player cannot interpret it.

jasna's default additionally passes `movflags +faststart`, which runs a second pass at close to relocate the moov to the head of the file.
That speeds up progressive playback of the *finished* file; it does nothing for a file still being written.

### Fragmented MP4

With `--fmp4`, movflags become `+frag_keyframe+empty_moov`.

- **empty_moov**: a moov without sample data is written at the head of the file immediately, so a player gets codec initialization data the moment it opens the file.
- **frag_keyframe**: media is written incrementally as per-keyframe moof + mdat fragments instead of one giant mdat with an index at the end. Each moof carries the sample table for its own mdat, so the playable range grows every time a fragment completes.

Fragment granularity follows the keyframe interval; with the NVENC default (`g=250`) that is roughly every 8 s at 30 fps.
`+faststart` is dropped because the mov muxer ignores it in fragmented mode.
`+default_base_moof` is deliberately omitted — it targets the browser Media Source Extensions and is unnecessary for local players.

jasna's native encoder muxes audio interleaved with video as it goes, so mid-run playback includes audio.

A side benefit: with no moov relocation pass at close, the run finishes faster, and a crashed or interrupted run leaves the fragments written so far as a playable file.

## Combinations that opt out

None of these are errors: each warns and continues with normal MP4 output.

| Condition | File output |
|---|---|
| combined with `--stream` | ignored (warning) — streaming mode saves no file, so the flag is meaningless |
| `--segments` (smart render) | falls back to a normal MP4 (warning) |
| `--secondary-restoration flashvsr` | not supported (warning) |
| `--encode-backend torchcodec` | not fMP4 (warning) |
| folder (batch) input | no restriction; fMP4 per file |
| non-mp4/mov output such as `.mkv` | movflags don't apply (already progressively playable) |

`--segments` assembles the output after processing finishes, so there is no growing file to play.
The torchcodec encode backend writes a temp file and remuxes at the end, so mid-run playback gains nothing.

## Limitations

- The up-front moov of an fMP4 carries no total duration, so a player opening the in-progress file may show an incomplete duration and seek imprecisely.
- Very old players that do not support fragmented MP4 cannot play the output.
- No measurable throughput cost (measured level with the baseline).
- Combines fine with `--frame-gen` / `--retarget-high-fps`.
- Interrupting with Ctrl+C does not terminate the process immediately (the interrupt lands while worker threads are being joined — pre-existing behavior, independent of this flag). The output file is left on disk playable up to that point.

## Implementation and tests

The main changes:

- `jasna/media/video_encoder.py`: movflags selection extracted into the pure function `_mov_container_options()`, plus an `fmp4` parameter on `NvidiaVideoEncoder`.
- `jasna/media/backend.py`: pass-through in `make_video_encoder()` and a warning when the torchcodec backend is selected.
- `jasna/pipeline.py`: the non-mp4 output / `--segments` fallbacks and handing the flag to the encoder.
- `jasna/main.py`: the flag definition and the warnings for `--stream`, `--segments` and flashvsr.
- GUI (`jasna/gui/`): `AppSettings.fmp4`, the encoding-settings toggle, locale strings in all five languages.

Tests were added to `test_video_encoder_unit.py` (`_mov_container_options` unit tests), `test_video_backend_fallback.py` (backend dispatch), `test_pipeline_run.py` (fallbacks), `test_main_validation.py` (CLI wiring), `test_locales.py` and `test_preset_migration.py`.
The CUDA-gated ones (fMP4 box layout, mid-run playback of a truncated file, audio mux — in `test_video_encoder_mux.py`) are code-only additions.

## Verification status

Both the CPU and CUDA tests pass (full suite: 1754 passed / 21 skipped).
The remaining 9 failures are pre-existing, caused by the missing `jasna.protection` module and a GUI layout test; they were confirmed to fail identically at the pre-change HEAD in a worktree.

Hardware verification was done on an RTX 5080 (Linux, a 1920x1080 29.97 fps 140 s clip — 4203 frames, AAC audio):

**Container layout**: with the flag on, the top level is `ftyp / moov / moof+mdat ×17 / mfra` — the moov comes first and fragments follow. The default output is `ftyp / moov / free / mdat` with a single mdat.

**Mid-run playback**: a copy of the output file taken at 61% progress decoded 2250 frames plus AAC audio with the flag on; the same copy taken with the default settings failed with `moov atom not found`. This confirms both the fMP4 effect and that mid-run playback includes audio.

**Fallbacks**: combining with `--stream` warns (`--fmp4 is ignored with --stream`) and continues as a normal streaming run rather than exiting on an argument error. `--segments` warned and produced a normal MP4 (moof count 0). `.mkv` output was produced normally as matroska. Folder input produced fMP4 for each output file (moof count 17) with no warning.

**Performance**: no measurable difference. Three alternating runs each gave 141.1 / 141.3 / 142.3 fps for the baseline and 141.8 / 140.0 / 140.8 fps with the flag on — the spread of both sits within run-to-run noise.

**Output parity**: baseline and flag-on both produced 4203 frames with AAC audio (the fMP4 container reports a total duration 0.11 s longer, 140.37 s, but the decoded frame count is identical).

The `--encode-backend torchcodec` fallback is covered by unit tests only — the optional dependency was not available here.
