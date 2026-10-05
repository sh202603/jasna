# Proposal: a lightweight five-frame ROI Distill secondary restorer

[日本語版](../ja/roi-distill-secondary-proposal.md)

This is a **design proposal and reference adapter**, not a working Jasna feature.
It targets the `modi` pipeline inspected at commit
`4120fac8e749a12afa72c2f13319cad261949de1`. No checkpoint, media, dataset,
private paths or changes to supporter-model protection are included.

## Motivation and integration point

Add an optional small PyTorch secondary restorer after BasicVSR++, using five
restored RGB crops to enhance the centre crop from 256 to 512 pixels.
Detection, tracking and the primary restorer remain unchanged. The output video
resolution and frame rate would still be controlled by the existing pipeline.
This does not add full-frame 4K upscaling or change object separation.

The model used for the local contract checks has 24 channels, 12 residual
blocks, five input frames and 2x output, with 130,860 parameters. Its operations
are Conv2d, ReLU, residual additions and PixelShuffle. A learned residual is
added to a bilinear 2x upscale of the centre input. It has no diffusion process,
explicit optical flow, or recurrent previous-output state.

This fits the existing synchronous `SecondaryRestorer` interface:

- Primary input to the adapter: float RGB `[T, 3, 256, 256]` in `[0, 1]`.
- Model input: FP32 `[B, 15, 256, 256]`, channels ordered as five consecutive
  RGB frames `t-2, t-1, t, t+1, t+2`.
- Model output: FP32 `[B, 3, 512, 512]`, representing the centre frame only.
- Adapter return: ordered uint8 RGB `[3, 512, 512]` tensors for the clamped
  `[keep_start, keep_end)` range.

The existing `RestorationPipeline._run_secondary`, `scale_offsets` and
`BlendBuffer` already have the relevant secondary-output/scale contracts.
The proposed adapter would stay in the same GPU process, keep the model loaded
for the session, and avoid per-frame checkpoint loading, CPU round trips and
intermediate video files.

## Reference adapter

`model` below must be an instance of the **matching model architecture**, already
loaded with a trusted checkpoint using `weights_only=True` and
`load_state_dict(..., strict=True)`. The example deliberately does **not** include
the model definition, checkpoint loader, registration or CLI changes.
Selecting an arbitrary `best.pt` is not sufficient to run it.

Start with FP32 eager execution, batch size one, and view smoothing disabled to
establish a numerical baseline before changing precision or geometry.

```python
import torch

class RoiDistillSecondaryRestorer:
    name = "roi-distill"
    num_workers = 1
    prefers_cpu_input = False

    def __init__(self, model, device, view_window=0):
        self.device = torch.device(device)
        self.model = model.eval().to(self.device, dtype=torch.float32)
        self.view_smoothing_window = int(view_window)

    @torch.inference_mode()
    def restore(self, frames_256, *, keep_start, keep_end):
        if frames_256.ndim != 4 or tuple(frames_256.shape[1:]) != (3, 256, 256):
            raise ValueError("expected RGB [T,3,256,256]")
        if not frames_256.is_floating_point():
            raise ValueError("expected float RGB in [0,1], not uint8/BGR")
        t = len(frames_256)
        ks, ke = max(0, int(keep_start)), min(t, int(keep_end))
        if ks >= ke:
            return []
        results = []
        with torch.autocast(device_type=self.device.type, enabled=False):
            for i in range(ks, ke):
                idx = [min(t - 1, max(0, i + d)) for d in (-2, -1, 0, 1, 2)]
                window = frames_256[idx].to(self.device, dtype=torch.float32)
                if not torch.isfinite(window).all():
                    raise ValueError("non-finite ROI input")
                x = window.clamp(0, 1).reshape(1, 15, 256, 256)
                y = self.model(x)
                if tuple(y.shape) != (1, 3, 512, 512) or not torch.isfinite(y).all():
                    raise ValueError("invalid ROI Distill output")
                results.append(y[0].clamp(0, 1).mul(255).round().to(torch.uint8))
        return results
```

This is intentionally a correctness-first example. The per-frame finite checks
may synchronize CUDA; they are not a throughput-optimized implementation.
Batching a small number of output centres is a later optimization, rather than
materializing all temporal windows for a long clip at once.

## Temporal boundaries and crop geometry

Use context from the same ROI clip, repeating the nearest available frame only
at the clip endpoints. For a five-frame clip, the first window is `[0,0,0,1,2]`
and the last is `[2,3,4,4,4]`. A singleton clip can generate exactly one output;
this adapter adds no short-detection rejection rule.

Keep the context frames outside the keep range available as input. They need
not receive their own secondary outputs, although existing crossfade ranges
still require both clip results. Do not share temporal history across unrelated
ROIs or scene cuts. The model needs five input slots but only one output per
selected centre; it has no 28-frame internal warmup output requirement.

Jasna's detection-duration filtering and gap/coasting settings remain separate.
With gap filling enabled, a clip can include synthetic observations. The current
secondary interface does not pass detection masks or original frame indices,
so a special mask-aware skipping policy would require a richer contract.

`view_smoothing_window` can reuse the existing `tracking/crop_view.py` capability:
the pipeline builds a smoothed 256px view, preserves `view_placements`, then
samples the enhanced view directly into the frame bbox. The adapter must not
also undo that coordinate transform. Suggested evaluation: windows 0, 5 and 15.

This existing whole-view compositor is **not** identical to a residual-only
compositor that maps enhancement deltas back onto an untouched primary image.
It also does not provide Apple's VideoToolbox temporal filtering. Temporal
stability and edge quality need video evaluation, not just a checkpoint swap.

## Follow-up implementation, if this direction is useful

1. Add a small inference-only architecture/loader module. Validate checkpoint
   version, architecture, weight keys/shapes and finite weights. Do not import
   an entire training script into the restoration runtime or load a 12-block
   checkpoint into a 6-block model with `strict=False`.
2. Add the synchronous adapter and register `roi-distill` in `SessionConfig`,
   `_build_secondary_restorer` and the CLI. Use a separate secondary checkpoint
   path rather than the primary restoration-model path.
3. Retain the existing primary-output lifetime and threading rules. Do not add
   an independent CUDA stream without explicitly handling dependencies.
4. Add temporal-window, empty-range, singleton, output-scale, padding,
   crossfade, multiple-ROI and scene-cut tests.
5. Add GUI controls, saved settings and session-key invalidation later. Model
   replacement at the same path must invalidate the loaded session/cache too.
6. Compare identical primary/secondary inputs and complete short videos, then
   measure crop-fps, end-to-end video-fps and peak VRAM separately.
7. Only after the FP32 baseline is validated, consider small-batch execution or
   TensorRT/FP16. Quantization and precision changes need their own quality tests.

No new dependency on Core ML, SwiftVR inference or the DLoRAL environment is
required for this adapter. The matching PyTorch architecture and compatible
weights would still need to be supplied. This is a separate optional backend,
not a replacement or bypass for the existing encrypted `unet-4x` feature.

## Validation performed and limitations

Local checks used a trusted matching checkpoint and **synthetic CPU inputs**:

- Output shape and finite values for first, middle and last temporal windows.
- `T=1/2/5`, partial keep ranges, empty keep ranges and `T=0` return counts.
- Exact uint8 output equality against the original FP32 model on the same windows.
- Rejection of invalid spatial shape, uint8 input and NaN input.
- Compatibility of a synchronous sample with the actual `SecondaryRestorer`
  protocol, without accidental `AsyncSecondaryRestorer` classification.
- Actual Jasna crop preparation → smoothed view → model → bbox sampling,
  including 2x padding/valid-region coordinate scaling.

Not yet validated: feature registration, CUDA execution in the complete Jasna
pipeline, real-video detail/flicker/edges, throughput, VRAM, or FP16/TensorRT.
No checkpoint is distributed by this PR, and there is no claim of lossless
detail recovery, quality parity or a measured speedup. Weight distribution
would require a separate review of the model/teacher/training-data terms.

The current tracker can merge overlapping boxes before restoration. Adding a
secondary enhancer does not solve loss of spatial information when a large
merged ROI is reduced to 256px. Keep tracking changes separate so quality and
performance comparisons remain attributable.

Feedback requested: would this small, optional in-process secondary backend be
a useful direction for `modi`, and is the existing `SecondaryRestorer`/crop-view
capability boundary the preferred integration point?
