from __future__ import annotations

import torch
import torch.nn.functional as F

from jasna.accelerator import is_nvidia_device

_KERNEL_CACHE: dict[tuple[str, torch.dtype, str, int], torch.Tensor] = {}

BLEND_DILATION_RATIO = 0.028
BLEND_FALLOFF_RATIO = 0.028


def blend_safe_border_px(frame_height: int) -> int:
    """Crop border (frame px) that fits the whole blend transition.

    The blend reaches dilation+falloff (~0.056*frame_height) past the mask
    edge, while ``expand_bbox``'s default border is max(20px, 6% of the
    detection box), so for typical detections the transition band is cut off
    at the enlarged bbox and the composite steps to the original along the
    box edge. Restorers that resynthesize the crop (not near-identity outside
    the mosaic) advertise ``blend_safe_border = True`` and get this as the
    border floor so the transition ends inside the composite region.
    """
    return round(frame_height * (BLEND_DILATION_RATIO + BLEND_FALLOFF_RATIO)) + 2


def _moving_average(x: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    """Moving average without creating a shape-specific convolution problem."""
    prefix = x.cumsum(dim=dim)
    zero_shape = list(prefix.shape)
    zero_shape[dim] = 1
    prefix = torch.cat((prefix.new_zeros(zero_shape), prefix), dim=dim)
    leading = prefix.narrow(dim, size, prefix.shape[dim] - size)
    trailing = prefix.narrow(dim, 0, prefix.shape[dim] - size)
    return (leading - trailing) / size


def _prefix_box_blur(x: torch.Tensor, kernel_h: int, kernel_w: int) -> torch.Tensor:
    # Kernels are clamped so reflect padding stays valid for small inputs.
    kh = min(kernel_h, 2 * x.shape[0] - 1)
    kw = min(kernel_w, 2 * x.shape[1] - 1)
    pad_h = kh // 2
    pad_w = kw // 2
    padded = F.pad(
        x.unsqueeze(0).unsqueeze(0),
        (pad_w, pad_w, pad_h, pad_h),
        mode="reflect",
    )
    blurred = _moving_average(padded, kw, dim=-1)
    blurred = _moving_average(blurred, kh, dim=-2)
    return blurred.squeeze(0).squeeze(0)


def _box_kernel(
    device: torch.device,
    dtype: torch.dtype,
    axis: str,
    size: int,
) -> torch.Tensor:
    cache_key = (str(device), dtype, axis, size)
    kernel = _KERNEL_CACHE.get(cache_key)
    if kernel is None:
        shape = (1, 1, 1, size) if axis == "w" else (1, 1, size, 1)
        kernel = torch.ones(shape, device=device, dtype=dtype) / size
        _KERNEL_CACHE[cache_key] = kernel
    return kernel


def _conv_box_blur(x: torch.Tensor, kernel_h: int, kernel_w: int) -> torch.Tensor:
    kh = min(kernel_h, 2 * x.shape[0] - 1)
    kw = min(kernel_w, 2 * x.shape[1] - 1)
    padded = F.pad(
        x.unsqueeze(0).unsqueeze(0),
        (kw // 2, kw // 2, kh // 2, kh // 2),
        mode="reflect",
    )
    blurred = F.conv2d(padded, _box_kernel(x.device, x.dtype, "w", kw))
    blurred = F.conv2d(blurred, _box_kernel(x.device, x.dtype, "h", kh))
    return blurred.squeeze(0).squeeze(0)


def _box_blur(x: torch.Tensor, kernel_h: int, kernel_w: int) -> torch.Tensor:
    if is_nvidia_device(x.device):
        return _conv_box_blur(x, kernel_h, kernel_w)
    return _prefix_box_blur(x, kernel_h, kernel_w)


def create_blend_mask(
    crop_mask: torch.Tensor,
    frame_height: int,
    scale_y: float,
    scale_x: float,
) -> torch.Tensor:
    """Create blend mask from detection mask with dilation and falloff.

    Dilation ensures blend weight=1.0 extends past the mask edge to cover
    any adjacent mosaic blocks the detector missed.  Falloff creates a
    smooth transition entirely outside the mosaic area.
    Both are proportional to frame height (~30px each at 1080p).

    ``scale_y``/``scale_x`` are frame pixels per mask pixel, so the mask can
    be processed at a lower resolution than the frame (kernels shrink
    accordingly).  Pass 1.0/1.0 for a mask sampled at frame resolution.
    """
    mask = crop_mask.reshape(crop_mask.shape[-2], crop_mask.shape[-1])
    blend_dtype = mask.dtype if mask.is_floating_point() else torch.get_default_dtype()

    dilation_px = max(3, round(frame_height * BLEND_DILATION_RATIO))
    falloff_px = max(3, round(frame_height * BLEND_FALLOFF_RATIO))

    dilation_y = max(1, round(dilation_px / scale_y))
    dilation_x = max(1, round(dilation_px / scale_x))
    falloff_y = max(1, round(falloff_px / scale_y))
    falloff_x = max(1, round(falloff_px / scale_x))

    blend = (mask > 0).to(dtype=blend_dtype)
    blend = _box_blur(blend, dilation_y * 2 + 1, dilation_x * 2 + 1)
    blend = (blend > 0.01).to(dtype=blend_dtype)
    blend = _box_blur(blend, falloff_y * 2 + 1, falloff_x * 2 + 1)

    return blend.clamp_(0.0, 1.0)


def create_bbox_blend_mask(
    mask_lr: torch.Tensor,
    bbox_xyxy: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
    out_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Blend mask for a frame-coords bbox, returned at bbox resolution.

    Computed on the low-res detection mask (cheap), then bilinearly
    upsampled to bbox size — instead of nearest-upsampling the mask first
    and blurring with frame-scale kernels.

    ``out_size`` overrides the output resolution ((y2-y1, x2-x1) by default):
    the restoration pipeline samples the same weight profile at the 256-space
    crop size for its outside-mask revert.

    The blur window is the bbox slice grown by the kernel reach on every side
    (clamped at the mask border = frame border), then cut back to the bbox:
    blurring the bare slice would let the reflect padding mirror any mask
    within kernel reach of a slice edge and pin the blend high along that
    edge, printing the bbox rectangle into the composite. With the grown
    window the profile matches a full-frame computation; only sides clamped
    at the frame border keep the reflect-pad termination (no seam exists
    there).
    """
    x1, y1, x2, y2 = bbox_xyxy
    frame_h, frame_w = frame_shape
    hm, wm = mask_lr.shape

    my1 = (y1 * hm) // frame_h
    my2 = ((y2 - 1) * hm) // frame_h + 1
    mx1 = (x1 * wm) // frame_w
    mx2 = ((x2 - 1) * wm) // frame_w + 1

    scale_y = frame_h / hm
    scale_x = frame_w / wm
    dilation_px = max(3, round(frame_h * BLEND_DILATION_RATIO))
    falloff_px = max(3, round(frame_h * BLEND_FALLOFF_RATIO))
    reach_y = max(1, round(dilation_px / scale_y)) + max(1, round(falloff_px / scale_y)) + 1
    reach_x = max(1, round(dilation_px / scale_x)) + max(1, round(falloff_px / scale_x)) + 1
    ey1 = max(0, my1 - reach_y)
    ey2 = min(hm, my2 + reach_y)
    ex1 = max(0, mx1 - reach_x)
    ex2 = min(wm, mx2 + reach_x)

    blend_ext = create_blend_mask(mask_lr[ey1:ey2, ex1:ex2], frame_h, scale_y, scale_x)
    blend_lr = blend_ext[my1 - ey1:my2 - ey1, mx1 - ex1:mx2 - ex1]
    return F.interpolate(
        blend_lr.unsqueeze(0).unsqueeze(0),
        size=out_size if out_size is not None else (y2 - y1, x2 - x1),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
