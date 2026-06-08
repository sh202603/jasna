import torch
import torch.nn.functional as F

# Limited-range RGB→YUV conversion for NVENC input surfaces.
#
# Y'CbCr coefficients are derived from each colorspace's (Kr, Kb) luma weights:
#   Y  =        Kr*R + Kg*G + Kb*B          (Kg = 1 - Kr - Kb)
#   Cb = (B - Y) / (2*(1 - Kb))
#   Cr = (R - Y) / (2*(1 - Kr))
# fused with the limited-range scale/offset of the target bit depth.
_LUMA_WEIGHTS: dict[str, tuple[float, float]] = {
    # colorspace: (Kr, Kb)
    "bt601": (0.299, 0.114),
    "bt709": (0.2126, 0.0722),
    "bt2020": (0.2627, 0.0593),
}

# Per-bit-depth limited-range parameters (8-bit NV12 / 10-bit P010).
#   y_scale/y_off  : Y'  = y_off + y_scale * Y           (clamped y_min..y_max)
#   c_scale/c_off  : Cb' = c_off + c_scale * Cb          (clamped c_min..c_max)
#   shift          : left-shift to pack into the surface word (P010 packs 10 bits high)
#   dtype          : surface element type
_DEPTH_PARAMS = {
    8: dict(y_scale=219.0, y_off=16.0, c_scale=224.0, c_off=128.0,
            y_min=16, y_max=235, c_min=16, c_max=240, shift=0, dtype=torch.uint8),
    10: dict(y_scale=876.0, y_off=64.0, c_scale=896.0, c_off=512.0,
             y_min=64, y_max=940, c_min=64, c_max=960, shift=6, dtype=torch.int16),
}

# Cache fused (matrix, offset) tensors per (colorspace, bit_depth, device).
_cache: dict[tuple[str, int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}


def _build_matrix(colorspace: str, bit_depth: int) -> tuple[torch.Tensor, torch.Tensor]:
    kr, kb = _LUMA_WEIGHTS[colorspace]
    kg = 1.0 - kr - kb
    p = _DEPTH_PARAMS[bit_depth]
    ys, cs = p["y_scale"], p["c_scale"]

    cb_r = -kr / (2.0 * (1.0 - kb))
    cb_g = -kg / (2.0 * (1.0 - kb))
    cr_g = -kg / (2.0 * (1.0 - kr))
    cr_b = -kb / (2.0 * (1.0 - kr))

    mat = torch.tensor([
        [ys * kr,   ys * kg,   ys * kb],
        [cs * cb_r, cs * cb_g, cs * 0.5],
        [cs * 0.5,  cs * cr_g, cs * cr_b],
    ], dtype=torch.float32)
    off = torch.tensor([p["y_off"], p["c_off"], p["c_off"]], dtype=torch.float32)
    return mat, off


def _get_coeffs(colorspace: str, bit_depth: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (colorspace, bit_depth, device)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    mat, off = _build_matrix(colorspace, bit_depth)
    mat = mat.to(device=device)
    off = off.to(device=device)
    _cache[key] = (mat, off)
    return mat, off


def chw_rgb_to_surface(img_chw: torch.Tensor, colorspace: str = "bt709", bit_depth: int = 10) -> torch.Tensor:
    """Convert a CHW RGB frame to an NVENC input surface (NV12 for 8-bit, P010 for 10-bit).

    Returns a contiguous (H + H/2, W) tensor: Y plane followed by interleaved UV at 4:2:0.
    dtype is uint8 for NV12 and int16 (10 bits packed high) for P010. All work stays on the
    input tensor's device.
    """
    colorspace = str(colorspace)
    if colorspace not in _LUMA_WEIGHTS:
        raise ValueError(f"Unsupported colorspace: {colorspace}")
    if bit_depth not in _DEPTH_PARAMS:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    C, H, W = img_chw.shape
    p = _DEPTH_PARAMS[bit_depth]

    if img_chw.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        rgb = img_chw.float().div_(255.0)
    else:
        rgb = img_chw.float()

    mat, off = _get_coeffs(colorspace, bit_depth, rgb.device)

    # (3, H*W) matmul → (3, H, W): produces Y, U, V planes
    yuv = mat.mm(rgb.reshape(3, -1)).reshape(3, H, W)
    yuv[0].add_(off[0])
    yuv[1].add_(off[1])
    yuv[2].add_(off[2])

    shift = p["shift"]
    dtype = p["dtype"]

    # Y plane: clamp to limited range, pack into the surface word
    Y = yuv[0].round_().clamp_(p["y_min"], p["y_max"])
    if shift:
        Y = Y.mul_(1 << shift)
    Y = Y.to(dtype)

    # UV planes: subsample 4:2:0 via avg_pool2d on both channels at once
    uv_full = yuv[1:3].unsqueeze(0)  # (1, 2, H, W)
    uv_ds = F.avg_pool2d(uv_full, 2).squeeze(0)  # (2, H/2, W/2)
    uv_ds.round_().clamp_(p["c_min"], p["c_max"])
    if shift:
        uv_ds.mul_(1 << shift)
    uv_i = uv_ds.to(dtype)

    # Interleave U and V: (H/2, W) with alternating U, V
    uv_interleaved = uv_i.permute(1, 2, 0).reshape(H // 2, W)

    return torch.cat([Y, uv_interleaved], dim=0).contiguous()


def chw_rgb_to_p010_bt709_limited(img_chw: torch.Tensor) -> torch.Tensor:
    """Backward-compatible wrapper: BT.709 limited-range 10-bit P010."""
    return chw_rgb_to_surface(img_chw, colorspace="bt709", bit_depth=10)


def chw_rgb_to_p010_bt601_limited(img_chw: torch.Tensor) -> torch.Tensor:
    """Backward-compatible wrapper: BT.601 limited-range 10-bit P010."""
    return chw_rgb_to_surface(img_chw, colorspace="bt601", bit_depth=10)
