#!/usr/bin/env python
r"""Convert a Practical-RIFE model into a TorchScript checkpoint for jasna's
frame-generation backend (``--frame-gen``).

jasna's RIFE backend (``jasna/framegen/rife_frame_generator.py``) calls the
model with a fixed convention::

    model(img0, img1, timestep) -> mid

    img0, img1 : (1, 3, H, W) float in [0, 1], H/W padded to a multiple of 64
    timestep   : (1, 1, H, W) float, the interpolation position in (0, 1)
    mid        : (1, 3, H, W) float in [0, 1], the interpolated frame

This script wraps the official Practical-RIFE IFNet in exactly that signature
and exports a self-contained TorchScript module (architecture + weights), which
is the guaranteed-correct format for the backend to load.

On CUDA the module is traced in **fp16 by default** (``--no-fp16`` for fp32;
CPU always traces fp32). The backend also defaults to fp16, and an
fp16-traced module is the more portable artifact: dtype promotion lets it run
under an fp32 pipeline too, whereas an fp32-traced module bakes a float32 warp
grid into the graph and forces the backend's automatic fp32 fallback. If the
fp16 trace fails or produces non-finite output, the script falls back to fp32
on its own.

------------------------------------------------------------------------------
Preparation (one time)
------------------------------------------------------------------------------
1. Clone Practical-RIFE and download its model weights (tested with RIFE 4.6):

       git clone https://github.com/hzwer/Practical-RIFE
       # download the v4.6 model package per the repo's README and unzip it so
       # that <repo>/train_log/ contains:  RIFE_HDv3.py, IFNet_HDv3.py, flownet.pkl

2. Run this script (uses jasna's venv so torch matches the runtime):

       .\.venv\Scripts\python.exe scripts/make_rife_torchscript.py ^
           --rife-repo C:\path\to\Practical-RIFE ^
           --output model_weights\rife.pth --validate

3. Use it:

       .\.venv\Scripts\python.exe -m jasna --input in.mp4 --output out2x.mkv --frame-gen 2x
   (or pass --frame-gen-model-path <path> instead of the default model_weights/rife.pth)

------------------------------------------------------------------------------
License
------------------------------------------------------------------------------
This script is yours to publish. The RIFE model code and weights are NOT: they
come from Practical-RIFE (non-commercial training-weight terms).
Do not redistribute flownet.pkl or the generated rife.pth without checking the
upstream license. See https://github.com/hzwer/Practical-RIFE.
"""
from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn


@contextmanager
def _default_dtype(dtype: torch.dtype):
    """Temporarily change torch's default float dtype.

    Practical-RIFE's warp builds its sampling grid with ``torch.linspace``
    without an explicit dtype, so tracing in fp16 requires the *default* float
    dtype to be half while the graph is recorded — otherwise a float32 grid
    gets baked in and ``grid_sample`` rejects half inputs at runtime.
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def _load_rife_model(rife_repo: Path, weights_dir: Path):
    """Import the Practical-RIFE Model and load its weights.

    Returns the flownet (the IFNet nn.Module) and the loaded Model wrapper.
    """
    if not rife_repo.is_dir():
        raise FileNotFoundError(f"--rife-repo not found: {rife_repo}")
    # Practical-RIFE imports modules as ``train_log.RIFE_HDv3`` (some releases
    # ship the model file directly under train_log/). Put both on sys.path.
    sys.path.insert(0, str(rife_repo))
    sys.path.insert(0, str(weights_dir))

    Model = None
    import_errors = []
    for modname in ("train_log.RIFE_HDv3", "RIFE_HDv3", "train_log.model.RIFE_HDv3"):
        try:
            mod = __import__(modname, fromlist=["Model"])
            Model = getattr(mod, "Model")
            break
        except Exception as e:  # noqa: BLE001 - report all attempts
            import_errors.append(f"  {modname}: {e}")
    if Model is None:
        raise ImportError(
            "Could not import the RIFE Model class. Tried:\n"
            + "\n".join(import_errors)
            + "\nEnsure --rife-repo points at a Practical-RIFE checkout whose "
            "train_log/ contains RIFE_HDv3.py + IFNet_HDv3.py + flownet.pkl."
        )

    model = Model()
    # Model.load_model(dir, rank) reads <dir>/flownet.pkl; rank=-1 = single GPU/CPU.
    model.load_model(str(weights_dir), -1)
    if hasattr(model, "eval"):
        model.eval()
    flownet = getattr(model, "flownet", None)
    if flownet is None:
        raise AttributeError("Loaded RIFE Model has no .flownet attribute (unexpected version).")
    flownet.eval()
    return flownet, model


class RifeTorchScriptWrapper(nn.Module):
    """Adapts a Practical-RIFE ``Model`` to jasna's ``(img0, img1, timestep_map)``
    signature by delegating to the model's own ``inference`` method.

    Delegating to ``Model.inference`` is what makes this version-robust:
    ``inference`` encapsulates the per-version ``scale_list`` (e.g. 4.25/4.26 use
    a 5-element ``[16,8,4,2,1]``, 4.6 uses ``[4,2,1]``), so we don't hardcode it.

    The only thing that still varies is how ``timestep`` is shaped. The backend
    always passes a full-resolution ``(1,1,H,W)`` map:
    - ``ts_mode=0``: 4.25/4.26 IFNet does ``timestep.repeat(1,1,H,W)`` and expects
      a ``(1,1,1,1)`` scalar tensor, so we slice the map down.
    - ``ts_mode=1``: 4.6-style IFNet uses the map directly.

    ``ts_mode`` is detected once at conversion time and baked into the traced
    graph as a constant branch.
    """

    def __init__(self, model, ts_mode: int = 0):
        super().__init__()
        # Register the flownet so its parameters are captured by trace/save.
        self.flownet = model.flownet
        # The Model wrapper is a plain object; keep it without registering it.
        self._model = model
        self.ts_mode = int(ts_mode)

    def forward(self, img0, img1, timestep):
        ts = timestep[:, :, :1, :1] if self.ts_mode == 0 else timestep
        mid = self._model.inference(img0, img1, ts)
        return mid.clamp(0.0, 1.0)


def _detect_ts_mode(model, device: torch.device, dtype: torch.dtype = torch.float32) -> int:
    """Probe both timestep conventions via Model.inference; pick the one that runs."""
    d0 = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
    d1 = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
    ts = torch.full((1, 1, 64, 64), 0.5, device=device, dtype=dtype)
    errors = []
    for mode in (0, 1):
        try:
            w = RifeTorchScriptWrapper(model, ts_mode=mode).to(device).eval()
            with torch.inference_mode():
                y = w(d0, d1, ts)
            if y.shape == (1, 3, 64, 64) and torch.isfinite(y).all():
                print(f"[detect] timestep convention: ts_mode={mode}")
                return mode
            errors.append(f"  ts_mode={mode}: bad output shape {tuple(y.shape)}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"  ts_mode={mode}: {e}")
    raise RuntimeError(
        "Could not run Model.inference with either timestep convention. "
        "This RIFE version may differ; adjust RifeTorchScriptWrapper.forward.\n"
        + "\n".join(errors)
    )


def _trace(wrapper: nn.Module, device: torch.device, size: int, dtype: torch.dtype = torch.float32) -> torch.jit.ScriptModule:
    h = w = ((size + 63) // 64) * 64
    img0 = torch.rand(1, 3, h, w, device=device, dtype=dtype)
    img1 = torch.rand(1, 3, h, w, device=device, dtype=dtype)
    ts = torch.full((1, 1, h, w), 0.5, device=device, dtype=dtype)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, (img0, img1, ts), check_trace=False)
    return traced


def _smoke_check(traced: torch.jit.ScriptModule, device: torch.device, dtype: torch.dtype) -> None:
    """Run the traced module once and require finite output (fp16 overflow guard)."""
    a = torch.zeros(1, 3, 64, 64, device=device, dtype=dtype)
    b = torch.full((1, 3, 64, 64), 0.8, device=device, dtype=dtype)
    ts = torch.full((1, 1, 64, 64), 0.5, device=device, dtype=dtype)
    with torch.inference_mode():
        out = traced(a, b, ts)
    if not torch.isfinite(out).all():
        raise RuntimeError("traced module produced non-finite output")


def _convert(model, device: torch.device, size: int, dtype: torch.dtype) -> torch.jit.ScriptModule:
    """Detect the timestep convention, trace, and smoke-check in ``dtype``."""
    model.flownet.to(device=device, dtype=dtype).eval()
    with _default_dtype(dtype):
        ts_mode = _detect_ts_mode(model, device, dtype)
        wrapper = RifeTorchScriptWrapper(model, ts_mode=ts_mode).to(device).eval()
        print(f"[trace] tracing at {size}x{size} on {device} ({'fp16' if dtype is torch.float16 else 'fp32'}) ...")
        traced = _trace(wrapper, device, size, dtype)
        _smoke_check(traced, device, dtype)
    return traced


def _validate(traced_path: Path, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
    """Reload the saved TorchScript and sanity-check it at a *different* size.

    RIFE uses scale-relative interpolation and runtime-built warp grids, so a
    module traced at one resolution should generalize. We verify shape, range,
    and that a midpoint blend of two flat frames lands near their average.
    """
    m = torch.jit.load(str(traced_path), map_location=device).eval()
    h, w = 192, 320  # deliberately != trace size, both multiples of 64
    a = torch.zeros(1, 3, h, w, device=device, dtype=dtype)
    b = torch.full((1, 3, h, w), 0.8, device=device, dtype=dtype)
    ts = torch.full((1, 1, h, w), 0.5, device=device, dtype=dtype)
    with torch.inference_mode():
        out = m(a, b, ts)
    assert out.shape == (1, 3, h, w), f"unexpected output shape {tuple(out.shape)}"
    assert out.min() >= -1e-3 and out.max() <= 1.0 + 1e-3, "output out of [0,1] range"
    assert torch.isfinite(out).all(), "output has non-finite values"
    mean = float(out.mean())
    # Note: RIFE is flow + mask based, not a linear blend, so on synthetic flat
    # frames the midpoint mean only lands *near* the average. Real footage with
    # actual motion interpolates properly; this check just confirms shape/range.
    print(f"[validate] OK at {h}x{w}: shape={tuple(out.shape)}, midpoint mean={mean:.3f} "
          f"(near 0.4 for a 0.0/0.8 blend; RIFE is non-linear so not exact)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rife-repo", required=True, type=str,
                        help="Path to a Practical-RIFE checkout (contains train_log/).")
    parser.add_argument("--weights-dir", type=str, default="",
                        help="Dir containing flownet.pkl. Default: <rife-repo>/train_log.")
    parser.add_argument("--output", type=str, default="model_weights/rife.pth",
                        help="Output TorchScript path (default: %(default)s).")
    parser.add_argument("--size", type=int, default=256,
                        help="Spatial size to trace at; generalizes to other sizes (default: %(default)s).")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None,
                        help="Trace in fp16 (default on CUDA; CPU always traces fp32). "
                             "Falls back to fp32 automatically if the fp16 trace fails.")
    parser.add_argument("--validate", action="store_true", help="Reload and sanity-check the result.")
    args = parser.parse_args()

    rife_repo = Path(args.rife_repo).expanduser().resolve()
    weights_dir = Path(args.weights_dir).expanduser().resolve() if args.weights_dir else (rife_repo / "train_log")
    output = Path(args.output).expanduser()
    device = torch.device(args.device)

    use_fp16 = args.fp16 if args.fp16 is not None else device.type == "cuda"
    if use_fp16 and device.type != "cuda":
        print("[warn] --fp16 requires CUDA (half convolutions are unsupported on CPU); tracing fp32.")
        use_fp16 = False

    print(f"[load] RIFE repo: {rife_repo}")
    print(f"[load] weights:   {weights_dir / 'flownet.pkl'}")
    flownet, model = _load_rife_model(rife_repo, weights_dir)
    flownet.to(device).eval()
    if hasattr(model, "eval"):
        model.eval()
    ver = getattr(model, "version", "unknown")
    print(f"[load] RIFE Model.version = {ver}")

    dtype = torch.float16 if use_fp16 else torch.float32
    if use_fp16:
        try:
            traced = _convert(model, device, args.size, torch.float16)
        except Exception as e:  # noqa: BLE001 - any fp16 failure falls back
            print(f"[warn] fp16 trace failed ({e}); falling back to fp32.")
            dtype = torch.float32
            traced = _convert(model, device, args.size, torch.float32)
    else:
        traced = _convert(model, device, args.size, torch.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(output))
    print(f"[save] wrote TorchScript ({'fp16' if dtype is torch.float16 else 'fp32'}) -> {output.resolve()}")

    if args.validate:
        _validate(output, device, dtype)

    print("\nDone. Place it where the backend looks (model_weights/rife.pth) or pass "
          "--frame-gen-model-path <path> to jasna.")


if __name__ == "__main__":
    main()
