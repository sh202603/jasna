"""cuDNN FP8 upsample backend for BasicVSR++ (experimental, opt-in).

Drop-in replacement for the TRT FP16 "upsample" sub-engine, enabled with
JASNA_FP8_RECON=1 (or --fp8-recon). Runs the 16-conv upsample chain (entry +
residual blocks + 2 pixel-shuffle upsamples + conv_hr + conv_last) as cuDNN
graph-API FP8 convolutions with fused epilogues, keeping the whole chain in
FP8/NHWC.

Why (measured on a Blackwell / sm120 GPU):
  - cuDNN >=9.17 ships Blackwell-native (sm120) FP8 conv kernels that TensorRT
    lacks: 2.1-3.0x vs best FP16 on all upsample-chain shapes.
  - End-to-end this stage runs ~1.6x faster than the TRT FP16 engine at
    PSNR 67dB / SSIM 0.99996 vs the FP32 reference (per-tensor PTQ; the FP16
    engine's own floor is 79dB).
  - The TRT upsample engine pre-allocates a multi-GB internal arena at load
    (dynamic profile sized by --max-clip-size); skipping that load frees the
    VRAM for the secondary restorer and larger queues.

Hardware: needs an FP8-capable GPU (sm89+, i.e. RTX 40 series / Hopper or
newer). The speedup is only validated on sm120 — on sm89/sm90 the cuDNN FP8
kernels exist but the gain vs the FP16 TRT engine is unverified. On unsupported
hardware the constructor raises and the caller falls back to the TRT engine.

Quantization: FP8 e4m3 is floating point, so per-tensor scales only need to
keep values inside the representable range. Activations here have amax of a
few units (<<448) and use scale 1.0; weights are amax-scaled per tensor (folded
into the conv epilogue) to make use of the subnormal range. No calibration
file is needed. Positive-homogeneous activations (ReLU/LeakyReLU) commute with
positive scaling, so each conv needs only bias -> act -> scale as epilogue.

Ported from lada-ex ``lada/models/basicvsrpp/fp8_recon.py`` (AGPL-3.0); jasna
carries the residual (+ lq) addition outside the engine, so unlike the lada
variant ``forward`` takes only the feature batch and returns the pre-residual
output.
"""
import glob
import logging
import os
import sys

import torch
from torch import nn

logger = logging.getLogger(__name__)

if sys.platform == "win32":
    # cudnn-frontend's cudart shim only probes Linux names (libcudart.so.12/13),
    # so without help it always raises on Windows and we'd fall back to TRT.
    # Point it at the cudart DLL bundled with the torch wheel; torch has already
    # loaded it into the process by now, so the bare DLL name always resolves.
    _cudart_dlls = glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "cudart64_*.dll"))
    if _cudart_dlls:
        os.environ.setdefault("CUDNN_FRONTEND_CUDART_LIB_NAME", os.path.basename(_cudart_dlls[0]))
else:
    # cudnn-frontend's loader scans os.environ["LD_LIBRARY_PATH"] first and
    # CDLLs the first libcudnn.so.9 it finds there. nvvfx prepends its libs dir
    # (holding a bare 9.7 dispatcher without its libcudnn_graph.so.*
    # sub-libraries) to that variable, which would make the frontend pick the
    # crippled copy and abort on cudnnCreate. Put torch's complete cuDNN
    # (>=9.17) in front so the frontend resolves to it regardless of whether
    # nvvfx (rtx-super-res) was constructed first.
    import importlib.util

    try:
        _cudnn_spec = importlib.util.find_spec("nvidia.cudnn")
        if _cudnn_spec is not None and _cudnn_spec.submodule_search_locations:
            _cudnn_dir = os.path.join(list(_cudnn_spec.submodule_search_locations)[0], "lib")
            if os.path.isfile(os.path.join(_cudnn_dir, "libcudnn.so.9")):
                _ld = os.environ.get("LD_LIBRARY_PATH", "")
                if _cudnn_dir not in _ld.split(os.pathsep):
                    os.environ["LD_LIBRARY_PATH"] = (
                        _cudnn_dir + (os.pathsep + _ld if _ld else "")
                    )
    except Exception:  # best-effort; the constructor fails loudly if cudnn is unusable
        logger.debug("Could not front-load torch's cudnn dir", exc_info=True)

FP8_MAX = 448.0
_BUCKET = 10
_TILE = 10   # T-tile for the upsample tail (frame-independent -> numerically exact)
_TAIL = ("ups1", "ups2", "conv_hr", "conv_last")
_FEAT = 64   # feature grid size; tail doubles it twice (64 -> 128 -> 256)


def _round_up_bucket(t: int) -> int:
    return -(-t // _BUCKET) * _BUCKET


def _cl(t):
    return t.contiguous(memory_format=torch.channels_last)


def _cast_cl_fp8_impl(x):
    return x.to(dtype=torch.float8_e4m3fn, memory_format=torch.channels_last)


def _ps_int8_impl(b, r: int):
    # NHWC-order pixel shuffle on int8 storage; torch's native pixel_shuffle
    # hits a slow gather path on channels-last 1-byte tensors (~4x off floor).
    t, cin, hh, ww = b.shape
    c = cin // (r * r)
    nhwc = b.permute(0, 2, 3, 1)
    y = (nhwc.reshape(t, hh, ww, c, r, r)
         .permute(0, 1, 4, 2, 5, 3)
         .reshape(t, hh * r, ww * r, c))
    return y.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def _compile_or_eager(fn, **compile_kwargs):
    """torch.compile with permanent eager fallback on the first failure.

    Windows inductor (via the triton-windows wheel) is not validated on every
    setup; a compile failure must degrade the glue to eager instead of raising
    inside the restore worker thread. Warmup exercises both glue paths, so a
    broken inductor surfaces at load time as a one-shot warning."""
    compiled = torch.compile(fn, **compile_kwargs)
    state = {"compiled": True}

    def wrapper(*args):
        if state["compiled"]:
            try:
                return compiled(*args)
            except Exception:
                logger.warning(
                    "FP8 glue torch.compile failed; falling back to eager", exc_info=True,
                )
                state["compiled"] = False
        return fn(*args)

    return wrapper


if os.environ.get("JASNA_FP8_RECON_NOCOMPILE") == "1" or not _has_triton():
    # inductor needs triton: pytorch-triton comes with torch on Linux, Windows
    # uses the triton-windows wheel (a pyproject dependency). Without it the
    # glue ops run eager — slower but functional.
    _cast_cl_fp8, _ps_int8 = _cast_cl_fp8_impl, _ps_int8_impl
else:
    # inductor reaches the memory floor on both glue ops (1.02->0.28ms, 1.06->0.63ms).
    # cast is dynamic over T: one compile covers every bucket shape (measured loss
    # vs static: none), so warmup doesn't pay 9 shape-specialized compiles.
    # ps runs on fixed _TILE-sized tail shapes (2 variants) — static is fine.
    _cast_cl_fp8 = _compile_or_eager(_cast_cl_fp8_impl, mode="max-autotune-no-cudagraphs", dynamic=True)
    _ps_int8 = _compile_or_eager(_ps_int8_impl, mode="max-autotune-no-cudagraphs", dynamic=False)


def _pixel_shuffle_fp8(x, r=2):
    return _ps_int8(x.view(torch.int8), r).view(torch.float8_e4m3fn)


class _ConvGraph:
    """One FP8 conv + fused epilogue as a cuDNN graph.

    kind 'act'  : bias -> relu/lrelu(0.1) -> mul(alpha) -> FP8 out
    kind 'res'  : mul(alpha) -> bias -> add identity -> FP8 out (no act)
    kind 'final': mul(alpha) -> bias -> HALF out
    """

    def __init__(self, handle, t, cin, cout, h, w, kind, act, weight, bias, device):
        import cudnn
        self.handle = handle
        self.kind = kind
        s_w = max(weight.float().abs().max().item(), 1e-8) / FP8_MAX
        alpha = s_w  # activation scales are 1.0 (see module docstring)
        self.w_fp8 = _cl((weight.float() / s_w).to(torch.float8_e4m3fn))

        E4M3, HALF, FLOAT = cudnn.data_type.FP8_E4M3, cudnn.data_type.HALF, cudnn.data_type.FLOAT
        g = cudnn.pygraph(io_data_type=E4M3, intermediate_data_type=FLOAT,
                          compute_data_type=FLOAT, handle=handle)
        x_proto = _cl(torch.empty(t, cin, h, w, dtype=torch.float8_e4m3fn, device=device))
        X = g.tensor_like(x_proto)
        W = g.tensor_like(self.w_fp8)
        C = g.conv_fprop(image=X, weight=W, padding=[1, 1], stride=[1, 1], dilation=[1, 1])

        io = {"X": X, "W": W}
        if kind == "act":
            self.b_t = (bias.float() / alpha).reshape(1, cout, 1, 1).to(device)
            B = g.tensor_like(self.b_t)
            zb = g.bias(input=C, bias=B)
            za = g.leaky_relu(input=zb, negative_slope=0.1) if act == "lrelu" else g.relu(input=zb)
            self.m_t = torch.full((1, 1, 1, 1), alpha, dtype=torch.float32, device=device)
            M = g.tensor_like(self.m_t)
            Y = g.mul(a=za, b=M)
            Y.set_output(True).set_data_type(E4M3)
            io.update({"B": B, "M": M})
        elif kind == "res":
            self.m_t = torch.full((1, 1, 1, 1), alpha, dtype=torch.float32, device=device)
            M = g.tensor_like(self.m_t)
            zs = g.mul(a=C, b=M)
            self.b_t = bias.float().reshape(1, cout, 1, 1).to(device)
            B = g.tensor_like(self.b_t)
            zb = g.bias(input=zs, bias=B)
            ID = g.tensor_like(x_proto)
            Y = g.add(a=zb, b=ID)
            Y.set_output(True).set_data_type(E4M3)
            io.update({"M": M, "B": B, "ID": ID})
        else:  # final
            self.m_t = torch.full((1, 1, 1, 1), alpha, dtype=torch.float32, device=device)
            M = g.tensor_like(self.m_t)
            zs = g.mul(a=C, b=M)
            self.b_t = bias.float().reshape(1, cout, 1, 1).to(device)
            B = g.tensor_like(self.b_t)
            Y = g.bias(input=zs, bias=B)
            Y.set_output(True).set_data_type(HALF)
            io.update({"M": M, "B": B})

        g.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        self.g, self.io, self.Y = g, io, Y
        self.ws_size = g.get_workspace_size()

    def run(self, x, out, ws, identity=None):
        feed = {self.io["X"]: x, self.io["W"]: self.w_fp8, self.Y: out,
                self.io["B"]: self.b_t, self.io["M"]: self.m_t}
        if self.kind == "res":
            feed[self.io["ID"]] = identity
        self.g.execute(feed, ws, handle=self.handle)
        return out


class CudnnFP8Upsample(nn.Module):
    """FP8 backend for the "upsample" sub-engine slot.

    forward(hr_batch (T, 5*mid, 64, 64) fp16) -> (T, 3, 256, 256) fp16, without
    the ``+ lq`` residual (the caller adds it, see
    ``BasicVSRPlusPlusNetSplit.upsample``). The return value is a view of a
    persistent output buffer that is overwritten by the next forward — the
    caller must consume it before calling again. T is bucketed to multiples of
    10 (graphs built lazily per bucket, buffers shared across buckets).

    nn.Module (with no parameters) so ``BasicVSRPlusPlusNetSplit.close()`` can
    hold it like the TRT engines; GPU resources are freed via ``release()``.
    """

    def __init__(self, generator: nn.Module, device: torch.device, max_clip_size: int = 60):
        super().__init__()
        cc = torch.cuda.get_device_capability(device)
        if cc < (8, 9):
            raise RuntimeError(f"FP8 convolutions need an FP8-capable GPU (sm89+), found sm{cc[0]}{cc[1]}")
        import cudnn
        self.device = torch.device(device)
        self._t_max = _round_up_bucket(int(max_clip_size))
        with torch.cuda.device(self.device):
            self.handle = cudnn.create_handle()
            cudnn.set_stream(handle=self.handle, stream=torch.cuda.current_stream(self.device).cuda_stream)

        entry = generator.reconstruction.main[0]
        blocks = list(generator.reconstruction.main[2])
        p = {"entry": (entry, "act", "lrelu", _FEAT)}
        for i, blk in enumerate(blocks):
            p[f"res{i}_1"] = (blk.conv1, "act", "relu", _FEAT)
            p[f"res{i}_2"] = (blk.conv2, "res", None, _FEAT)
        p["ups1"] = (generator.upsample1.upsample_conv, "act", "lrelu", _FEAT)
        p["ups2"] = (generator.upsample2.upsample_conv, "act", "lrelu", 2 * _FEAT)
        p["conv_hr"] = (generator.conv_hr, "act", "lrelu", 4 * _FEAT)
        p["conv_last"] = (generator.conv_last, "final", None, 4 * _FEAT)
        # cin/cout from the weights: works for any mid_channels
        self.spec = {k: (m.weight.data.detach(), m.bias.data.detach(), kind, act,
                         m.weight.shape[1], m.weight.shape[0], hw)
                     for k, (m, kind, act, hw) in p.items()}
        self._num_blocks = len(blocks)
        self._in_ch = entry.weight.shape[1]        # 5 * mid_channels
        self._mid = entry.weight.shape[0]          # mid_channels
        self._c_u1 = p["ups1"][0].weight.shape[0]  # mid * 4 (pre pixel-shuffle)
        self._c_u2 = p["ups2"][0].weight.shape[0]
        self._c_hr = p["conv_hr"][0].weight.shape[0]
        self._out_ch = p["conv_last"][0].weight.shape[0]
        self.graphs = {}      # bucket -> {name: _ConvGraph}
        self.bufs = None      # shared buffers sized to the largest bucket seen
        self.bufs_t = 0
        self.ws = None
        # Build the largest bucket eagerly, NOT inside warmup's try/except: if
        # cuDNN can't provide FP8 conv kernels here (old cuDNN, unsupported GPU),
        # fail now so the caller falls back to the TRT engine instead of dying
        # mid-pipeline in a worker thread.
        self._build_bucket(self._t_max)
        logger.info("CudnnFP8Upsample: enabled (cuDNN FP8 upsample backend, t_max=%d)", self._t_max)
        if os.environ.get("JASNA_FP8_RECON_NOWARM") != "1":
            self.warmup()

    def warmup(self):
        """Front-load every lazy cost (cuDNN graph builds, inductor compiles,
        allocator pool growth) into engine-load time.

        Without this the first clips reaching the restorer trigger builds and
        max-autotune benchmarking mid-pipeline; startup can absorb it instead."""
        import time
        t0 = time.time()
        try:
            # buffers at max size once; eager graphs for every bucket. Unlike
            # lada's fixed chunk lengths, jasna clip lengths vary freely over
            # [1, max_clip_size], so every bucket gets hit in practice — lazy
            # builds would land inside the restore stage (~0.1-0.2s each,
            # measured ~+1s per run across 6 buckets).
            self._ensure_bufs(self._t_max)
            for tb in range(_BUCKET, self._t_max + 1, _BUCKET):
                if tb not in self.graphs:
                    self._build_bucket(tb)
            with torch.inference_mode():
                h = torch.zeros(self._t_max, self._in_ch, _FEAT, _FEAT,
                                dtype=torch.float16, device=self.device)
                self(h)                  # bucket-exact path (compiled cast + both ps tiles)
                self(h[:_TILE // 2])     # pad path + remainder tile
            del h
            torch.cuda.synchronize(self.device)
            logger.info("CudnnFP8Upsample: warmup done in %.1fs (%d T-buckets)",
                        time.time() - t0, len(self.graphs))
        except Exception as e:  # warmup is best-effort; lazy paths still work
            logger.warning("CudnnFP8Upsample: warmup failed (%s); continuing with lazy builds", e)

    def _ensure_bufs(self, tb):
        if self.bufs is not None and self.bufs_t >= tb:
            return
        tt = _TILE
        s1, s2, s3 = _FEAT, 2 * _FEAT, 4 * _FEAT

        def mk(n, c, s):
            return _cl(torch.empty(n, c, s, s, dtype=torch.float8_e4m3fn, device=self.device))

        # The upsample tail (ups1..conv_last) is tiled over T (frames are
        # independent), so its big 128^2/256^2 buffers are _TILE-sized — this is
        # what keeps the module's footprint far below the arena of the TRT
        # engine it replaces.
        self.bufs = {
            "a64": mk(tb, self._mid, s1), "b64": mk(tb, self._mid, s1), "c64": mk(tb, self._mid, s1),
            # u1/u2 hold conv outputs *before* their pixel shuffle: 4c channels
            # at the pre-shuffle resolution (64^2 / 128^2)
            "u1": mk(tt, self._c_u1, s1), "u2": mk(tt, self._c_u2, s2), "hr": mk(tt, self._c_hr, s3),
            "out": _cl(torch.empty(tb, self._out_ch, s3, s3, dtype=torch.float16, device=self.device)),
            # FP8 pad buffer for non-bucket-exact T (padding tail may hold stale
            # frames — their outputs are sliced off, so no zeroing needed)
            "x_pad": mk(tb, self._in_ch, s1),
        }
        self.bufs_t = tb

    def _build_bucket(self, tb):
        gl = {}
        with torch.cuda.device(self.device):
            for name, (w, b, kind, act, cin, cout, hw) in self.spec.items():
                n = _TILE if name in _TAIL else tb
                if name in _TAIL and self.graphs:  # tail graphs are bucket-independent — reuse
                    gl[name] = next(iter(self.graphs.values()))[name]
                    continue
                gl[name] = _ConvGraph(self.handle, n, cin, cout, hw, hw, kind, act, w, b, self.device)
        ws_size = max(g.ws_size for g in gl.values())
        if self.ws is None or self.ws.numel() < ws_size:
            self.ws = torch.empty(max(ws_size, 8), dtype=torch.uint8, device=self.device)
        self.graphs[tb] = gl

    def forward(self, h_in):
        t = h_in.shape[0]
        tb = min(self._t_max, _round_up_bucket(t))
        if tb < t:  # t beyond the profile — never happens, guard anyway
            tb = t
        self._ensure_bufs(tb)
        if tb not in self.graphs:
            logger.info("CudnnFP8Upsample: building graphs for T-bucket %d...", tb)
            self._build_bucket(tb)
        gl, bf, ws = self.graphs[tb], self.bufs, self.ws

        if tb == t:
            x = _cast_cl_fp8(h_in)  # compiled cast — only bucket shapes
        else:
            # rare (non-bucket-exact clips): eager cast at arbitrary T to avoid
            # inductor recompiles, copied into the persistent FP8 pad buffer
            x = bf["x_pad"][:tb]
            x[:t].copy_(_cast_cl_fp8_impl(h_in))

        a64, b64, c64 = bf["a64"][:tb], bf["b64"][:tb], bf["c64"][:tb]
        cur = gl["entry"].run(x, a64, ws)
        for i in range(self._num_blocks):
            h1 = gl[f"res{i}_1"].run(cur, b64, ws)
            nxt = c64 if cur.data_ptr() == a64.data_ptr() else a64
            cur = gl[f"res{i}_2"].run(h1, nxt, ws, identity=cur)

        out = bf["out"][:tb]
        for i in range(0, tb, _TILE):
            u1 = gl["ups1"].run(cur[i:i + _TILE], bf["u1"], ws)
            u2 = gl["ups2"].run(_pixel_shuffle_fp8(u1), bf["u2"], ws)
            hr = gl["conv_hr"].run(_pixel_shuffle_fp8(u2), bf["hr"], ws)
            gl["conv_last"].run(hr, out[i:i + _TILE], ws)
        return out[:t]

    def release(self):
        """Idempotently free GPU resources (graphs, buffers, workspace, handle)."""
        self.graphs = {}
        self.bufs = None
        self.bufs_t = 0
        self.ws = None
        self.spec = None
        handle, self.handle = getattr(self, "handle", None), None
        if handle is not None:
            try:
                import cudnn
                cudnn.destroy_handle(handle)
            except Exception as e:
                logger.debug("CudnnFP8Upsample: destroy_handle failed (%s)", e)
