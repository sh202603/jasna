from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

from jasna.trt._backend import TRT_FLAVOR, trt

_TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def get_trt_logger() -> trt.ILogger:
    torchtrt = sys.modules.get("torch_tensorrt")
    if torchtrt is not None:
        return torchtrt.logging.TRT_LOGGER
    return _TRT_LOGGER


def _engine_io_names(engine: trt.ICudaEngine) -> tuple[list[str], list[str]]:
    input_names: list[str] = []
    output_names: list[str] = []

    if hasattr(engine, "num_io_tensors"):
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)
        return input_names, output_names

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        if engine.binding_is_input(i):
            input_names.append(name)
        else:
            output_names.append(name)
    return input_names, output_names


def _trt_dtype_to_torch(trt_dtype: trt.DataType) -> torch.dtype:
    if trt_dtype == trt.DataType.FLOAT:
        return torch.float32
    if trt_dtype == trt.DataType.HALF:
        return torch.float16
    if trt_dtype == trt.DataType.INT8:
        return torch.int8
    if trt_dtype == trt.DataType.INT32:
        return torch.int32
    if trt_dtype == trt.DataType.BOOL:
        return torch.bool
    raise ValueError(f"Unsupported TensorRT dtype: {trt_dtype}")


from jasna.engine_paths import get_onnx_tensorrt_engine_path  # noqa: E402


def _convert_onnx_bytes_to_fp16(onnx_bytes: bytes) -> bytes:
    """Convert an fp32 ONNX graph fully to fp16 (weights, activations, I/O).

    TensorRT-RTX builds strongly typed: precision follows the graph's tensor
    dtypes, and an fp32 graph compiles to an fp32 engine (~5x slower for
    RF-DETR than the standard-TRT fp16 build). Converting the whole graph is
    the equivalent of the BuilderFlag.FP16 + I/O-HALF-forcing recipe used with
    standard TensorRT.

    Pre-existing ``Cast(to=float32)`` nodes (e.g. DINOv2's embedding cast) are
    rewritten to float16 — the converter leaves their targets untouched, which
    would otherwise mix fp32 activations into the fp16 graph and fail the
    strongly-typed parse (Conv input Float vs kernel Half).
    """
    import warnings

    import onnx
    from onnx import TensorProto
    from onnxconverter_common import float16

    model = onnx.load_from_string(onnx_bytes)
    with warnings.catch_warnings():
        # The converter warns per truncated denormal constant; thousands of
        # lines of noise for transformer models.
        warnings.simplefilter("ignore")
        model = float16.convert_float_to_float16(model, keep_io_types=False, op_block_list=[])
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16
    return model.SerializeToString()


def _build_serialized_engine(
    onnx_bytes: bytes,
    device: torch.device,
    *,
    batch_size: int | None,
    fp16: bool,
    optimization_level: int,
    workspace_gb: int,
    dynamic_batch: bool = False,
) -> bytes:
    # Strongly-typed RTX: fp16 must be expressed in the graph itself (the
    # BuilderFlag.FP16 recipe below is standard-TRT only). After conversion
    # every float tensor is already HALF, so the I/O forcing loops below
    # become no-ops under RTX.
    if fp16 and TRT_FLAVOR == "rtx":
        onnx_bytes = _convert_onnx_bytes_to_fp16(onnx_bytes)

    logger = get_trt_logger()
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_bytes):
        errors = [parser.get_error(i) for i in range(parser.num_errors)]
        raise ValueError("TensorRT ONNX parse failed:\n" + "\n".join(str(e) for e in errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb) * 1024**3)
    if not (0 <= int(optimization_level) <= 5):
        raise ValueError(f"optimization_level must be in [0, 5], got {optimization_level}")
    config.builder_optimization_level = int(optimization_level)

    # TensorRT-RTX builds strongly typed and rejects BuilderFlag.FP16 with
    # "Error Code 3: API Usage Error"; precision follows the network tensor
    # dtypes instead. The I/O HALF forcing below is what makes the RTX build
    # effectively FP16, so it stays unconditional.
    if fp16 and TRT_FLAVOR != "rtx":
        config.set_flag(trt.BuilderFlag.FP16)

    needs_profile = False
    has_dynamic_batch_input = False
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        t = network.get_input(i)
        shape = tuple(int(d) for d in t.shape)
        dynamic_axes = [axis for axis, dimension in enumerate(shape) if dimension < 0]
        if any(axis != 0 for axis in dynamic_axes):
            raise ValueError(
                f"ONNX input '{t.name}' has unsupported non-batch dynamic shape {shape}"
            )
        if dynamic_axes:
            if batch_size is None:
                raise ValueError(
                    f"ONNX input '{t.name}' has dynamic shape {shape}; "
                    "provide batch_size to fix dynamic dimensions."
                )
            opt = (int(batch_size), *shape[1:])
            min_shape = (1, *shape[1:]) if dynamic_batch else opt
            profile.set_shape(t.name, min=min_shape, opt=opt, max=opt)
            needs_profile = True
            has_dynamic_batch_input = True
        elif dynamic_batch:
            raise ValueError(
                "dynamic_batch=True requires an ONNX input with a dynamic batch axis"
            )
        elif (
            batch_size is not None
            and shape
            and shape[0] != int(batch_size)
        ):
            raise ValueError(
                f"ONNX input '{t.name}' has fixed batch {shape[0]}, "
                f"requested batch_size={batch_size}"
            )
        if fp16 and t.dtype == trt.DataType.FLOAT:
            t.dtype = trt.DataType.HALF
    if dynamic_batch and not has_dynamic_batch_input:
        raise ValueError(
            "dynamic_batch=True requires an ONNX input with a dynamic batch axis"
        )
    if needs_profile:
        config.add_optimization_profile(profile)

    if fp16:
        for i in range(network.num_outputs):
            t = network.get_output(i)
            if t.dtype == trt.DataType.FLOAT:
                t.dtype = trt.DataType.HALF

    with torch.cuda.device(device):
        engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise ValueError("TensorRT engine build returned None")
    return bytes(engine_bytes)


def compile_onnx_to_tensorrt_engine(
    onnx_path: str | Path,
    device: torch.device,
    *,
    batch_size: int | None = None,
    fp16: bool = True,
    optimization_level: int = 5,
    workspace_gb: int,
    dynamic_batch: bool = False,
) -> Path:
    onnx_path = Path(onnx_path)
    engine_path = get_onnx_tensorrt_engine_path(
        onnx_path, batch_size=batch_size, fp16=bool(fp16), dynamic_batch=dynamic_batch,
    )

    if engine_path.exists():
        return engine_path

    if not onnx_path.exists():
        raise FileNotFoundError(str(onnx_path))
    log = logging.getLogger(__name__)
    msg = (
        f"Compiling TensorRT engine for {onnx_path} (this can take a few minutes). "
        f"Output: {engine_path}"
    )
    print(msg)
    log.info("%s", msg)

    engine_bytes = _build_serialized_engine(
        onnx_path.read_bytes(),
        device,
        batch_size=batch_size,
        fp16=bool(fp16),
        optimization_level=optimization_level,
        workspace_gb=workspace_gb,
        dynamic_batch=dynamic_batch,
    )

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(engine_bytes)
    return engine_path


def compile_onnx_bytes_to_encrypted_engine(
    onnx_bytes: bytes,
    model_id: str,
    engine_path: Path,
    device: torch.device,
    *,
    batch_size: int | None = None,
    fp16: bool = True,
    optimization_level: int = 5,
    workspace_gb: int,
) -> Path:
    from jasna.protection import protected_model

    if engine_path.exists():
        return engine_path

    log = logging.getLogger(__name__)
    msg = f"Compiling encrypted TensorRT engine for '{model_id}'. Output: {engine_path}"
    print(msg)
    log.info("%s", msg)

    engine_bytes = _build_serialized_engine(
        onnx_bytes,
        device,
        batch_size=batch_size,
        fp16=bool(fp16),
        optimization_level=optimization_level,
        workspace_gb=workspace_gb,
    )
    encrypted = protected_model.encrypt_engine_bytes(model_id, engine_bytes)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(encrypted)
    return engine_path
