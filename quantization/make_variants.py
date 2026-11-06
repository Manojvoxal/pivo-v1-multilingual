"""Build reduced-precision variants of the PiVo-Multi ONNX graphs.

Variants (folder name -> what changes):
  fp32        original export (copied)
  fp16        weights stored as float16, Cast to float32 at load; all maths stays float32
  int8        weight-only int8, symmetric per output channel (DequantizeLinear, opset 17); maths float32
  int8dyn     onnxruntime quantize_dynamic: int8 weights AND int8 activations (MatMul/Gemm/Gather only;
              ConvInteger with int8 weights is missing in ORT 1.20 / ORT Web)
  int4        weight-only int4, symmetric, blocks of 32 along the input dim (DequantizeLinear opset 21)
  int4b16     same with blocks of 16 (more scales, less error)
  int4mix     int4 (blocks of 16) for MatMul / Gemm weights, int8 for conv kernels and embedding tables
usage: python make_variants.py models/multi_w128_refine/onnx OUT_DIR fp32,fp16,int8,int4

Only the weight matrices are reduced (MatMul / Gemm / Conv weights, embedding tables). Biases,
norm gains and small constants stay float32: they are a few KB in total.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, version_converter

SRC = Path(sys.argv[1])          # .../models/multi_w128_refine/onnx
OUT = Path(sys.argv[2])          # output root
GRAPHS = ["encoder", "decoder", "conditioning", "vocoder_step"]
MIN_ELEMS = 1024                 # smaller tensors are not worth quantizing
BLOCK = 32


def weight_uses(model):
    """initializer name -> (op_type, input_index, node) for tensors used as weights."""
    inits = {i.name: i for i in model.graph.initializer}
    uses = {}
    for nd in model.graph.node:
        for j, x in enumerate(nd.input):
            if x in inits:
                uses.setdefault(x, []).append((nd.op_type, j, nd))
    out = {}
    for name, us in uses.items():
        a = numpy_helper.to_array(inits[name])
        if a.dtype != np.float32 or a.size < MIN_ELEMS:
            continue
        kinds = {(op, j) for op, j, _ in us}
        if kinds <= {("MatMul", 1)}:
            out[name] = ("matmul", a)                 # (K, N), output channels on axis 1
        elif kinds <= {("Conv", 1)}:
            out[name] = ("conv", a)                   # (Cout, Cin, k)
        elif kinds <= {("Gemm", 1)}:
            tb = [at.i for at in us[0][2].attribute if at.name == "transB"]
            out[name] = ("gemm_t" if tb and tb[0] else "matmul", a)   # transB=1: (N, K)
        elif kinds <= {("Gather", 0)}:
            out[name] = ("embed", a)                  # (V, D), one row per token
    return out


def replace_init(model, name, new_inits, nodes):
    g = model.graph
    keep = [i for i in g.initializer if i.name != name]
    del g.initializer[:]
    g.initializer.extend(keep + new_inits)
    for n in reversed(nodes):
        g.node.insert(0, n)


def to_fp16(model):
    for name, (_, a) in weight_uses(model).items():
        replace_init(model, name, [numpy_helper.from_array(a.astype(np.float16), name + "_f16")],
                     [helper.make_node("Cast", [name + "_f16"], [name], to=TensorProto.FLOAT, name=name + "_cast")])
    return model


def per_channel_axis(kind):
    # axis that indexes output channels / rows (one scale each)
    return {"matmul": 1, "conv": 0, "gemm_t": 0, "embed": 0}[kind]


def int8_one(model, name, kind, a):
    ax = per_channel_axis(kind)
    red = tuple(i for i in range(a.ndim) if i != ax)
    scale = np.maximum(np.abs(a).max(axis=red), 1e-12) / 127.0
    shp = [1] * a.ndim
    shp[ax] = -1
    q = np.clip(np.rint(a / scale.reshape(shp)), -127, 127).astype(np.int8)
    replace_init(model, name, [numpy_helper.from_array(q, name + "_q"),
                               numpy_helper.from_array(scale.astype(np.float32), name + "_s"),
                               numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), name + "_z")],
                 [helper.make_node("DequantizeLinear", [name + "_q", name + "_s", name + "_z"], [name],
                                   axis=ax, name=name + "_dq")])


def to_int8(model):
    for name, (kind, a) in weight_uses(model).items():
        int8_one(model, name, kind, a)
    return model


def int4_tensor(name, q):
    """INT4 initializer as packed raw bytes (2 values per byte, low nibble first)."""
    flat = (q.flatten().astype(np.int16) & 0xF).astype(np.uint8)
    if flat.size % 2:
        flat = np.append(flat, np.uint8(0))
    raw = (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8).tobytes()
    return helper.make_tensor(name, TensorProto.INT4, q.shape, raw, raw=True)


def int4_one(model, name, kind, a, block):
    # block along the input (reduction) dim: matmul K = axis 0, gemm_t K = axis 1,
    # conv Cin = axis 1, embedding D = axis 1
    bax = {"matmul": 0, "gemm_t": 1, "conv": 1, "embed": 1}[kind]
    n = a.shape[bax]
    nb = -(-n // block)
    pad = nb * block - n
    ap = np.moveaxis(a, bax, 0)
    if pad:
        ap = np.concatenate([ap, np.zeros((pad,) + ap.shape[1:], np.float32)])
    blocks = ap.reshape((nb, block) + ap.shape[1:])
    scale = np.maximum(np.abs(blocks).max(axis=1), 1e-12) / 7.0           # (nb, ...)
    q = np.clip(np.rint(blocks / scale[:, None]), -8, 7).reshape(ap.shape)[:n]
    q = np.moveaxis(q, 0, bax).astype(np.int8)
    scale = np.moveaxis(scale, 0, bax).astype(np.float32)
    replace_init(model, name, [int4_tensor(name + "_q", q), numpy_helper.from_array(scale, name + "_s"),
                               int4_tensor(name + "_z", np.zeros(scale.shape, np.int8))],
                 [helper.make_node("DequantizeLinear", [name + "_q", name + "_s", name + "_z"], [name],
                                   axis=bax, block_size=block, name=name + "_dq")])


def to_int4(model, block=BLOCK, int4_kinds=("matmul", "gemm_t", "conv", "embed")):
    """int4 for int4_kinds, int8 per channel for the other weights. Needs opset 21 (int4 + blocked DQ)."""
    model = version_converter.convert_version(model, 21)
    for name, (kind, a) in weight_uses(model).items():
        if kind in int4_kinds:
            int4_one(model, name, kind, a, block)
        else:
            int8_one(model, name, kind, a)
    return model


def to_int8dyn(src, dst):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul", "Gemm", "Gather"])  # ConvInteger(int8) is missing in ORT 1.20 / Web


def build(variant):
    out = OUT / variant
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy(SRC / "metadata.json", out / "metadata.json")
    for g in GRAPHS:
        src = SRC / f"{g}.onnx"
        dst = out / f"{g}.onnx"
        if variant == "fp32":
            shutil.copy(src, dst)
            continue
        if variant == "int8dyn":
            to_int8dyn(src, dst)
            continue
        m = onnx.load(str(src))
        m = {"fp16": to_fp16, "int8": to_int8, "int4": to_int4,
             "int4b16": lambda m: to_int4(m, block=16),
             "int4mix": lambda m: to_int4(m, block=16, int4_kinds=("matmul", "gemm_t"))}[variant](m)
        onnx.checker.check_model(m)
        onnx.save(m, str(dst))


if __name__ == "__main__":
    for v in sys.argv[3].split(","):
        try:
            build(v)
            print("built", v)
        except Exception as e:
            print("FAILED", v, type(e).__name__, str(e)[:300])
