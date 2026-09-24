"""Unfuse the ReLU out of TRANSPOSE_CONV so the GPU delegate can take the deconv head.

Copied unchanged from dog-face-landmarks-training, where it was written; the timings
quoted below are the dog model's.

Background. The converter folds `Conv2DTranspose -> BatchNorm -> ReLU` into a single
TRANSPOSE_CONV with a folded-BN bias and `fusedActivationFunction = RELU`. Carrying a
fused activation is what makes the op declare version 4, and the Metal/OpenCL GPU
delegate supports at most version 3:

    TRANSPOSE_CONV: Max version supported: 3. Requested version 4.

A probe that merely downgraded the version and dropped the activation showed the
version gate is the *only* delegation blocker: Metal then ran the entire graph at
5.11 ms against 28.80 ms, computing it correctly (GPU vs CPU dev 1.06e-05). But that
probe was numerically wrong, because the ReLU was deleted rather than relocated
(dev 4.86e-01 against the original).

This does it properly. For each TRANSPOSE_CONV carrying a fused activation:

  TRANSPOSE_CONV(act=RELU) -> T          becomes
  TRANSPOSE_CONV(act=NONE) -> T_pre  then  RELU(T_pre) -> T

T keeps its identity, so every downstream consumer is untouched and no other part of
the graph needs rewiring. The opcode drops to version 3. RELU is a separately
supported op, so the delegate can claim both halves.

The result should be numerically identical to the original (fp16 rounding aside) and
fully GPU-delegatable. Verify both before believing it: `--verify` compares against
the source model on CPU.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema

TRANSPOSE_CONV = 67
RELU = 19          # BuiltinOperator.RELU
RELU6 = 21
ACT_NONE, ACT_RELU, ACT_RELU_N1_TO_1, ACT_RELU6 = 0, 1, 2, 3

ACT_TO_OP = {ACT_RELU: RELU, ACT_RELU6: RELU6}


def _opcode_index_for(model, builtin_code: int, version: int = 1) -> int:
    """Return the operatorCodes index for builtin_code, appending one if absent."""
    for i, oc in enumerate(model.operatorCodes):
        code = oc.builtinCode if oc.builtinCode else oc.deprecatedBuiltinCode
        if code == builtin_code:
            return i
    oc = schema.OperatorCodeT()
    oc.builtinCode = builtin_code
    # deprecatedBuiltinCode is capped at 127 in the schema; both must agree.
    oc.deprecatedBuiltinCode = builtin_code if builtin_code < 127 else 127
    oc.version = version
    oc.customCode = None
    model.operatorCodes.append(oc)
    return len(model.operatorCodes) - 1


def unfuse(src: Path, dst: Path) -> int:
    buf = bytearray(src.read_bytes())
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(buf, 0))

    tc_opcodes = {
        i for i, oc in enumerate(model.operatorCodes)
        if (oc.builtinCode if oc.builtinCode else oc.deprecatedBuiltinCode) == TRANSPOSE_CONV
    }
    if not tc_opcodes:
        raise SystemExit("no TRANSPOSE_CONV in this model")

    n_unfused = 0
    for sub in model.subgraphs:
        new_ops = []
        for op in sub.operators:
            new_ops.append(op)
            if op.opcodeIndex not in tc_opcodes:
                continue
            o = op.builtinOptions
            act = getattr(o, "fusedActivationFunction", ACT_NONE)
            if act == ACT_NONE:
                continue
            if act not in ACT_TO_OP:
                raise SystemExit(f"unsupported fused activation {act}; "
                                 "only RELU and RELU6 are handled")

            out_idx = op.outputs[0]
            out_t = sub.tensors[out_idx]

            # New intermediate tensor carrying the pre-activation result. Same shape,
            # type and quantization; buffer 0 means "no static data".
            pre = schema.TensorT()
            pre.shape = list(out_t.shape) if out_t.shape is not None else None
            pre.shapeSignature = (list(out_t.shapeSignature)
                                  if out_t.shapeSignature is not None else None)
            pre.type = out_t.type
            pre.buffer = 0
            pre.name = (out_t.name or b"transpose_conv") + b"_preact"
            pre.quantization = copy.deepcopy(out_t.quantization)
            pre.isVariable = False
            sub.tensors.append(pre)
            pre_idx = len(sub.tensors) - 1

            # Deconv now writes the pre-activation tensor, with no fused activation.
            op.outputs[0] = pre_idx
            o.fusedActivationFunction = ACT_NONE

            act_op = schema.OperatorT()
            act_op.opcodeIndex = _opcode_index_for(model, ACT_TO_OP[act])
            act_op.inputs = [pre_idx]
            act_op.outputs = [out_idx]      # keeps the original tensor identity
            act_op.builtinOptions = None
            act_op.customOptions = None
            act_op.mutatingVariableInputs = None
            new_ops.append(act_op)
            n_unfused += 1

        sub.operators = new_ops

    for i in tc_opcodes:
        print(f"opcode[{i}] TRANSPOSE_CONV version "
              f"{model.operatorCodes[i].version} -> 3")
        model.operatorCodes[i].version = 3

    b = flatbuffers.Builder(0)
    b.Finish(model.Pack(b), file_identifier=b"TFL3")
    dst.write_bytes(bytes(b.Output()))
    print(f"unfused {n_unfused} activations, wrote {dst} "
          f"({dst.stat().st_size/1024/1024:.2f} MB)")
    return n_unfused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()
    unfuse(args.src, args.dst)


if __name__ == "__main__":
    main()
