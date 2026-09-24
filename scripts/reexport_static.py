"""Re-export a trained cat landmark model with a fully static inference graph.

Port of the same fix made in dogs-in-the-wild-ml. `export_tflite` converts with
`from_keras_model`, which leaves the batch dimension dynamic, so each of the four
Conv2DTranspose layers builds its output shape at run time out of
SHAPE / STRIDED_SLICE / PACK. That leaves a dynamic-sized tensor in the graph and
TFLite then refuses to hand the affected region to the XNNPACK delegate:

  Attempting to use a delegate that only supports static-sized tensors with a
  graph that has dynamic-sized tensors (tensor#... )

Converting from a concrete function with the batch pinned to 1 constant-folds all
of it away. Weights are untouched, so this is purely a graph-shape change.
Inference on device is always batch 1, so nothing is given up.

The shipped cat_face_landmarks_full.tflite has exactly the same 295-op signature
as the dog model did (7 PACK, 5 SHAPE, 5 STRIDED_SLICE), because both come from
the same small_v3large_384_long recipe.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import tensorflow as tf

REPO = Path(__file__).resolve().parent.parent


def load_model(keras_path: Path) -> tf.keras.Model:
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import train_cat_face_landmarks as T  # registers SoftArgmax2D / WarmupSchedule
    _ = T
    return tf.keras.models.load_model(keras_path, compile=False)


def export_static_fp16(model: tf.keras.Model, out_path: Path, img_size: int) -> None:
    @tf.function(input_signature=[
        tf.TensorSpec([1, img_size, img_size, 3], tf.float32, name="crop")
    ])
    def serve(x):
        return {"landmarks_xy": model(x, training=False)}

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serve.get_concrete_function()], model
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    blob = converter.convert()
    out_path.write_bytes(blob)
    print(f"Saved {out_path} ({len(blob)/1024/1024:.2f} MB)")


def op_histogram(path: Path) -> dict[str, int]:
    from tensorflow.lite.python import schema_py_generated as schema
    buf = open(path, "rb").read()
    m = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(bytearray(buf), 0))
    names = {v: k for k, v in schema.BuiltinOperator.__dict__.items()
             if isinstance(v, int)}
    codes = [oc.builtinCode if oc.builtinCode else 0 for oc in m.operatorCodes]
    cnt = collections.Counter()
    for op in m.subgraphs[0].operators:
        cnt[names.get(codes[op.opcodeIndex], "?")] += 1
    return dict(cnt)


def transpose_conv_versions(path: Path) -> list[int]:
    """Opcode versions of every TRANSPOSE_CONV in a .tflite file."""
    from tensorflow.lite.python import schema_py_generated as schema

    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(path.read_bytes(), 0))
    return [oc.version for oc in model.operatorCodes
            if (oc.builtinCode if oc.builtinCode else oc.deprecatedBuiltinCode) == 67]


def ensure_gpu_compatible(path: Path) -> None:
    """Leave every TRANSPOSE_CONV at version 3 or lower, or fail.

    The converter fuses the ReLU after each Conv2DTranspose into the op, which makes it
    version 4. LiteRT's GPU accelerator accepts at most version 3, so it leaves the whole
    deconv head on the CPU. The static conversion above does not change this: every
    landmark export so far came out at version 4, MobileNetV3-Large and EfficientNetV2-S,
    cat and dog. The shipped landmark files are version 3 only because
    unfuse_transpose_conv_relu.py was run on them as a separate step, which is easy to
    miss, so it runs here on every export. A graph with nothing above version 3 is left
    byte-for-byte unchanged. See "READ THIS BEFORE EXPORTING ANY MODEL" in
    LANDMARK_DETECTION_REPORT.md.
    """
    if max(transpose_conv_versions(path), default=0) > 3:
        from unfuse_transpose_conv_relu import unfuse

        unfuse(path, path)
    versions = transpose_conv_versions(path)
    if max(versions, default=0) > 3:
        raise SystemExit(f"TRANSPOSE_CONV is still version {max(versions)} in {path}")
    print("TRANSPOSE_CONV versions:", versions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--img-size", type=int, default=384)
    args = ap.parse_args()

    export_static_fp16(load_model(args.keras), args.out, args.img_size)
    ensure_gpu_compatible(args.out)

    hist = op_histogram(args.out)
    print("ops:", sum(hist.values()))
    for k in sorted(hist, key=lambda k: -hist[k]):
        print(f"  {k:22s} {hist[k]}")


if __name__ == "__main__":
    main()
