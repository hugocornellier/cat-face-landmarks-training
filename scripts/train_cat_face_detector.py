#!/usr/bin/env python3
"""Train a small CatFLW cat-face detector/localizer and export TFLite.

This script uses CatFLW per-image landmarks to derive a reliable face bounding
box (instead of trusting the provided bounding_boxes field).

Model output:
  - one normalized bbox in letterboxed input coordinates: [x1, y1, x2, y2]
  - values are in [0, 1] relative to the fixed model input size.

Intended runtime pipeline:
  1) letterbox image/frame to model input size
  2) run TFLite detector
  3) de-letterbox bbox back to original coordinates
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

NUM_LANDMARKS = 48


@dataclass(frozen=True)
class Record:
    image_path: str
    bbox_xyxy_abs: tuple[float, float, float, float]
    orig_size_wh: tuple[int, int]


def parse_args() -> argparse.Namespace:
    default_root = (
        Path.home()
        / ".cache"
        / "kagglehub"
        / "datasets"
        / "georgemartvel"
        / "catflw"
        / "versions"
        / "2"
        / "CatFLW dataset"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/cat_face_detector"))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10, help="Frozen-backbone epochs")
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--finetune-last-layers", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--bbox-margin", type=float, default=0.12)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--tflite-only", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def configure_ca_bundle() -> None:
    try:
        import certifi
    except Exception:
        return
    ca_path = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca_path)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_path)


# ---------------------------------------------------------------------------
# Data loading  (CatFLW: flat images/ + labels/ with no train/test split)
# ---------------------------------------------------------------------------

def _landmark_bbox(
    landmarks: list[list[float]],
    img_w: int,
    img_h: int,
    margin: float,
) -> tuple[float, float, float, float] | None:
    xs, ys = [], []
    for pt in landmarks:
        if len(pt) != 2:
            continue
        x = float(min(max(pt[0], 0.0), img_w))
        y = float(min(max(pt[1], 0.0), img_h))
        xs.append(x)
        ys.append(y)
    if not xs or not ys:
        return None
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return None
    mx, my = bw * margin, bh * margin
    x1 = max(0.0, x1 - mx)
    y1 = max(0.0, y1 - my)
    x2 = min(float(img_w), x2 + mx)
    y2 = min(float(img_h), y2 + my)
    if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
        return None
    return (x1, y1, x2, y2)


def load_all_records(data_root: Path, margin: float) -> list[Record]:
    image_dir = data_root / "images"
    label_dir = data_root / "labels"
    records: list[Record] = []
    skipped = 0

    for img_path in sorted(image_dir.glob("*.png")):
        lbl_path = label_dir / f"{img_path.stem}.json"
        if not lbl_path.exists():
            skipped += 1
            continue
        ann = json.loads(lbl_path.read_text(encoding="utf-8"))
        landmarks = ann.get("labels", [])
        if len(landmarks) != NUM_LANDMARKS:
            skipped += 1
            continue
        if any(
            not (isinstance(pt[0], (int, float)) and isinstance(pt[1], (int, float))
                 and pt[0] == pt[0] and pt[1] == pt[1])
            for pt in landmarks
        ):
            skipped += 1
            continue
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            skipped += 1
            continue
        bbox = _landmark_bbox(landmarks, img_w, img_h, margin)
        if bbox is None:
            skipped += 1
            continue
        records.append(Record(
            image_path=str(img_path),
            bbox_xyxy_abs=bbox,
            orig_size_wh=(img_w, img_h),
        ))

    print(f"[all] usable={len(records)} skipped={skipped}")
    return records


def split_records(
    records: list[Record], test_fraction: float, seed: int,
) -> tuple[list[Record], list[Record]]:
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_test = max(1, int(len(records) * test_fraction))
    test_indices = set(indices[:n_test])
    train = [records[i] for i in range(len(records)) if i not in test_indices]
    test = [records[i] for i in range(len(records)) if i in test_indices]
    print(f"[split] train={len(train)} test={len(test)} (test_fraction={test_fraction})")
    return train, test


# ---------------------------------------------------------------------------
# Letterbox + augmentation
# ---------------------------------------------------------------------------

def build_tf_dataset(
    records: list[Record],
    img_size: int,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    paths = np.array([r.image_path for r in records], dtype=object)
    boxes = np.array([r.bbox_xyxy_abs for r in records], dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, boxes))
    if training:
        ds = ds.shuffle(len(records), seed=seed, reshuffle_each_iteration=True)

    def _decode_and_letterbox(path: tf.Tensor, box_abs: tf.Tensor):
        img_bytes = tf.io.read_file(path)
        image = tf.io.decode_png(img_bytes, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image, box_norm = letterbox_image_and_box(image, box_abs, img_size)
        if training:
            image, box_norm = augment_image_and_box(image, box_norm, img_size)
        return image, box_norm

    ds = ds.map(_decode_and_letterbox, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def letterbox_image_and_box(
    image: tf.Tensor, box_abs: tf.Tensor, img_size: int
) -> tuple[tf.Tensor, tf.Tensor]:
    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    target = tf.cast(img_size, tf.float32)

    scale = tf.minimum(target / w, target / h)
    new_w = tf.cast(tf.round(w * scale), tf.int32)
    new_h = tf.cast(tf.round(h * scale), tf.int32)

    resized = tf.image.resize(image, [new_h, new_w], antialias=True)
    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2
    image_lb = tf.image.pad_to_bounding_box(resized, pad_y, pad_x, img_size, img_size)

    pad = tf.cast(tf.stack([pad_x, pad_y, pad_x, pad_y]), tf.float32)
    box_lb = box_abs * scale + pad
    box_norm = tf.clip_by_value(box_lb / target, 0.0, 1.0)
    return image_lb, box_norm


def random_zoom_out(
    image: tf.Tensor, box_xyxy: tf.Tensor, img_size: int
) -> tuple[tf.Tensor, tf.Tensor]:
    def _zoomed():
        zoom = tf.random.uniform((), 1.3, 2.0)
        canvas_size_f = tf.cast(img_size, tf.float32) * zoom
        canvas_size = tf.cast(canvas_size_f, tf.int32)
        max_offset = canvas_size - img_size
        ox = tf.random.uniform((), 0, max_offset + 1, dtype=tf.int32)
        oy = tf.random.uniform((), 0, max_offset + 1, dtype=tf.int32)
        canvas = tf.image.pad_to_bounding_box(image, oy, ox, canvas_size, canvas_size)
        canvas = tf.image.resize(canvas, [img_size, img_size], antialias=True)
        canvas = tf.cast(tf.clip_by_value(canvas, 0.0, 1.0), tf.float32)
        ox_f = tf.cast(ox, tf.float32)
        oy_f = tf.cast(oy, tf.float32)
        img_f = tf.cast(img_size, tf.float32)
        offsets = tf.stack([ox_f, oy_f, ox_f, oy_f])
        box_new = (box_xyxy * img_f + offsets) / canvas_size_f
        box_new = tf.clip_by_value(box_new, 0.0, 1.0)
        return canvas, box_new

    do_zoom = tf.random.uniform(()) < 0.5
    return tf.cond(do_zoom, _zoomed, lambda: (image, box_xyxy))


def augment_image_and_box(
    image: tf.Tensor, box_xyxy: tf.Tensor, img_size: int = 224
) -> tuple[tf.Tensor, tf.Tensor]:
    image, box_xyxy = random_zoom_out(image, box_xyxy, img_size)

    do_flip = tf.random.uniform(()) < 0.5
    if do_flip:
        image = tf.image.flip_left_right(image)
        x1, y1, x2, y2 = tf.unstack(box_xyxy)
        box_xyxy = tf.stack([1.0 - x2, y1, 1.0 - x1, y2], axis=0)

    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, box_xyxy


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------

def order_and_clip_boxes(boxes_xyxy: tf.Tensor) -> tf.Tensor:
    boxes_xyxy = tf.clip_by_value(boxes_xyxy, 0.0, 1.0)
    x1 = tf.minimum(boxes_xyxy[..., 0], boxes_xyxy[..., 2])
    y1 = tf.minimum(boxes_xyxy[..., 1], boxes_xyxy[..., 3])
    x2 = tf.maximum(boxes_xyxy[..., 0], boxes_xyxy[..., 2])
    y2 = tf.maximum(boxes_xyxy[..., 1], boxes_xyxy[..., 3])
    return tf.stack([x1, y1, x2, y2], axis=-1)


def bbox_iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = order_and_clip_boxes(y_true)
    y_pred = order_and_clip_boxes(y_pred)
    x1 = tf.maximum(y_true[..., 0], y_pred[..., 0])
    y1 = tf.maximum(y_true[..., 1], y_pred[..., 1])
    x2 = tf.minimum(y_true[..., 2], y_pred[..., 2])
    y2 = tf.minimum(y_true[..., 3], y_pred[..., 3])
    inter = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    area_true = tf.maximum(0.0, y_true[..., 2] - y_true[..., 0]) * tf.maximum(0.0, y_true[..., 3] - y_true[..., 1])
    area_pred = tf.maximum(0.0, y_pred[..., 2] - y_pred[..., 0]) * tf.maximum(0.0, y_pred[..., 3] - y_pred[..., 1])
    union = tf.maximum(area_true + area_pred - inter, 1e-8)
    return inter / union


def bbox_ciou_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_pred = order_and_clip_boxes(y_pred)
    y_true = order_and_clip_boxes(y_true)

    ix1 = tf.maximum(y_true[..., 0], y_pred[..., 0])
    iy1 = tf.maximum(y_true[..., 1], y_pred[..., 1])
    ix2 = tf.minimum(y_true[..., 2], y_pred[..., 2])
    iy2 = tf.minimum(y_true[..., 3], y_pred[..., 3])
    inter = tf.maximum(0.0, ix2 - ix1) * tf.maximum(0.0, iy2 - iy1)

    area_t = tf.maximum(0.0, y_true[..., 2] - y_true[..., 0]) * tf.maximum(0.0, y_true[..., 3] - y_true[..., 1])
    area_p = tf.maximum(0.0, y_pred[..., 2] - y_pred[..., 0]) * tf.maximum(0.0, y_pred[..., 3] - y_pred[..., 1])
    union = tf.maximum(area_t + area_p - inter, 1e-8)
    iou = inter / union

    cx1 = tf.minimum(y_true[..., 0], y_pred[..., 0])
    cy1 = tf.minimum(y_true[..., 1], y_pred[..., 1])
    cx2 = tf.maximum(y_true[..., 2], y_pred[..., 2])
    cy2 = tf.maximum(y_true[..., 3], y_pred[..., 3])
    c_diag_sq = tf.maximum((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2, 1e-8)

    ct_x = (y_true[..., 0] + y_true[..., 2]) / 2.0
    ct_y = (y_true[..., 1] + y_true[..., 3]) / 2.0
    cp_x = (y_pred[..., 0] + y_pred[..., 2]) / 2.0
    cp_y = (y_pred[..., 1] + y_pred[..., 3]) / 2.0
    rho_sq = (ct_x - cp_x) ** 2 + (ct_y - cp_y) ** 2

    w_t = tf.maximum(y_true[..., 2] - y_true[..., 0], 1e-8)
    h_t = tf.maximum(y_true[..., 3] - y_true[..., 1], 1e-8)
    w_p = tf.maximum(y_pred[..., 2] - y_pred[..., 0], 1e-8)
    h_p = tf.maximum(y_pred[..., 3] - y_pred[..., 1], 1e-8)
    v = (4.0 / (math.pi ** 2)) * tf.square(tf.atan(w_t / h_t) - tf.atan(w_p / h_p))
    alpha = v / tf.maximum(1.0 - iou + v, 1e-8)

    ciou = iou - rho_sq / c_diag_sq - alpha * v
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)
    return l1 + (1.0 - ciou)


bbox_regression_loss = bbox_ciou_loss


def bbox_iou_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return bbox_iou(y_true, y_pred)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(img_size: int, pretrained: bool) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = tf.keras.layers.Rescaling(scale=255.0, offset=0.0, name="to_0_255")(inputs)

    backbone = tf.keras.applications.EfficientNetB2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights=None if not pretrained else "imagenet",
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(4, activation="sigmoid", name="bbox_xyxy")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cat_face_localizer")
    return model


def compile_model(model: tf.keras.Model, lr) -> None:
    try:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    except AttributeError:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss=bbox_regression_loss,
        metrics=[bbox_iou_metric, tf.keras.metrics.MeanAbsoluteError(name="mae_xyxy")],
        run_eagerly=True,
    )


def get_backbone(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("efficientnet"):
            return layer
    raise RuntimeError("EfficientNet backbone not found in model")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    out_dir: Path,
    args: argparse.Namespace,
    num_train: int,
) -> tf.keras.Model:
    out_dir.mkdir(parents=True, exist_ok=True)
    phase1_model_path = out_dir / "phase1_best.keras"
    phase2_model_path = out_dir / "phase2_best.keras"
    phase1_weights_path = out_dir / "phase1_best.weights.h5"
    phase2_weights_path = out_dir / "phase2_best.weights.h5"
    ckpt_weights_path = out_dir / "best.weights.h5"
    for stale in (
        phase1_model_path, phase2_model_path,
        phase1_weights_path, phase2_weights_path,
        ckpt_weights_path, out_dir / "best.keras",
    ):
        if stale.exists() and stale.is_file():
            stale.unlink()

    phase1_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_bbox_iou_metric", mode="max",
            patience=6, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_bbox_iou_metric", mode="max",
            factor=0.5, patience=3, min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / "train_log.csv"), append=False),
    ]

    print("\n=== Phase 1: frozen backbone ===")
    compile_model(model, lr=args.learning_rate)
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=max(args.epochs, 0),
        callbacks=phase1_callbacks, verbose=2,
    )
    phase1_metrics = evaluate_model(model, val_ds)
    phase1_score = float(phase1_metrics.get("bbox_iou_metric", -float("inf")))
    phase1_state = model.get_weights()
    model.save(phase1_model_path)
    model.save_weights(phase1_weights_path)

    best_score = phase1_score
    best_source = "phase1"
    best_state = [np.array(w, copy=True) for w in phase1_state]

    if not args.skip_finetune and args.finetune_epochs > 0:
        print("\n=== Phase 2: fine-tune backbone tail ===")
        backbone = get_backbone(model)
        backbone.trainable = True
        if args.finetune_last_layers > 0:
            for layer in backbone.layers[:-args.finetune_last_layers]:
                layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        steps_per_epoch = math.ceil(num_train / args.batch_size)
        total_steps = steps_per_epoch * args.finetune_epochs
        cosine_lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.finetune_learning_rate,
            decay_steps=total_steps, alpha=1e-6,
        )
        compile_model(model, lr=cosine_lr)

        phase2_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(phase2_model_path), monitor="val_bbox_iou_metric",
                mode="max", save_best_only=True, verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_bbox_iou_metric", mode="max",
                patience=10, restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(out_dir / "train_log.csv"), append=True),
        ]
        model.fit(
            train_ds, validation_data=val_ds,
            epochs=max(args.finetune_epochs, 0),
            callbacks=phase2_callbacks, verbose=2,
        )
        phase2_metrics = evaluate_model(model, val_ds)
        phase2_score = float(phase2_metrics.get("bbox_iou_metric", -float("inf")))
        model.save(phase2_model_path)
        model.save_weights(phase2_weights_path)
        if phase2_score >= best_score:
            best_score = phase2_score
            best_source = "phase2"
            best_state = [np.array(w, copy=True) for w in model.get_weights()]

    print(f"Selecting {best_source} model (val_bbox_iou_metric={best_score:.6f})")
    model.set_weights(best_state)
    model.save(out_dir / "best.keras")
    model.save_weights(ckpt_weights_path)
    return model


def evaluate_model(model: tf.keras.Model, val_ds: tf.data.Dataset) -> dict[str, float]:
    values = model.evaluate(val_ds, verbose=0)
    names = model.metrics_names
    metrics = {name: float(val) for name, val in zip(names, values)}
    print("Validation metrics:", metrics)
    return metrics


# ---------------------------------------------------------------------------
# TFLite export + sanity check
# ---------------------------------------------------------------------------

def export_tflite(model: tf.keras.Model, out_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    print(f"Saved TFLite model: {out_path} ({len(tflite_model)/1024/1024:.2f} MB)")


def tflite_sanity_check(
    tflite_path: Path,
    val_records: list[Record],
    img_size: int,
    num_samples: int = 16,
) -> dict[str, float]:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    print("TFLite input:", in_det["shape"], in_det["dtype"],
          "output:", out_det["shape"], out_det["dtype"])

    sample_records = val_records[:min(num_samples, len(val_records))]
    ious: list[float] = []
    for rec in sample_records:
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image_lb, y_true = letterbox_image_and_box(
            image, tf.constant(rec.bbox_xyxy_abs, tf.float32), img_size
        )
        inp = tf.expand_dims(image_lb, 0).numpy().astype(in_det["dtype"])
        interpreter.set_tensor(in_det["index"], inp)
        interpreter.invoke()
        pred = interpreter.get_tensor(out_det["index"]).astype(np.float32)[0]
        pred = order_and_clip_boxes(tf.constant(pred)).numpy()
        iou = float(bbox_iou(
            tf.expand_dims(y_true, 0), tf.expand_dims(pred, 0)
        ).numpy()[0])
        ious.append(iou)

    result = {"tflite_mean_iou_sample": float(np.mean(ious)) if ious else math.nan}
    print("TFLite sanity:", result)
    return result


def save_metadata(
    out_dir: Path,
    img_size: int,
    train_records: list[Record],
    val_records: list[Record],
    val_metrics: dict[str, float],
    tflite_path: Path,
    tflite_sanity: dict[str, float],
    args: argparse.Namespace,
) -> None:
    meta = {
        "model_name": "cat_face_localizer",
        "class_names": ["cat_face"],
        "data_root": str(args.data_root),
        "dataset": "CatFLW",
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "input": {
            "shape": [1, img_size, img_size, 3],
            "dtype": "float32",
            "range": [0.0, 1.0],
            "preprocessing": "letterbox to square, then rescale to [0, 255] for EfficientNetB2 in-model",
        },
        "output": {
            "name": "bbox_xyxy",
            "format": "normalized xyxy in letterboxed coordinates",
            "range": [0.0, 1.0],
            "notes": "single cat-face bounding box",
        },
        "training": {
            "bbox_source": "derived from landmarks with margin",
            "bbox_margin": args.bbox_margin,
            "epochs_frozen": args.epochs,
            "epochs_finetune": 0 if args.skip_finetune else args.finetune_epochs,
            "pretrained_backbone": not args.no_pretrained,
            "img_size": img_size,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "validation_metrics": val_metrics,
        "tflite_sanity": tflite_sanity,
        "artifacts": {
            "tflite": str(tflite_path),
            "keras_best": str(out_dir / "best.keras"),
            "weights_best": str(out_dir / "best.weights.h5"),
            "train_log_csv": str(out_dir / "train_log.csv"),
        },
    }
    (out_dir / "model_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def maybe_limit(records: list[Record], limit: int | None) -> list[Record]:
    if limit is None or limit >= len(records):
        return records
    return records[:limit]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    configure_ca_bundle()
    set_seed(args.seed)
    tf.config.optimizer.set_jit(False)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"TensorFlow: {tf.__version__}")
    print(f"Data root: {args.data_root}")
    if not args.data_root.exists():
        raise FileNotFoundError(f"CatFLW dataset not found at {args.data_root}")

    all_records = load_all_records(args.data_root, margin=args.bbox_margin)
    train_records, val_records = split_records(all_records, args.test_fraction, args.seed)
    train_records = maybe_limit(train_records, args.limit_train)
    val_records = maybe_limit(val_records, args.limit_val)
    if not train_records or not val_records:
        raise RuntimeError("Need non-empty train and validation records.")

    train_ds = build_tf_dataset(
        train_records, img_size=args.img_size, batch_size=args.batch_size,
        training=True, seed=args.seed,
    )
    val_ds = build_tf_dataset(
        val_records, img_size=args.img_size, batch_size=args.batch_size,
        training=False, seed=args.seed,
    )

    model = build_model(img_size=args.img_size, pretrained=not args.no_pretrained)
    model.summary()

    if not args.tflite_only:
        model = train_model(model, train_ds, val_ds, out_dir=args.out_dir, args=args,
                            num_train=len(train_records))
        model.save(args.out_dir / "final.keras")
    else:
        best_weights = args.out_dir / "best.weights.h5"
        if not best_weights.exists():
            raise FileNotFoundError(f"--tflite-only: {best_weights} not found")
        model.load_weights(str(best_weights))

    compile_model(model, lr=args.finetune_learning_rate)
    val_metrics = evaluate_model(model, val_ds)

    tflite_path = args.out_dir / f"cat_face_localizer_{args.img_size}_float16.tflite"
    export_tflite(model, tflite_path)

    tflite_sanity = tflite_sanity_check(tflite_path, val_records, img_size=args.img_size)
    save_metadata(
        out_dir=args.out_dir, img_size=args.img_size,
        train_records=train_records, val_records=val_records,
        val_metrics=val_metrics, tflite_path=tflite_path,
        tflite_sanity=tflite_sanity, args=args,
    )

    print("\nDone.")
    print(f"TFLite: {tflite_path}")
    print(f"Metadata: {args.out_dir / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
