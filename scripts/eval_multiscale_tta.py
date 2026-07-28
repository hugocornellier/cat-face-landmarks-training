#!/usr/bin/env python3
"""Multi-scale TTA evaluation for the tight_margin (224px) model.

Evaluates four configurations:
  1. Baseline (no TTA)
  2. Flip TTA only
  3. Multi-scale TTA (3 scales, no flip)
  4. Multi-scale + flip TTA (6 passes)

Scale TTA works on the crop tensor directly:
  scale=1.0 -> identity
  scale=0.9 -> center-crop to round(0.9*H), resize back to H (zoom in)
  scale=1.1 -> pad borders to round(1.1*H), resize back to H (zoom out)

Coordinate mapping (scale s, center 0.5):
  p_original = (p_scaled - 0.5) * s + 0.5

Usage:
    PYTHONUNBUFFERED=1 python scripts/eval_multiscale_tta.py
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cat_face_landmarks import (
    NUM_LANDMARKS,
    FLIP_INDEX,
    LEFT_OUTER_EYE_IDX,
    RIGHT_OUTER_EYE_IDX,
    EXPERIMENT_PRESETS,
    SoftArgmax2D,
    WarmupSchedule,
    set_seed,
    configure_ca_bundle,
    load_all_records,
    split_records,
    crop_and_normalize,
)

DATA_ROOT = (
    Path.home() / ".cache" / "kagglehub" / "datasets"
    / "georgemartvel" / "catflw" / "versions" / "2" / "CatFLW dataset"
)
# Scale factors for multi-scale TTA
SCALES = [0.9, 1.0, 1.1]

# Landmark region definitions (CatFLW has 48 landmarks)
REGIONS = {
    "right_eye": [3, 4, 5, 6, 7, 36, 37, 38],
    "left_eye": [1, 8, 9, 10, 11, 39, 40, 41],
    "right_ear": [22, 23, 24, 25, 26],
    "left_ear": [27, 28, 29, 30, 31],
    "nose": [12, 13, 14, 15, 32, 33, 34, 35, 42, 43, 44, 45],
    "mouth_chin": [0, 2, 16, 17, 18, 19, 20, 21, 46, 47],
}


def scale_crop(crop: tf.Tensor, scale: float) -> tf.Tensor:
    """Transform a crop tensor to simulate a different scale.

    scale < 1.0 (e.g. 0.9): zoom in -- center-crop to scale*H, resize back.
    scale = 1.0: identity.
    scale > 1.0 (e.g. 1.1): zoom out -- pad borders to scale*H, resize back.

    Args:
        crop: float32 tensor [H, W, C] in [0, 1].
        scale: float scale factor.

    Returns:
        Transformed crop tensor [H, W, C].
    """
    if abs(scale - 1.0) < 1e-6:
        return crop

    h = tf.shape(crop)[0]
    w = tf.shape(crop)[1]
    h_f = tf.cast(h, tf.float32)
    w_f = tf.cast(w, tf.float32)

    if scale < 1.0:
        # Zoom in: center-crop to (scale * H) x (scale * W), then resize back.
        new_h = tf.cast(tf.math.round(h_f * scale), tf.int32)
        new_w = tf.cast(tf.math.round(w_f * scale), tf.int32)
        offset_y = (h - new_h) // 2
        offset_x = (w - new_w) // 2
        cropped = tf.image.crop_to_bounding_box(crop, offset_y, offset_x, new_h, new_w)
        return tf.image.resize(cropped, [h, w], antialias=True)
    else:
        # Zoom out: pad borders symmetrically, then resize back.
        pad_h = tf.cast(tf.math.round(h_f * (scale - 1.0) / 2.0), tf.int32)
        pad_w = tf.cast(tf.math.round(w_f * (scale - 1.0) / 2.0), tf.int32)
        # Reflect-pad to preserve border content (avoids black border artifacts).
        padded = tf.pad(
            crop,
            [[pad_h, pad_h], [pad_w, pad_w], [0, 0]],
            mode="REFLECT",
        )
        return tf.image.resize(padded, [h, w], antialias=True)


def unscale_coords(coords_2d: tf.Tensor, scale: float) -> tf.Tensor:
    """Map predicted coordinates from the scaled crop frame back to original frame.

    The transformation applied to the image was:
        p_scaled = (p_original - 0.5) / s + 0.5

    So the inverse is:
        p_original = (p_scaled - 0.5) * s + 0.5

    Args:
        coords_2d: float32 tensor [NUM_LANDMARKS, 2] in [0, 1].
        scale: the scale factor that was applied to produce the crop.

    Returns:
        Coordinates in the original (scale=1.0) crop frame.
    """
    if abs(scale - 1.0) < 1e-6:
        return coords_2d
    return (coords_2d - 0.5) * scale + 0.5


def predict_scaled(model, crop: tf.Tensor, scale: float) -> tf.Tensor:
    """Run inference on a scaled version of crop, return coords in original frame.

    Args:
        model: Keras model.
        crop: [H, W, C] float32 tensor.
        scale: scale factor.

    Returns:
        coords_2d: [NUM_LANDMARKS, 2] float32 in original crop frame.
    """
    scaled = scale_crop(crop, scale)
    raw = model(tf.expand_dims(scaled, 0), training=False)[0]  # [NUM_LANDMARKS*2]
    coords_2d = tf.reshape(raw, [NUM_LANDMARKS, 2])
    return unscale_coords(coords_2d, scale)


def predict_scaled_flipped(
    model, crop: tf.Tensor, scale: float, flip_idx: tf.Tensor
) -> tf.Tensor:
    """Run inference on a flipped + scaled crop, return coords in original frame.

    Returns:
        coords_2d: [NUM_LANDMARKS, 2] float32 in original crop frame (unflipped).
    """
    scaled = scale_crop(crop, scale)
    flipped = tf.image.flip_left_right(scaled)
    raw = model(tf.expand_dims(flipped, 0), training=False)[0]
    coords_2d = tf.reshape(raw, [NUM_LANDMARKS, 2])
    # Unflip: remap landmark indices, mirror x coordinate.
    remapped = tf.gather(coords_2d, flip_idx, axis=0)
    unflipped = tf.stack([1.0 - remapped[:, 0], remapped[:, 1]], axis=-1)
    return unscale_coords(unflipped, scale)


def compute_nme(pred_2d: tf.Tensor, true_2d: tf.Tensor) -> float:
    """Compute NME_IOD (%) for a single sample."""
    iod = tf.sqrt(
        tf.reduce_sum(
            tf.square(true_2d[LEFT_OUTER_EYE_IDX] - true_2d[RIGHT_OUTER_EYE_IDX])
        ) + 1e-8
    )
    dists = tf.sqrt(tf.reduce_sum(tf.square(pred_2d - true_2d), axis=-1) + 1e-8)
    return float(tf.reduce_mean(dists) / tf.maximum(iod, 1e-8) * 100.0)


def evaluate_all(model, val_records, cfg):
    """Run all four evaluation modes over val_records.

    Returns dict with keys:
        'baseline', 'flip_tta', 'multiscale', 'multiscale_flip'
    """
    flip_idx = tf.constant(FLIP_INDEX, dtype=tf.int32)

    nmes_baseline = []
    nmes_flip = []
    nmes_ms = []
    nmes_ms_flip = []

    n = len(val_records)
    for i, rec in enumerate(val_records):
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        lm_flat = tf.constant(
            [c for pt in rec.landmarks_abs for c in pt], dtype=tf.float32
        )
        crop, lm_norm = crop_and_normalize(
            image,
            tf.constant(rec.bbox_xyxy_abs, tf.float32),
            lm_flat,
            cfg.img_size,
            cfg.crop_margin,
            square_pad=getattr(cfg, 'square_pad', False),
        )
        true_2d = tf.reshape(lm_norm, [NUM_LANDMARKS, 2])

        # --- Baseline (scale=1.0, no flip) ---
        coords_base = predict_scaled(model, crop, 1.0)
        nmes_baseline.append(compute_nme(coords_base, true_2d))

        # --- Flip TTA (scale=1.0 only, original + flipped) ---
        coords_flip = predict_scaled_flipped(model, crop, 1.0, flip_idx)
        coords_flip_avg = (coords_base + coords_flip) / 2.0
        nmes_flip.append(compute_nme(coords_flip_avg, true_2d))

        # --- Multi-scale TTA (3 scales, no flip) ---
        preds_ms = [predict_scaled(model, crop, s) for s in SCALES]
        coords_ms_avg = tf.reduce_mean(tf.stack(preds_ms, axis=0), axis=0)
        nmes_ms.append(compute_nme(coords_ms_avg, true_2d))

        # --- Multi-scale + flip TTA (6 passes total) ---
        preds_ms_flip = [predict_scaled_flipped(model, crop, s, flip_idx) for s in SCALES]
        all_preds = preds_ms + preds_ms_flip  # 6 tensors [NUM_LANDMARKS, 2]
        coords_msf_avg = tf.reduce_mean(tf.stack(all_preds, axis=0), axis=0)
        nmes_ms_flip.append(compute_nme(coords_msf_avg, true_2d))

        if (i + 1) % 100 == 0:
            print(
                f"  [{i+1}/{n}]  "
                f"baseline={np.mean(nmes_baseline):.4f}  "
                f"flip={np.mean(nmes_flip):.4f}  "
                f"ms={np.mean(nmes_ms):.4f}  "
                f"ms+flip={np.mean(nmes_ms_flip):.4f}"
            )

    return {
        "baseline": float(np.mean(nmes_baseline)),
        "flip_tta": float(np.mean(nmes_flip)),
        "multiscale": float(np.mean(nmes_ms)),
        "multiscale_flip": float(np.mean(nmes_ms_flip)),
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=Path("artifacts/tight_margin_384_long/best.keras"),
                   help="Path to best.keras model file")
    p.add_argument("--experiment", type=str, default="tight_margin_384",
                   help="Experiment preset name (for img_size/crop_margin)")
    p.add_argument("--img-size", type=int, default=None,
                   help="Override image size from preset")
    args = p.parse_args()

    configure_ca_bundle()
    set_seed(42)

    cfg = copy.deepcopy(EXPERIMENT_PRESETS[args.experiment])
    if args.img_size:
        cfg.img_size = args.img_size
    print(f"Preset:      {args.experiment}")
    print(f"img_size:    {cfg.img_size}")
    print(f"crop_margin: {cfg.crop_margin}")
    print(f"lm_margin:   {cfg.lm_margin}")
    print(f"Scales:      {SCALES}")

    print(f"\nLoading val set (lm_margin={cfg.lm_margin})...")
    all_records = load_all_records(DATA_ROOT, lm_margin=0.05)
    _, val_records = split_records(all_records, test_fraction=0.15, seed=42)
    print(f"Val records: {len(val_records)}")

    model_path = args.model
    print(f"\nLoading model from {model_path} ...")
    custom_objects = {"SoftArgmax2D": SoftArgmax2D, "WarmupSchedule": WarmupSchedule}
    model = tf.keras.models.load_model(
        str(model_path), custom_objects=custom_objects, compile=False
    )
    print(f"Model output shape: {model.output_shape}")

    print(f"\nRunning multi-scale TTA evaluation ({len(val_records)} samples, 6 passes each)...")
    print(f"  Modes: baseline | flip | ms(3 scales) | ms+flip(6 passes)")
    results = evaluate_all(model, val_records, cfg)

    print()
    print("=" * 58)
    print("RESULTS -- CatFLW tight_margin multi-scale TTA")
    print("=" * 58)
    print(f"  1. Baseline (no TTA):          {results['baseline']:.4f}")
    print(f"  2. Flip TTA only:              {results['flip_tta']:.4f}")
    print(f"  3. Multi-scale (3 scales):     {results['multiscale']:.4f}")
    print(f"  4. Multi-scale + flip (6p):    {results['multiscale_flip']:.4f}")
    print("=" * 58)

    best_key = min(results, key=results.__getitem__)
    best_val = results[best_key]
    label_map = {
        "baseline": "Baseline",
        "flip_tta": "Flip TTA",
        "multiscale": "Multi-scale (3 scales)",
        "multiscale_flip": "Multi-scale + flip (6p)",
    }
    print(f"\nBest configuration: {label_map[best_key]}  NME={best_val:.4f}")


if __name__ == "__main__":
    main()
