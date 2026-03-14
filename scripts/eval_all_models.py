#!/usr/bin/env python3
"""Evaluate all trained models with TTA and all ensemble combinations.

Covers:
  - Each model: baseline, flip TTA, multi-scale+flip TTA
  - All pair ensembles with ms+flip TTA
  - Best 3-model ensemble with ms+flip TTA
  - Per-region NME breakdown for the best config

Usage:
    PYTHONUNBUFFERED=1 python scripts/eval_all_models.py
"""

from __future__ import annotations

import copy
import itertools
import os
import sys
import time
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

# Models to evaluate — add new models here
MODELS_CONFIG = [
    {
        "name": "tight_margin",
        "path": Path("artifacts/tight_margin/best.keras"),
        "preset": "tight_margin",
    },
    {
        "name": "tight_margin_256",
        "path": Path("artifacts/tight_margin_256/best.keras"),
        "preset": "tight_margin_256",
    },
    {
        "name": "tight_margin_320",
        "path": Path("artifacts/tight_margin_320/best.keras"),
        "preset": "tight_margin_320",
    },
    {
        "name": "tight_margin_384",
        "path": Path("artifacts/tight_margin_384/best.keras"),
        "preset": "tight_margin_384",
    },
]


def load_model(model_path: Path):
    custom_objects = {"SoftArgmax2D": SoftArgmax2D, "WarmupSchedule": WarmupSchedule}
    return tf.keras.models.load_model(
        str(model_path), custom_objects=custom_objects, compile=False
    )


def scale_crop(crop: tf.Tensor, scale: float) -> tf.Tensor:
    if abs(scale - 1.0) < 1e-6:
        return crop
    h = tf.shape(crop)[0]
    w = tf.shape(crop)[1]
    h_f = tf.cast(h, tf.float32)
    w_f = tf.cast(w, tf.float32)
    if scale < 1.0:
        new_h = tf.cast(tf.math.round(h_f * scale), tf.int32)
        new_w = tf.cast(tf.math.round(w_f * scale), tf.int32)
        offset_y = (h - new_h) // 2
        offset_x = (w - new_w) // 2
        cropped = tf.image.crop_to_bounding_box(crop, offset_y, offset_x, new_h, new_w)
        return tf.image.resize(cropped, [h, w], antialias=True)
    else:
        pad_h = tf.cast(tf.math.round(h_f * (scale - 1.0) / 2.0), tf.int32)
        pad_w = tf.cast(tf.math.round(w_f * (scale - 1.0) / 2.0), tf.int32)
        padded = tf.pad(crop, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode="REFLECT")
        return tf.image.resize(padded, [h, w], antialias=True)


def unscale_coords(coords_2d: tf.Tensor, scale: float) -> tf.Tensor:
    if abs(scale - 1.0) < 1e-6:
        return coords_2d
    return (coords_2d - 0.5) * scale + 0.5


def predict_scaled(model, crop, scale):
    scaled = scale_crop(crop, scale)
    raw = model(tf.expand_dims(scaled, 0), training=False)[0]
    coords_2d = tf.reshape(raw, [NUM_LANDMARKS, 2])
    return unscale_coords(coords_2d, scale)


def predict_scaled_flipped(model, crop, scale, flip_idx):
    scaled = scale_crop(crop, scale)
    flipped = tf.image.flip_left_right(scaled)
    raw = model(tf.expand_dims(flipped, 0), training=False)[0]
    coords_2d = tf.reshape(raw, [NUM_LANDMARKS, 2])
    remapped = tf.gather(coords_2d, flip_idx, axis=0)
    unflipped = tf.stack([1.0 - remapped[:, 0], remapped[:, 1]], axis=-1)
    return unscale_coords(unflipped, scale)


def compute_nme(pred_2d, true_2d):
    iod = tf.sqrt(
        tf.reduce_sum(
            tf.square(true_2d[LEFT_OUTER_EYE_IDX] - true_2d[RIGHT_OUTER_EYE_IDX])
        ) + 1e-8
    )
    dists = tf.sqrt(tf.reduce_sum(tf.square(pred_2d - true_2d), axis=-1) + 1e-8)
    return float(tf.reduce_mean(dists) / tf.maximum(iod, 1e-8) * 100.0)


def compute_per_region_nme(pred_2d, true_2d):
    iod = tf.sqrt(
        tf.reduce_sum(
            tf.square(true_2d[LEFT_OUTER_EYE_IDX] - true_2d[RIGHT_OUTER_EYE_IDX])
        ) + 1e-8
    )
    dists = tf.sqrt(tf.reduce_sum(tf.square(pred_2d - true_2d), axis=-1) + 1e-8)
    region_nmes = {}
    for name, indices in REGIONS.items():
        region_dists = tf.gather(dists, indices)
        region_nmes[name] = float(tf.reduce_mean(region_dists) / tf.maximum(iod, 1e-8) * 100.0)
    return region_nmes


def evaluate_single_model(model, val_records, cfg, tag):
    flip_idx = tf.constant(FLIP_INDEX, dtype=tf.int32)
    nmes_baseline, nmes_flip, nmes_ms_flip = [], [], []
    t0 = time.time()
    print(f"\n  [{tag}]  img_size={cfg.img_size}  scales={SCALES}")

    for i, rec in enumerate(val_records):
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        lm_flat = tf.constant([c for pt in rec.landmarks_abs for c in pt], dtype=tf.float32)
        crop, lm_norm = crop_and_normalize(
            image, tf.constant(rec.bbox_xyxy_abs, tf.float32),
            lm_flat, cfg.img_size, cfg.crop_margin,
        )
        true_2d = tf.reshape(lm_norm, [NUM_LANDMARKS, 2])

        coords_base = predict_scaled(model, crop, 1.0)
        nmes_baseline.append(compute_nme(coords_base, true_2d))

        coords_flip = predict_scaled_flipped(model, crop, 1.0, flip_idx)
        nmes_flip.append(compute_nme((coords_base + coords_flip) / 2.0, true_2d))

        preds = [predict_scaled(model, crop, s) for s in SCALES]
        preds += [predict_scaled_flipped(model, crop, s, flip_idx) for s in SCALES]
        nmes_ms_flip.append(compute_nme(tf.reduce_mean(tf.stack(preds), axis=0), true_2d))

        if (i + 1) % 120 == 0:
            print(f"    [{i+1}/{len(val_records)}]  base={np.mean(nmes_baseline):.2f}  flip={np.mean(nmes_flip):.2f}  ms+flip={np.mean(nmes_ms_flip):.2f}  [{time.time()-t0:.0f}s]")

    return {
        "baseline": float(np.mean(nmes_baseline)),
        "flip_tta": float(np.mean(nmes_flip)),
        "ms_flip_tta": float(np.mean(nmes_ms_flip)),
        "elapsed": time.time() - t0,
    }


def evaluate_ensemble(models_info, val_records, tag):
    flip_idx = tf.constant(FLIP_INDEX, dtype=tf.int32)
    num_passes = len(models_info) * len(SCALES) * 2
    print(f"\n  [{tag}]  {len(models_info)} models x {len(SCALES)} scales x 2 flips = {num_passes} passes")

    nmes = []
    region_nmes_all = {r: [] for r in REGIONS}
    t0 = time.time()

    for i, rec in enumerate(val_records):
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        bbox = tf.constant(rec.bbox_xyxy_abs, tf.float32)
        lm_flat = tf.constant([c for pt in rec.landmarks_abs for c in pt], dtype=tf.float32)

        # GT from first model's crop (all share same lm_margin)
        ref_cfg = models_info[0]["cfg"]
        _, lm_norm = crop_and_normalize(image, bbox, lm_flat, ref_cfg.img_size, ref_cfg.crop_margin)
        true_2d = tf.reshape(lm_norm, [NUM_LANDMARKS, 2])

        all_preds = []
        for info in models_info:
            cfg = info["cfg"]
            crop, _ = crop_and_normalize(image, bbox, tf.zeros([NUM_LANDMARKS * 2]), cfg.img_size, cfg.crop_margin)
            for s in SCALES:
                all_preds.append(predict_scaled(info["model"], crop, s))
                all_preds.append(predict_scaled_flipped(info["model"], crop, s, flip_idx))

        avg_pred = tf.reduce_mean(tf.stack(all_preds), axis=0)
        nmes.append(compute_nme(avg_pred, true_2d))

        region_nme = compute_per_region_nme(avg_pred, true_2d)
        for r, v in region_nme.items():
            region_nmes_all[r].append(v)

        if (i + 1) % 120 == 0:
            print(f"    [{i+1}/{len(val_records)}]  NME={np.mean(nmes):.2f}  [{time.time()-t0:.0f}s]")

    return float(np.mean(nmes)), time.time() - t0, {r: float(np.mean(v)) for r, v in region_nmes_all.items()}


def main():
    configure_ca_bundle()
    set_seed(42)

    print("=" * 72)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 72)

    all_records = load_all_records(DATA_ROOT, lm_margin=0.05)
    _, val_records = split_records(all_records, test_fraction=0.15, seed=42)
    print(f"Val records: {len(val_records)}")

    # Load available models
    loaded = {}
    for mc in MODELS_CONFIG:
        if not mc["path"].exists():
            print(f"  SKIP {mc['name']}: {mc['path']} not found")
            continue
        print(f"  Loading {mc['name']}...")
        model = load_model(mc["path"])
        cfg = copy.deepcopy(EXPERIMENT_PRESETS[mc["preset"]])
        loaded[mc["name"]] = {"name": mc["name"], "model": model, "cfg": cfg}
        print(f"    OK (img_size={cfg.img_size})")

    if not loaded:
        print("No models found!")
        return

    results = []

    # Part 1: Single model evaluations
    print("\n" + "=" * 72)
    print("PART 1: SINGLE MODEL EVALUATION")
    print("=" * 72)

    for name, info in loaded.items():
        r = evaluate_single_model(info["model"], val_records, info["cfg"], name)
        results.append({"label": f"{name} (no TTA)", "nme": r["baseline"]})
        results.append({"label": f"{name} (flip TTA)", "nme": r["flip_tta"]})
        results.append({"label": f"{name} (ms+flip TTA)", "nme": r["ms_flip_tta"]})
        print(f"\n  {name}: baseline={r['baseline']:.2f}  flip={r['flip_tta']:.2f}  ms+flip={r['ms_flip_tta']:.2f}")

    # Part 2: All pair ensembles
    if len(loaded) >= 2:
        print("\n" + "=" * 72)
        print("PART 2: PAIR ENSEMBLES (ms+flip TTA)")
        print("=" * 72)

        names = list(loaded.keys())
        for a, b in itertools.combinations(names, 2):
            pair = [loaded[a], loaded[b]]
            nme, elapsed, regions = evaluate_ensemble(pair, val_records, f"{a} + {b}")
            results.append({"label": f"ensemble({a}+{b}) ms+flip", "nme": nme})
            print(f"\n  {a} + {b}: NME={nme:.2f}  [{elapsed:.0f}s]")

    # Part 3: Full ensemble
    if len(loaded) >= 3:
        print("\n" + "=" * 72)
        print("PART 3: FULL ENSEMBLE (ms+flip TTA)")
        print("=" * 72)

        all_info = list(loaded.values())
        nme, elapsed, regions = evaluate_ensemble(all_info, val_records, "all models")
        results.append({"label": f"ensemble(all {len(all_info)}) ms+flip", "nme": nme})
        print(f"\n  All {len(all_info)} models: NME={nme:.2f}  [{elapsed:.0f}s]")
        print(f"\n  Per-region NME:")
        for r, v in sorted(regions.items(), key=lambda x: -x[1]):
            print(f"    {r:<16s} {v:.2f}")

    # Summary
    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    print(f"  {'Configuration':<52} {'NME_IOD':>8}")
    print("  " + "-" * 62)

    for r in sorted(results, key=lambda x: x["nme"]):
        print(f"  {r['label']:<52} {r['nme']:>8.2f}")

    best = min(results, key=lambda r: r["nme"])
    print(f"\n  BEST: {best['label']}  NME={best['nme']:.2f}")


if __name__ == "__main__":
    main()
