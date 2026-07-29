"""Accuracy harness for verifying the cat landmark static re-export.

Mirrors the dog repo's pareto_harness.py. Its only job here is to prove that the
static-shape re-export changes nothing about the predictions, so it needs to
reproduce the exact val split and crop geometry the model was trained against:

  * CatFLW is a single directory, not a train/test layout. The split is a
    deterministic seeded shuffle with test_fraction=0.15, reproduced by calling
    the training script's own `load_all_records` and `split_records`.
  * 48 landmarks (not the dogs' 46), IOD taken between landmarks 4 and 8.

Reports both the crop-space NME_IOD (the training metric) and the absolute
image-pixel NME_IOD (what cat_detection's CHANGELOG publishes).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

CACHE = Path("/private/tmp/claude-501/-Users-hugocornellier-IdeaProjects-dog-detection"
             "/ee61cbfd-dd84-4c4a-82ea-cb1141e124d6/scratchpad/catcache")

NUM_LANDMARKS = 48
LEFT_OUTER_EYE_IDX = 4
RIGHT_OUTER_EYE_IDX = 8


def build_cache(img_size=384, lm_margin=0.05, crop_margin=0.10, seed=42,
                test_fraction=0.15):
    import tensorflow as tf
    import train_cat_face_landmarks as T

    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"{img_size}_{lm_margin}_{crop_margin}_{seed}"
    crops_p = CACHE / f"crops_{tag}.npy"
    if crops_p.exists():
        return crops_p, tag

    data_root = Path(
        "/Users/hugocornellier/.cache/kagglehub/datasets/georgemartvel/catflw"
        "/versions/2/CatFLW dataset"
    )
    all_records = T.load_all_records(data_root, lm_margin)
    _, val_records = T.split_records(all_records, test_fraction, seed)
    n = len(val_records)

    crops = np.lib.format.open_memmap(crops_p, mode="w+", dtype=np.float32,
                                      shape=(n, img_size, img_size, 3))
    gt = np.zeros((n, NUM_LANDMARKS * 2), dtype=np.float32)
    boxes = np.zeros((n, 4), dtype=np.float32)
    gt_abs = np.zeros((n, NUM_LANDMARKS * 2), dtype=np.float32)

    for i, rec in enumerate(val_records):
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        lm_flat = tf.constant([c for pt in rec.landmarks_abs for c in pt],
                              dtype=tf.float32)
        crop, lm_norm = T.crop_and_normalize(
            image, tf.constant(rec.bbox_xyxy_abs, tf.float32), lm_flat,
            img_size, crop_margin,
        )
        crops[i] = crop.numpy()
        gt[i] = lm_norm.numpy()

        iw, ih = rec.orig_size_wh
        x1, y1, x2, y2 = rec.bbox_xyxy_abs
        mx, my = (x2 - x1) * crop_margin, (y2 - y1) * crop_margin
        cx1i = int(np.floor(max(0.0, x1 - mx)))
        cy1i = int(np.floor(max(0.0, y1 - my)))
        cx2i = min(int(np.ceil(min(float(iw), x2 + mx))), iw)
        cy2i = min(int(np.ceil(min(float(ih), y2 + my))), ih)
        boxes[i] = (cx1i, cy1i, max(cx2i - cx1i, 1), max(cy2i - cy1i, 1))
        gt_abs[i] = np.asarray(rec.landmarks_abs, dtype=np.float32).reshape(-1)

        if (i + 1) % 100 == 0:
            print(f"  cached {i + 1}/{n}")

    crops.flush()
    np.save(CACHE / f"gt_{tag}.npy", gt)
    np.save(CACHE / f"boxes_{tag}.npy", boxes)
    np.save(CACHE / f"gtabs_{tag}.npy", gt_abs)
    print(f"Cached {n} CatFLW val crops")
    return crops_p, tag


def load_cache(**kw):
    crops_p, tag = build_cache(**kw)
    return (np.load(crops_p, mmap_mode="r"),
            np.load(CACHE / f"gt_{tag}.npy"),
            np.load(CACHE / f"boxes_{tag}.npy"),
            np.load(CACHE / f"gtabs_{tag}.npy"))


def _per_lm(gt, pred):
    g = gt.reshape(-1, NUM_LANDMARKS, 2).astype(np.float64)
    p = pred.reshape(-1, NUM_LANDMARKS, 2).astype(np.float64)
    iod = np.sqrt(np.sum(
        (g[:, LEFT_OUTER_EYE_IDX] - g[:, RIGHT_OUTER_EYE_IDX]) ** 2, axis=-1) + 1e-8)
    dist = np.sqrt(np.sum((p - g) ** 2, axis=-1) + 1e-8)
    return dist / np.maximum(iod, 1e-8)[:, None] * 100.0


def nme_iod(gt, pred):
    per_sample = _per_lm(gt, pred).mean(axis=1)
    return float(per_sample.mean()), per_sample


def nme_iod_abs(pred_norm, boxes, gt_abs):
    p = pred_norm.reshape(-1, NUM_LANDMARKS, 2).astype(np.float64)
    g = gt_abs.reshape(-1, NUM_LANDMARKS, 2).astype(np.float64)
    x0, y0, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    pa = np.stack([p[:, :, 0] * bw[:, None] + x0[:, None],
                   p[:, :, 1] * bh[:, None] + y0[:, None]], axis=-1)
    iod = np.sqrt(np.sum(
        (g[:, LEFT_OUTER_EYE_IDX] - g[:, RIGHT_OUTER_EYE_IDX]) ** 2, axis=-1) + 1e-8)
    dist = np.sqrt(np.sum((pa - g) ** 2, axis=-1) + 1e-8)
    return float((dist / np.maximum(iod, 1e-8)[:, None] * 100.0).mean(axis=1).mean())


def paired_delta(gt, a, b):
    da = _per_lm(gt, a).mean(axis=1)
    db = _per_lm(gt, b).mean(axis=1)
    d = db - da
    sem = d.std(ddof=1) / np.sqrt(len(d))
    return {"mean_delta": float(d.mean()), "sem": float(sem),
            "t": float(d.mean() / sem) if sem > 0 else 0.0,
            "n_better": int((d < 0).sum()), "n_worse": int((d > 0).sum())}
