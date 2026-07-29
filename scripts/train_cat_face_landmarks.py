#!/usr/bin/env python3
"""Train a cat facial landmark regressor on CatFLW and export TFLite.

Two-stage inference pipeline:
  1) Run the face bbox model  (train_cat_face_detector.py)  -> face bbox in original coords
  2) Crop + resize to landmark model input size (--img-size, default 128)
  3) Run this landmark model  -> 48 (x, y) pairs normalized in [0, 1] relative to crop
  4) Map back to original image coordinates

CatFLW provides 48 facial landmarks per image (eyes, nose, mouth contour).
Ground-truth boxes used for cropping at train time are derived from landmarks
with a margin (--crop-margin).  At inference time the predicted bbox from
stage-1 is used with the same margin.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

NUM_LANDMARKS = 48

# Outer eye corner landmark indices for IOD normalization.
# Landmark 4 = outer corner of right eye in image = left eye of cat.
# Landmark 8 = outer corner of left eye in image = right eye of cat.
LEFT_OUTER_EYE_IDX = 4
RIGHT_OUTER_EYE_IDX = 8

# Landmark index permutation for horizontal flip (CatFLW label convention).
FLIP_INDEX = [
    0, 3, 2, 1, 8, 9, 10, 11,
    4, 5, 6, 7, 13, 12, 15, 14,
    16, 17, 20, 21, 18, 19, 31, 30,
    29, 28, 27, 26, 25, 24, 23, 22,
    35, 34, 33, 32, 41, 40, 39, 38,
    37, 36, 43, 42, 45, 44, 47, 46,
]


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """All tuneable experiment knobs in one place."""
    name: str = "default"

    # Architecture
    backbone: str = "efficientnetb2"  # "efficientnetb2", "efficientnetv2s", or "densenet121"
    head_type: str = "dense"          # "dense" (GAP+FC) or "heatmap" (deconv+soft-argmax)
    heatmap_channels: int = 256       # intermediate channels in deconv head
    heatmap_dropout: float = 0.0      # SpatialDropout2D rate in deconv head


    # Training schedule
    epochs: int = 300
    finetune_epochs: int = 0        # 0 = single-phase training
    batch_size: int = 16
    learning_rate: float = 1e-3
    finetune_learning_rate: float = 1e-4
    finetune_last_layers: int = 120
    lr_schedule: str = "constant"   # "constant" or "cosine"
    lr_min: float = 1e-6

    # Loss
    loss: str = "mse"               # "mse" or "wing"
    wing_omega: float = 0.1
    wing_epsilon: float = 0.02

    # Per-landmark loss weights (None = equal weights)
    # Expects a list of 48 floats. Useful for upweighting hard landmarks (ears).
    landmark_weights: list | None = None

    # Optimizer
    optimizer: str = "adam"         # "adam" or "adamw"
    weight_decay: float = 1e-4

    # Regularisation
    use_swa: bool = False
    swa_start_frac: float = 0.5

    # Augmentations
    aug_rotation: bool = True
    aug_rotation_deg: float = 15.0
    aug_crop_jitter: bool = False
    aug_crop_jitter_frac: float = 0.05
    aug_brightness: bool = True
    aug_brightness_delta: float = 0.10
    aug_contrast: bool = True
    aug_contrast_range: tuple = (0.80, 1.20)
    aug_saturation: bool = True
    aug_saturation_range: tuple = (0.80, 1.20)
    aug_color_balance: bool = False
    aug_sharpness: bool = False
    aug_blur: bool = False
    aug_noise: bool = False
    aug_flip: bool = False
    aug_scale: bool = False
    aug_scale_range: tuple = (0.85, 1.15)

    # Mixup augmentation
    aug_mixup: bool = False
    aug_mixup_alpha: float = 0.2      # Beta distribution parameter
    aug_mixup_prob: float = 0.4       # probability of applying mixup

    # Random erasing augmentation
    aug_random_erase: bool = False
    aug_random_erase_prob: float = 0.25
    aug_random_erase_area_low: float = 0.02
    aug_random_erase_area_high: float = 0.12

    # Heatmap supervision (hybrid loss with Gaussian targets)
    heatmap_supervision: bool = False
    pure_heatmap_supervision: bool = False  # True = heatmap MSE only, no coord loss
    heatmap_sigma: float = 1.75       # Gaussian sigma in heatmap pixels
    coord_loss_weight: float = 0.25   # weight for coord MSE in hybrid mode
    num_deconv_layers: int = 3        # 2 -> 28x28, 3 -> 56x56, 4 -> 112x112
    per_landmark_heatmap_loss: bool = False  # normalize heatmap MSE per-landmark (fixes background domination)

    # SoftArgmax2D temperature
    softargmax_beta: float = 1.0      # temperature parameter (higher = sharper)

    # Data
    img_size: int = 224
    crop_margin: float = 0.20
    lm_margin: float = 0.12
    square_pad: bool = False          # True = pad shorter side to square (preserve aspect ratio)
    use_dataset_bbox: bool = False    # True = use annotation bounding_boxes instead of landmark-derived

    # Metric used for early stopping / model selection
    nme_mode: str = "iod"           # "crop" or "iod"

    # Early stopping
    patience: int = 50

    unfreeze_backbone: bool = False  # True = full fine-tuning; False = head-only (paper default)

    seed: int = 42
    no_pretrained: bool = False


EXPERIMENT_PRESETS: dict[str, ExperimentConfig] = {
    # --- Tight margin experiments (proven best recipes) ---
    "tight_margin": ExperimentConfig(
        name="tight_margin",
        backbone="efficientnetv2s",
        head_type="heatmap",
        heatmap_dropout=0.1,
        num_deconv_layers=4,
        epochs=100,
        finetune_epochs=200,
        finetune_learning_rate=1e-5,
        finetune_last_layers=50,
        batch_size=16,
        learning_rate=1e-4,
        lr_schedule="constant",
        loss="mse",
        optimizer="adamw",
        weight_decay=1e-4,
        use_swa=False,
        lm_margin=0.05,
        crop_margin=0.10,
        aug_rotation=True,
        aug_flip=True,
        aug_crop_jitter=True,
        aug_crop_jitter_frac=0.08,
        aug_scale=True,
        aug_brightness=True,
        aug_contrast=True,
        aug_saturation=True,
        aug_color_balance=True,
        aug_sharpness=True,
        aug_blur=True,
        aug_noise=True,
        nme_mode="iod",
        patience=50,
    ),
    # --- 256x256 input resolution ---
    "tight_margin_256": ExperimentConfig(
        name="tight_margin_256",
        backbone="efficientnetv2s",
        head_type="heatmap",
        heatmap_dropout=0.1,
        num_deconv_layers=4,
        epochs=100,
        finetune_epochs=200,
        finetune_learning_rate=1e-5,
        finetune_last_layers=50,
        batch_size=16,
        learning_rate=1e-4,
        lr_schedule="constant",
        loss="mse",
        optimizer="adamw",
        weight_decay=1e-4,
        use_swa=False,
        img_size=256,
        lm_margin=0.05,
        crop_margin=0.10,
        aug_rotation=True,
        aug_rotation_deg=15.0,
        aug_flip=True,
        aug_crop_jitter=True,
        aug_crop_jitter_frac=0.08,
        aug_scale=True,
        aug_brightness=True,
        aug_contrast=True,
        aug_saturation=True,
        aug_color_balance=True,
        aug_sharpness=True,
        aug_blur=True,
        aug_noise=True,
        nme_mode="iod",
        patience=50,
    ),
    # --- 320x320 resolution ---
    "tight_margin_320": ExperimentConfig(
        name="tight_margin_320",
        backbone="efficientnetv2s",
        head_type="heatmap",
        heatmap_dropout=0.1,
        num_deconv_layers=4,
        epochs=100,
        finetune_epochs=200,
        finetune_learning_rate=1e-5,
        finetune_last_layers=50,
        batch_size=8,  # smaller batch for memory
        learning_rate=1e-4,
        lr_schedule="constant",
        loss="mse",
        optimizer="adamw",
        weight_decay=1e-4,
        use_swa=False,
        img_size=320,
        lm_margin=0.05,
        crop_margin=0.10,
        aug_rotation=True,
        aug_rotation_deg=15.0,
        aug_flip=True,
        aug_crop_jitter=True,
        aug_crop_jitter_frac=0.08,
        aug_scale=True,
        aug_brightness=True,
        aug_contrast=True,
        aug_saturation=True,
        aug_color_balance=True,
        aug_sharpness=True,
        aug_blur=True,
        aug_noise=True,
        nme_mode="iod",
        patience=50,
    ),
    # --- 384x384 resolution ---
    "tight_margin_384": ExperimentConfig(
        name="tight_margin_384",
        backbone="efficientnetv2s",
        head_type="heatmap",
        heatmap_dropout=0.1,
        num_deconv_layers=4,
        epochs=100,
        finetune_epochs=200,
        finetune_learning_rate=1e-5,
        finetune_last_layers=50,
        batch_size=8,  # smaller batch for memory
        learning_rate=1e-4,
        lr_schedule="constant",
        loss="mse",
        optimizer="adamw",
        weight_decay=1e-4,
        use_swa=False,
        img_size=384,
        lm_margin=0.05,
        crop_margin=0.10,
        aug_rotation=True,
        aug_rotation_deg=15.0,
        aug_flip=True,
        aug_crop_jitter=True,
        aug_crop_jitter_frac=0.08,
        aug_scale=True,
        aug_brightness=True,
        aug_contrast=True,
        aug_saturation=True,
        aug_color_balance=True,
        aug_sharpness=True,
        aug_blur=True,
        aug_noise=True,
        nme_mode="iod",
        patience=50,
    ),
    # --- Small models (MobileNetV3) ---
    "small_v3large_256": ExperimentConfig(
        name="small_v3large_256",
        backbone="mobilenetv3large",
        head_type="heatmap",
        heatmap_channels=128,
        heatmap_dropout=0.1,
        num_deconv_layers=4,
        epochs=100,
        finetune_epochs=200,
        finetune_learning_rate=1e-5,
        finetune_last_layers=40,
        batch_size=16,
        learning_rate=1e-4,
        lr_schedule="constant",
        loss="mse",
        optimizer="adamw",
        weight_decay=1e-4,
        use_swa=False,
        img_size=256,
        lm_margin=0.05,
        crop_margin=0.10,
        aug_rotation=True,
        aug_rotation_deg=15.0,
        aug_flip=True,
        aug_crop_jitter=True,
        aug_crop_jitter_frac=0.08,
        aug_scale=True,
        aug_brightness=True,
        aug_contrast=True,
        aug_saturation=True,
        aug_color_balance=True,
        aug_sharpness=True,
        aug_blur=True,
        aug_noise=True,
        nme_mode="iod",
        patience=50,
    ),
    "small_v3large_384": ExperimentConfig(
        name="small_v3large_384",
        backbone="mobilenetv3large",
        head_type="heatmap",
        heatmap_channels=128,
        heatmap_dropout=0.1,
        num_deconv_layers=4,
        epochs=100,
        finetune_epochs=300,
        finetune_learning_rate=1e-5,
        finetune_last_layers=40,
        batch_size=8,
        learning_rate=1e-4,
        lr_schedule="constant",
        loss="mse",
        optimizer="adamw",
        weight_decay=1e-4,
        use_swa=False,
        img_size=384,
        lm_margin=0.05,
        crop_margin=0.10,
        aug_rotation=True,
        aug_rotation_deg=15.0,
        aug_flip=True,
        aug_crop_jitter=True,
        aug_crop_jitter_frac=0.08,
        aug_scale=True,
        aug_brightness=True,
        aug_contrast=True,
        aug_saturation=True,
        aug_color_balance=True,
        aug_sharpness=True,
        aug_blur=True,
        aug_noise=True,
        nme_mode="iod",
        patience=50,
    ),
    "small_v3small_256": ExperimentConfig(
        name="small_v3small_256",
        backbone="mobilenetv3small",
        head_type="heatmap",
        heatmap_channels=128,
        heatmap_dropout=0.1,
        num_deconv_layers=4,
        epochs=100,
        finetune_epochs=200,
        finetune_learning_rate=1e-5,
        finetune_last_layers=30,
        batch_size=16,
        learning_rate=1e-4,
        lr_schedule="constant",
        loss="mse",
        optimizer="adamw",
        weight_decay=1e-4,
        use_swa=False,
        img_size=256,
        lm_margin=0.05,
        crop_margin=0.10,
        aug_rotation=True,
        aug_rotation_deg=15.0,
        aug_flip=True,
        aug_crop_jitter=True,
        aug_crop_jitter_frac=0.08,
        aug_scale=True,
        aug_brightness=True,
        aug_contrast=True,
        aug_saturation=True,
        aug_color_balance=True,
        aug_sharpness=True,
        aug_blur=True,
        aug_noise=True,
        nme_mode="iod",
        patience=50,
    ),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Record:
    image_path: str
    bbox_xyxy_abs: tuple[float, float, float, float]   # landmark-derived GT box
    landmarks_abs: tuple                                # ((x0,y0), ..., (x47,y47))
    orig_size_wh: tuple[int, int]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_root = (
        Path.home() / ".cache" / "kagglehub" / "datasets"
        / "georgemartvel" / "catflw" / "versions" / "2" / "CatFLW dataset"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=default_root)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/cat_face_landmarks"))
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--finetune-epochs", type=int, default=None)
    p.add_argument("--finetune-last-layers", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--finetune-learning-rate", type=float, default=None)
    p.add_argument("--lm-margin", type=float, default=None)
    p.add_argument("--crop-margin", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true", default=False)
    p.add_argument("--skip-finetune", action="store_true")
    p.add_argument("--tflite-only", action="store_true")
    p.add_argument("--test-fraction", type=float, default=0.15,
                   help="Fraction of data to hold out for testing (default 0.15)")

    # --- Experiment framework flags ---
    p.add_argument("--experiment", type=str, default=None,
                   help="Named experiment preset: 'tight_margin', 'tight_margin_256', etc.")
    p.add_argument("--loss", choices=["mse", "wing"], default=None)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default=None)
    p.add_argument("--use-swa", action="store_true", default=None)
    p.add_argument("--no-swa", dest="use_swa", action="store_false")
    p.add_argument("--swa-start-frac", type=float, default=None,
                   help="Fraction of epochs after which SWA averaging starts (default 0.5)")
    p.add_argument("--nme-mode", choices=["crop", "iod"], default=None)
    p.add_argument("--backbone", choices=["efficientnetb2", "efficientnetv2s", "densenet121", "mobilenetv3large", "mobilenetv3small"], default=None)
    p.add_argument("--head-type", choices=["dense", "heatmap"], default=None)
    p.add_argument("--unfreeze-backbone", action="store_true", default=None)
    p.add_argument("--no-unfreeze-backbone", dest="unfreeze_backbone", action="store_false")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--aug-blur", action="store_true", default=None)
    p.add_argument("--no-aug-blur", dest="aug_blur", action="store_false")
    p.add_argument("--aug-noise", action="store_true", default=None)
    p.add_argument("--no-aug-noise", dest="aug_noise", action="store_false")
    p.add_argument("--aug-sharpness", action="store_true", default=None)
    p.add_argument("--no-aug-sharpness", dest="aug_sharpness", action="store_false")
    p.add_argument("--aug-color-balance", action="store_true", default=None)
    p.add_argument("--no-aug-color-balance", dest="aug_color_balance", action="store_false")
    p.add_argument("--aug-crop-jitter", action="store_true", default=None)
    p.add_argument("--no-aug-crop-jitter", dest="aug_crop_jitter", action="store_false")
    p.add_argument("--aug-flip", action="store_true", default=None)
    p.add_argument("--no-aug-flip", dest="aug_flip", action="store_false")
    p.add_argument("--aug-scale", action="store_true", default=None)
    p.add_argument("--no-aug-scale", dest="aug_scale", action="store_false")
    p.add_argument("--optimizer", choices=["adam", "adamw"], default=None)
    p.add_argument("--heatmap-supervision", action="store_true", default=None)
    p.add_argument("--no-heatmap-supervision", dest="heatmap_supervision", action="store_false")
    p.add_argument("--heatmap-sigma", type=float, default=None)
    p.add_argument("--coord-loss-weight", type=float, default=None)
    p.add_argument("--num-deconv-layers", type=int, default=None)
    p.add_argument("--square-pad", action="store_true", default=None,
                   help="Pad crop to square before resize (preserve aspect ratio)")
    p.add_argument("--no-square-pad", dest="square_pad", action="store_false")
    p.add_argument("--use-dataset-bbox", action="store_true", default=None,
                   help="Use annotation bounding_boxes instead of landmark-derived")
    p.add_argument("--no-use-dataset-bbox", dest="use_dataset_bbox", action="store_false")
    return p.parse_args()


def resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build an ExperimentConfig from a preset + CLI overrides."""
    if args.experiment:
        if args.experiment in EXPERIMENT_PRESETS:
            cfg = copy.deepcopy(EXPERIMENT_PRESETS[args.experiment])
        else:
            raise ValueError(f"Unknown experiment preset: {args.experiment}")
    else:
        # Default: tight_margin
        cfg = copy.deepcopy(EXPERIMENT_PRESETS["tight_margin"])

    # Apply explicit CLI overrides (only if not None).
    _override = {
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "finetune_epochs": args.finetune_epochs,
        "finetune_last_layers": args.finetune_last_layers,
        "learning_rate": args.learning_rate,
        "finetune_learning_rate": args.finetune_learning_rate,
        "lm_margin": args.lm_margin,
        "crop_margin": args.crop_margin,
        "seed": args.seed,
        "no_pretrained": args.no_pretrained or None,
        "loss": args.loss,
        "lr_schedule": args.lr_schedule,
        "use_swa": args.use_swa,
        "swa_start_frac": args.swa_start_frac,
        "nme_mode": args.nme_mode,
        "backbone": args.backbone,
        "head_type": args.head_type,
        "unfreeze_backbone": args.unfreeze_backbone,
        "patience": args.patience,
        "aug_blur": args.aug_blur,
        "aug_noise": args.aug_noise,
        "aug_sharpness": args.aug_sharpness,
        "aug_color_balance": args.aug_color_balance,
        "aug_crop_jitter": args.aug_crop_jitter,
        "aug_flip": args.aug_flip,
        "aug_scale": args.aug_scale,
        "optimizer": args.optimizer,
        "heatmap_supervision": args.heatmap_supervision,
        "heatmap_sigma": args.heatmap_sigma,
        "coord_loss_weight": args.coord_loss_weight,
        "num_deconv_layers": args.num_deconv_layers,
        "square_pad": args.square_pad,
        "use_dataset_bbox": args.use_dataset_bbox,
    }
    for key, val in _override.items():
        if val is not None:
            setattr(cfg, key, val)

    # Legacy flag compat
    if args.skip_finetune:
        cfg.finetune_epochs = 0

    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def configure_ca_bundle() -> None:
    try:
        import certifi
    except Exception:
        return
    ca = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)


# ---------------------------------------------------------------------------
# Dataset loading
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
        xs.append(float(np.clip(pt[0], 0.0, img_w)))
        ys.append(float(np.clip(pt[1], 0.0, img_h)))
    if not xs:
        return None
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return None
    x1 = max(0.0, x1 - bw * margin)
    y1 = max(0.0, y1 - bh * margin)
    x2 = min(float(img_w), x2 + bw * margin)
    y2 = min(float(img_h), y2 + bh * margin)
    if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
        return None
    return (x1, y1, x2, y2)


def load_all_records(data_root: Path, lm_margin: float,
                     use_dataset_bbox: bool = False) -> list[Record]:
    """Load all CatFLW records from the flat directory structure.

    CatFLW has no train/test split — images/ and labels/ contain all samples.
    If use_dataset_bbox=True, uses annotation bounding_boxes instead of
    landmark-derived boxes.
    """
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
        # Reject records where any landmark coordinate is NaN.
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
        if use_dataset_bbox:
            raw_bb = ann.get("bounding_boxes")
            if raw_bb and len(raw_bb) == 4:
                x1, y1, x2, y2 = float(raw_bb[0]), float(raw_bb[1]), float(raw_bb[2]), float(raw_bb[3])
                x1 = max(0.0, min(x1, float(img_w)))
                y1 = max(0.0, min(y1, float(img_h)))
                x2 = max(0.0, min(x2, float(img_w)))
                y2 = max(0.0, min(y2, float(img_h)))
                if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
                    bbox = _landmark_bbox(landmarks, img_w, img_h, lm_margin)
                else:
                    bbox = (x1, y1, x2, y2)
            else:
                bbox = _landmark_bbox(landmarks, img_w, img_h, lm_margin)
        else:
            bbox = _landmark_bbox(landmarks, img_w, img_h, lm_margin)
        if bbox is None:
            skipped += 1
            continue
        lm_clean = tuple(
            (float(np.clip(pt[0], 0.0, img_w)), float(np.clip(pt[1], 0.0, img_h)))
            for pt in landmarks
        )
        records.append(Record(
            image_path=str(img_path),
            bbox_xyxy_abs=bbox,
            landmarks_abs=lm_clean,
            orig_size_wh=(img_w, img_h),
        ))
    print(f"[all] usable={len(records)} skipped={skipped}")
    return records


def split_records(
    records: list[Record], test_fraction: float, seed: int,
) -> tuple[list[Record], list[Record]]:
    """Deterministically split records into train and test sets."""
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
# TF dataset pipeline
# ---------------------------------------------------------------------------

def _mixup_batch(images, targets, alpha=0.2, prob=0.4, is_heatmap=False):
    """Apply mixup to a batch of images and targets.

    For coordinate targets: linear interpolation.
    For heatmap targets: linear interpolation of Gaussian heatmaps.
    For dict targets (heatmap supervision): interpolate both.
    """
    batch_size = tf.shape(images)[0]
    do_mixup = tf.random.uniform(()) < prob

    # Sample mixup coefficient from Beta(alpha, alpha)
    # Use uniform approximation for simplicity in tf.data
    lam = tf.random.uniform((), 0.0, 1.0)
    # Clamp to keep lambda > 0.5 (keep original image dominant)
    lam = tf.maximum(lam, 1.0 - lam) if alpha < 0.5 else lam

    # Shuffle indices for mixing pairs
    indices = tf.random.shuffle(tf.range(batch_size))
    images_shuffled = tf.gather(images, indices)

    mixed_images = lam * images + (1.0 - lam) * images_shuffled

    if isinstance(targets, dict):
        targets_shuffled = {k: tf.gather(v, indices) for k, v in targets.items()}
        mixed_targets = {k: lam * v + (1.0 - lam) * targets_shuffled[k] for k, v in targets.items()}
    else:
        targets_shuffled = tf.gather(targets, indices)
        mixed_targets = lam * targets + (1.0 - lam) * targets_shuffled

    mixed_images = tf.where(do_mixup, mixed_images, images)
    if isinstance(targets, dict):
        mixed_targets = {k: tf.where(do_mixup, mixed_targets[k], targets[k]) for k in targets}
    else:
        mixed_targets = tf.where(do_mixup, mixed_targets, targets)

    return mixed_images, mixed_targets


def build_tf_dataset(
    records: list[Record],
    cfg: ExperimentConfig,
    training: bool,
) -> tf.data.Dataset:
    paths = np.array([r.image_path for r in records], dtype=object)
    boxes = np.array([r.bbox_xyxy_abs for r in records], dtype=np.float32)
    lmarks = np.array(
        [[coord for pt in r.landmarks_abs for coord in pt] for r in records],
        dtype=np.float32,
    )  # shape [N, 96]

    ds = tf.data.Dataset.from_tensor_slices((paths, boxes, lmarks))
    if training:
        ds = ds.shuffle(len(records), seed=cfg.seed, reshuffle_each_iteration=True)

    img_size = cfg.img_size
    crop_margin = cfg.crop_margin
    sq_pad = cfg.square_pad
    hm_size = (img_size // 32) * (2 ** cfg.num_deconv_layers)  # 56 or 112

    def _load_and_crop(
        path: tf.Tensor,
        box_abs: tf.Tensor,
        lm_flat: tf.Tensor,
    ):
        img_bytes = tf.io.read_file(path)
        image = tf.io.decode_png(img_bytes, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        box_aug = box_abs
        if training and cfg.aug_scale:
            box_aug = scale_crop_box(box_aug, image, cfg.aug_scale_range)
        if training and cfg.aug_crop_jitter:
            box_aug = jitter_crop_box(box_aug, image, cfg.aug_crop_jitter_frac)
        crop, lm_norm = crop_and_normalize(image, box_aug, lm_flat, img_size, crop_margin, sq_pad)
        if training:
            if cfg.aug_rotation:
                crop, lm_norm = rotate_augment(crop, lm_norm, img_size, cfg.aug_rotation_deg)
            if cfg.aug_flip:
                crop, lm_norm = flip_augment(crop, lm_norm)
            crop = photometric_augment(crop, cfg)
            if cfg.aug_random_erase:
                crop = augment_random_erase(
                    crop, prob=cfg.aug_random_erase_prob,
                    area_low=cfg.aug_random_erase_area_low,
                    area_high=cfg.aug_random_erase_area_high,
                )
        if cfg.heatmap_supervision or cfg.pure_heatmap_supervision:
            hm_targets = generate_gaussian_heatmaps(lm_norm, hm_size, cfg.heatmap_sigma)
            if cfg.pure_heatmap_supervision:
                return crop, hm_targets
            return crop, {"hm": hm_targets, "xy": lm_norm}
        return crop, lm_norm

    ds = ds.map(_load_and_crop, num_parallel_calls=tf.data.AUTOTUNE)

    # Mixup augmentation (operates on batches)
    if training and cfg.aug_mixup:
        ds = ds.batch(cfg.batch_size)
        ds = ds.map(
            lambda x, y: _mixup_batch(x, y, cfg.aug_mixup_alpha, cfg.aug_mixup_prob,
                                       cfg.pure_heatmap_supervision or cfg.heatmap_supervision),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = ds.batch(cfg.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def crop_and_normalize(
    image: tf.Tensor,
    bbox_abs: tf.Tensor,
    lm_flat: tf.Tensor,
    img_size: int,
    crop_margin: float,
    square_pad: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Crop image to bbox + margin, resize to (img_size, img_size).

    Normalizes landmark coordinates relative to the crop window so that
    each value is in [0, 1].  At inference time, call with the predicted bbox.

    If square_pad=True, the crop is padded (not stretched) to a square before
    resizing, preserving the original aspect ratio.
    """
    img_h = tf.cast(tf.shape(image)[0], tf.float32)
    img_w = tf.cast(tf.shape(image)[1], tf.float32)

    x1, y1, x2, y2 = bbox_abs[0], bbox_abs[1], bbox_abs[2], bbox_abs[3]
    bw = x2 - x1
    bh = y2 - y1

    mx = bw * crop_margin
    my = bh * crop_margin
    cx1 = tf.maximum(0.0, x1 - mx)
    cy1 = tf.maximum(0.0, y1 - my)
    cx2 = tf.minimum(img_w, x2 + mx)
    cy2 = tf.minimum(img_h, y2 + my)

    cx1i = tf.cast(tf.math.floor(cx1), tf.int32)
    cy1i = tf.cast(tf.math.floor(cy1), tf.int32)
    cx2i = tf.minimum(tf.cast(tf.math.ceil(cx2), tf.int32), tf.shape(image)[1])
    cy2i = tf.minimum(tf.cast(tf.math.ceil(cy2), tf.int32), tf.shape(image)[0])
    crop_w = tf.maximum(cx2i - cx1i, 1)
    crop_h = tf.maximum(cy2i - cy1i, 1)

    cropped = tf.image.crop_to_bounding_box(image, cy1i, cx1i, crop_h, crop_w)

    if square_pad:
        # Pad shorter side symmetrically to make a square, then resize.
        max_side = tf.maximum(crop_h, crop_w)
        pad_h = max_side - crop_h
        pad_w = max_side - crop_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        cropped = tf.pad(cropped, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        resized = tf.image.resize(cropped, [img_size, img_size], antialias=True)
        resized = tf.cast(tf.clip_by_value(resized, 0.0, 1.0), tf.float32)

        # Normalize landmarks: shift by padding offset, then divide by square side
        cx1f = tf.cast(cx1i, tf.float32)
        cy1f = tf.cast(cy1i, tf.float32)
        max_side_f = tf.cast(max_side, tf.float32)
        pad_left_f = tf.cast(pad_left, tf.float32)
        pad_top_f = tf.cast(pad_top, tf.float32)

        lm = tf.reshape(lm_flat, [NUM_LANDMARKS, 2])  # [48, 2]
        lm_x = (lm[:, 0] - cx1f + pad_left_f) / max_side_f
        lm_y = (lm[:, 1] - cy1f + pad_top_f) / max_side_f
        lm_norm = tf.clip_by_value(tf.stack([lm_x, lm_y], axis=-1), 0.0, 1.0)
        lm_norm_flat = tf.reshape(lm_norm, [NUM_LANDMARKS * 2])
    else:
        resized = tf.image.resize(cropped, [img_size, img_size], antialias=True)
        resized = tf.cast(tf.clip_by_value(resized, 0.0, 1.0), tf.float32)

        # Normalize landmarks relative to crop
        cx1f = tf.cast(cx1i, tf.float32)
        cy1f = tf.cast(cy1i, tf.float32)
        crop_wf = tf.cast(crop_w, tf.float32)
        crop_hf = tf.cast(crop_h, tf.float32)

        lm = tf.reshape(lm_flat, [NUM_LANDMARKS, 2])  # [48, 2]
        lm_x = (lm[:, 0] - cx1f) / crop_wf
        lm_y = (lm[:, 1] - cy1f) / crop_hf
        lm_norm = tf.clip_by_value(tf.stack([lm_x, lm_y], axis=-1), 0.0, 1.0)
        lm_norm_flat = tf.reshape(lm_norm, [NUM_LANDMARKS * 2])

    return resized, lm_norm_flat


def jitter_crop_box(
    bbox_abs: tf.Tensor, image: tf.Tensor, max_frac: float = 0.05
) -> tf.Tensor:
    """Randomly shift the crop box by up to +/-max_frac of its width/height.

    Simulates the imperfect bboxes the landmark model will receive from the
    detector at inference time.
    """
    img_h = tf.cast(tf.shape(image)[0], tf.float32)
    img_w = tf.cast(tf.shape(image)[1], tf.float32)
    x1, y1, x2, y2 = bbox_abs[0], bbox_abs[1], bbox_abs[2], bbox_abs[3]
    bw, bh = x2 - x1, y2 - y1
    dx = tf.random.uniform((), -max_frac, max_frac) * bw
    dy = tf.random.uniform((), -max_frac, max_frac) * bh
    x1 = tf.clip_by_value(x1 + dx, 0.0, img_w)
    y1 = tf.clip_by_value(y1 + dy, 0.0, img_h)
    x2 = tf.clip_by_value(x2 + dx, 0.0, img_w)
    y2 = tf.clip_by_value(y2 + dy, 0.0, img_h)
    return tf.stack([x1, y1, x2, y2])


def scale_crop_box(
    bbox_abs: tf.Tensor, image: tf.Tensor,
    scale_range: tuple = (0.85, 1.15),
) -> tf.Tensor:
    """Randomly scale the crop box to simulate different face sizes."""
    img_h = tf.cast(tf.shape(image)[0], tf.float32)
    img_w = tf.cast(tf.shape(image)[1], tf.float32)
    x1, y1, x2, y2 = bbox_abs[0], bbox_abs[1], bbox_abs[2], bbox_abs[3]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = x2 - x1, y2 - y1
    scale = tf.random.uniform((), scale_range[0], scale_range[1])
    new_bw, new_bh = bw * scale, bh * scale
    x1 = tf.clip_by_value(cx - new_bw / 2.0, 0.0, img_w)
    y1 = tf.clip_by_value(cy - new_bh / 2.0, 0.0, img_h)
    x2 = tf.clip_by_value(cx + new_bw / 2.0, 0.0, img_w)
    y2 = tf.clip_by_value(cy + new_bh / 2.0, 0.0, img_h)
    return tf.stack([x1, y1, x2, y2])


def rotate_augment(
    crop: tf.Tensor, lm_norm_flat: tf.Tensor, img_size: int,
    max_deg: float = 15.0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Random rotation of the crop with corresponding landmark transform."""
    angle_deg = tf.random.uniform((), -max_deg, max_deg)
    angle_rad = angle_deg * (math.pi / 180.0)

    # Rotate image about its centre.
    crop = tf.expand_dims(crop, 0)  # [1, H, W, 3]
    # angles is counter-clockwise in radians
    crop = tf.raw_ops.ImageProjectiveTransformV3(
        images=crop,
        transforms=_rotation_matrix(angle_rad, img_size),
        output_shape=tf.constant([img_size, img_size], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )
    crop = tf.squeeze(crop, 0)
    crop = tf.clip_by_value(crop, 0.0, 1.0)

    # Rotate landmarks (pivot = centre of unit square).
    lm = tf.reshape(lm_norm_flat, [NUM_LANDMARKS, 2])
    cos_a = tf.cos(-angle_rad)
    sin_a = tf.sin(-angle_rad)
    cx = lm[:, 0] - 0.5
    cy = lm[:, 1] - 0.5
    rx = cx * cos_a - cy * sin_a + 0.5
    ry = cx * sin_a + cy * cos_a + 0.5
    lm_rot = tf.clip_by_value(tf.stack([rx, ry], axis=-1), 0.0, 1.0)
    return crop, tf.reshape(lm_rot, [NUM_LANDMARKS * 2])


def _rotation_matrix(angle_rad: tf.Tensor, img_size: int) -> tf.Tensor:
    """Build a [1, 8] projective transform matrix for tf.raw_ops.ImageProjectiveTransformV3."""
    cos_a = tf.cos(angle_rad)
    sin_a = tf.sin(angle_rad)
    half = tf.cast(img_size, tf.float32) / 2.0
    # Translate to origin, rotate, translate back.
    tx = half - half * cos_a + half * sin_a
    ty = half - half * sin_a - half * cos_a
    return tf.expand_dims(
        tf.stack([cos_a, -sin_a, tx, sin_a, cos_a, ty, 0.0, 0.0]), 0
    )


def flip_augment(
    crop: tf.Tensor, lm_norm_flat: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Random horizontal flip with landmark index swapping (50% probability)."""
    do_flip = tf.random.uniform(()) < 0.5
    flip_idx = tf.constant(FLIP_INDEX, dtype=tf.int32)

    flipped_img = tf.image.flip_left_right(crop)
    lm = tf.reshape(lm_norm_flat, [NUM_LANDMARKS, 2])
    lm_flipped = tf.gather(lm, flip_idx, axis=0)
    lm_flipped = tf.stack([1.0 - lm_flipped[:, 0], lm_flipped[:, 1]], axis=-1)
    lm_flipped_flat = tf.reshape(lm_flipped, [NUM_LANDMARKS * 2])

    crop_out = tf.where(do_flip, flipped_img, crop)
    lm_out = tf.where(do_flip, lm_flipped_flat, lm_norm_flat)
    return crop_out, lm_out


def augment_gaussian_blur(image: tf.Tensor) -> tf.Tensor:
    """Random average-pooling blur (50% probability)."""
    img_4d = tf.expand_dims(image, 0)
    blurred = tf.nn.avg_pool2d(img_4d, ksize=3, strides=1, padding="SAME")
    blurred = tf.squeeze(blurred, 0)
    do_blur = tf.random.uniform(()) < 0.5
    return tf.where(do_blur, blurred, image)


def augment_gaussian_noise(image: tf.Tensor, max_stddev: float = 0.05) -> tf.Tensor:
    """Add random Gaussian noise."""
    stddev = tf.random.uniform((), 0.0, max_stddev)
    noise = tf.random.normal(tf.shape(image), stddev=stddev)
    return tf.clip_by_value(image + noise, 0.0, 1.0)


def augment_sharpness(image: tf.Tensor) -> tf.Tensor:
    """Random sharpness alteration via unsharp mask."""
    factor = tf.random.uniform((), 0.5, 2.0)
    img_4d = tf.expand_dims(image, 0)
    blurred = tf.nn.avg_pool2d(img_4d, ksize=3, strides=1, padding="SAME")
    blurred = tf.squeeze(blurred, 0)
    sharpened = image + factor * (image - blurred)
    return tf.clip_by_value(sharpened, 0.0, 1.0)


def augment_color_balance(image: tf.Tensor, max_shift: float = 0.05) -> tf.Tensor:
    """Random per-channel intensity shift."""
    shifts = tf.random.uniform([3], -max_shift, max_shift)
    return tf.clip_by_value(image + shifts, 0.0, 1.0)


def photometric_augment(image: tf.Tensor, cfg: ExperimentConfig) -> tf.Tensor:
    if cfg.aug_brightness:
        image = tf.image.random_brightness(image, max_delta=cfg.aug_brightness_delta)
    if cfg.aug_contrast:
        image = tf.image.random_contrast(
            image, lower=cfg.aug_contrast_range[0], upper=cfg.aug_contrast_range[1])
    if cfg.aug_saturation:
        image = tf.image.random_saturation(
            image, lower=cfg.aug_saturation_range[0], upper=cfg.aug_saturation_range[1])
    if cfg.aug_color_balance:
        image = augment_color_balance(image)
    if cfg.aug_sharpness:
        image = augment_sharpness(image)
    if cfg.aug_blur:
        image = augment_gaussian_blur(image)
    if cfg.aug_noise:
        image = augment_gaussian_noise(image)
    return tf.clip_by_value(image, 0.0, 1.0)


def augment_random_erase(
    image: tf.Tensor, prob: float = 0.25,
    area_low: float = 0.02, area_high: float = 0.12,
) -> tf.Tensor:
    """Random erasing augmentation (Zhong et al. 2020).

    Randomly erases a rectangular region with random pixel values.
    """
    do_erase = tf.random.uniform(()) < prob
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    area = tf.cast(h * w, tf.float32)

    erase_area = tf.random.uniform((), area_low, area_high) * area
    aspect = tf.random.uniform((), 0.3, 1.0 / 0.3)
    eh = tf.cast(tf.math.sqrt(erase_area * aspect), tf.int32)
    ew = tf.cast(tf.math.sqrt(erase_area / aspect), tf.int32)
    eh = tf.minimum(eh, h)
    ew = tf.minimum(ew, w)

    ey = tf.random.uniform((), 0, h - eh + 1, dtype=tf.int32)
    ex = tf.random.uniform((), 0, w - ew + 1, dtype=tf.int32)

    noise = tf.random.uniform([eh, ew, 3], 0.0, 1.0)
    # Create mask
    padding = [[ey, h - ey - eh], [ex, w - ex - ew], [0, 0]]
    mask = tf.pad(tf.ones([eh, ew, 3]), padding)
    noise_padded = tf.pad(noise, padding)

    erased = tf.where(tf.cast(mask, tf.bool), noise_padded, image)
    return tf.where(do_erase, erased, image)


def generate_gaussian_heatmaps(
    lm_norm_flat: tf.Tensor, hm_size: int, sigma: float,
) -> tf.Tensor:
    """Generate 2D Gaussian heatmaps for each landmark.

    Args:
        lm_norm_flat: [96] tensor of normalized coords [x0,y0,x1,y1,...]
        hm_size: heatmap resolution (e.g. 56 or 112)
        sigma: Gaussian sigma in heatmap pixels

    Returns:
        [hm_size, hm_size, 48] tensor of Gaussian heatmaps
    """
    lm = tf.reshape(lm_norm_flat, [NUM_LANDMARKS, 2])  # [48, 2]
    size_f = tf.cast(hm_size - 1, tf.float32)
    mu_x = lm[:, 0] * size_f  # [48]
    mu_y = lm[:, 1] * size_f  # [48]

    grid = tf.cast(tf.range(hm_size), tf.float32)  # [hm_size]
    grid_x = tf.reshape(grid, [1, 1, hm_size])  # [1, 1, W]
    grid_y = tf.reshape(grid, [1, hm_size, 1])  # [1, H, 1]

    mu_x = tf.reshape(mu_x, [NUM_LANDMARKS, 1, 1])  # [48, 1, 1]
    mu_y = tf.reshape(mu_y, [NUM_LANDMARKS, 1, 1])  # [48, 1, 1]

    dx2 = tf.square(grid_x - mu_x)  # [48, 1, W]
    dy2 = tf.square(grid_y - mu_y)  # [48, H, 1]
    heatmaps = tf.exp(-(dx2 + dy2) / (2.0 * sigma * sigma))  # [48, H, W]

    return tf.transpose(heatmaps, [1, 2, 0])  # [H, W, 48]


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------

def wing_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    omega: float = 0.1,
    epsilon: float = 0.02,
) -> tf.Tensor:
    """Wing loss (Feng et al. 2018) adapted for normalized [0,1] coords.

    omega and epsilon are in the same units as the predictions (normalized).
    """
    delta = tf.abs(y_true - y_pred)
    C = omega - omega * tf.math.log(1.0 + omega / epsilon)
    loss = tf.where(
        delta < omega,
        omega * tf.math.log(1.0 + delta / epsilon),
        delta - C,
    )
    return tf.reduce_mean(loss, axis=-1)


def landmark_nme(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean per-landmark Euclidean distance in normalized crop coordinates.

    Equivalent to NME normalized by crop size.  Lower is better.
    Typical good values are < 0.05 (i.e. < 5% of the crop dimension).
    """
    diff = tf.reshape(y_true - y_pred, [-1, NUM_LANDMARKS, 2])
    dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-8)
    return tf.reduce_mean(dist)


def landmark_nme_iod(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """NME normalized by inter-ocular distance (IOD).

    IOD = Euclidean distance between outer corners of left and right eyes
    (landmarks 4 and 8).  This is the standard metric used in the CatFLW
    benchmark.  Returns a percentage value — typical good values < 7.0.
    """
    true_2d = tf.reshape(y_true, [-1, NUM_LANDMARKS, 2])
    pred_2d = tf.reshape(y_pred, [-1, NUM_LANDMARKS, 2])

    # Outer eye corners from ground truth.
    left_eye_outer = true_2d[:, LEFT_OUTER_EYE_IDX, :]    # [B, 2]
    right_eye_outer = true_2d[:, RIGHT_OUTER_EYE_IDX, :]  # [B, 2]

    iod = tf.sqrt(
        tf.reduce_sum(tf.square(left_eye_outer - right_eye_outer), axis=-1) + 1e-8
    )  # [B]

    # Per-landmark Euclidean distance.
    diff = pred_2d - true_2d  # [B, 48, 2]
    dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-8)  # [B, 48]
    mean_dist = tf.reduce_mean(dist, axis=-1)  # [B]

    nme = mean_dist / tf.maximum(iod, 1e-8)  # [B]
    return tf.reduce_mean(nme) * 100.0


# ---------------------------------------------------------------------------
# SWA (Stochastic Weight Averaging)
# ---------------------------------------------------------------------------

class SWACallback(tf.keras.callbacks.Callback):
    """Collect weights from `start_epoch` onward and average them."""

    def __init__(self, start_epoch: int = 0):
        super().__init__()
        self.start_epoch = start_epoch
        self._weight_sum: list[np.ndarray] | None = None
        self._count = 0

    def on_epoch_end(self, epoch: int, logs=None):
        if epoch >= self.start_epoch:
            weights = self.model.get_weights()
            if self._weight_sum is None:
                self._weight_sum = [np.zeros_like(w) for w in weights]
            for s, w in zip(self._weight_sum, weights):
                s += w
            self._count += 1

    def get_averaged_weights(self) -> list[np.ndarray] | None:
        if self._weight_sum is None or self._count == 0:
            return None
        return [s / self._count for s in self._weight_sum]


# ---------------------------------------------------------------------------
# Heatmap head components
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="CatFLW")
class SoftArgmax2D(tf.keras.layers.Layer):
    """Extract (x, y) coordinates from heatmaps via differentiable soft-argmax.

    Input:  (B, H, W, K) — K heatmaps of spatial size H*W
    Output: (B, K*2) — flattened [x0, y0, x1, y1, ...] in [0, 1]

    Args:
        beta: Temperature parameter for softmax. Higher values produce sharper
              distributions. Default 1.0 (standard softmax). Use 20-60 for
              112x112 heatmaps with direct heatmap supervision.

    All ops are TFLite-compatible (softmax, multiply, reduce_sum, constants).
    """

    def __init__(self, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def build(self, input_shape):
        _, h, w, k = input_shape
        # Keep the spatial dims as Python ints so call() can use static shapes.
        # Reading them back with tf.shape() at call time makes the reshapes
        # dynamic, which leaves a dynamic-sized tensor in the exported graph:
        # TFLite then warns that delegates supporting only static shapes cannot
        # cover it, and LiteRT Next's compiled runtime produces all-zero output
        # and fails on the second invocation. Only the batch dim needs to stay
        # dynamic, expressed as -1.
        self._h, self._w, self._k = int(h), int(w), int(k)
        # Coordinate grids normalized to [0, 1].
        # x varies along width (axis=1), y varies along height (axis=0).
        x_coords = tf.linspace(0.0, 1.0, w)  # [W]
        y_coords = tf.linspace(0.0, 1.0, h)  # [H]
        # Broadcast-ready shapes: x_grid [1, 1, W, 1], y_grid [1, H, 1, 1]
        self.x_grid = tf.reshape(x_coords, [1, 1, w, 1])
        self.y_grid = tf.reshape(y_coords, [1, h, 1, 1])
        super().build(input_shape)

    def call(self, heatmaps):
        # heatmaps: (B, H, W, K) with H, W, K static from build().
        h, w, k = self._h, self._w, self._k

        # Spatial softmax with temperature: flatten H*W, softmax, reshape back.
        flat = tf.reshape(heatmaps, [-1, h * w, k])       # (B, H*W, K)
        weights = tf.nn.softmax(flat * self.beta, axis=1)   # (B, H*W, K)
        weights = tf.reshape(weights, [-1, h, w, k])        # (B, H, W, K)

        # Weighted sum of coordinates.
        x = tf.reduce_sum(weights * self.x_grid, axis=[1, 2])  # (B, K)
        y = tf.reduce_sum(weights * self.y_grid, axis=[1, 2])  # (B, K)

        # Interleave as [x0, y0, x1, y1, ...].
        coords = tf.stack([x, y], axis=-1)  # (B, K, 2)
        return tf.reshape(coords, [-1, k * 2])  # (B, K*2)

    def get_config(self):
        config = super().get_config()
        config["beta"] = self.beta
        return config


def _build_deconv_head(backbone_output, num_landmarks: int, channels: int,
                       dropout: float = 0.0, num_deconv: int = 3,
                       softargmax_beta: float = 1.0):
    """SimpleBaseline-style deconv head: N* deconv + 1*1 conv + soft-argmax.

    backbone_output: (B, 7, 7, C) feature map from EfficientNet
    num_deconv: 3 for 56*56 heatmaps, 4 for 112*112
    Returns: (heatmaps_tensor, coords_tensor)
    """
    x = backbone_output

    for i in range(num_deconv):
        x = tf.keras.layers.Conv2DTranspose(
            channels, kernel_size=4, strides=2, padding="same",
            use_bias=False, name=f"deconv_{i+1}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"deconv_bn_{i+1}")(x)
        x = tf.keras.layers.ReLU(name=f"deconv_relu_{i+1}")(x)
        if dropout > 0:
            x = tf.keras.layers.SpatialDropout2D(dropout, name=f"deconv_drop_{i+1}")(x)

    # 1*1 conv to produce one heatmap per landmark.
    heatmaps = tf.keras.layers.Conv2D(
        num_landmarks, kernel_size=1, padding="same",
        name="heatmap_conv",
    )(x)

    # Differentiable coordinate extraction.
    coords = SoftArgmax2D(beta=softargmax_beta, name="soft_argmax")(heatmaps)
    return heatmaps, coords


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(cfg: ExperimentConfig) -> tf.keras.Model:
    img_size = cfg.img_size
    pretrained = not cfg.no_pretrained
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="crop")

    if cfg.backbone == "densenet121":
        # DenseNet121 uses torch-style preprocessing: normalize [0,1] input
        # with ImageNet mean=[0.485,0.456,0.406] and std=[0.229,0.224,0.225]
        x = tf.keras.layers.Normalization(
            mean=[0.485, 0.456, 0.406],
            variance=[0.229**2, 0.224**2, 0.225**2],
            name="imagenet_norm",
        )(inputs)
        backbone = tf.keras.applications.DenseNet121(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights=None if not pretrained else "imagenet",
        )
    elif cfg.backbone == "efficientnetv2s":
        # EfficientNetV2S includes internal Rescaling; expects [0, 255] input.
        x = tf.keras.layers.Rescaling(scale=255.0, offset=0.0, name="to_0_255")(inputs)
        backbone = tf.keras.applications.EfficientNetV2S(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights=None if not pretrained else "imagenet",
        )
    elif cfg.backbone == "mobilenetv3large":
        # MobileNetV3Large expects [0, 255] input (has internal preprocessing).
        x = tf.keras.layers.Rescaling(scale=255.0, offset=0.0, name="to_0_255")(inputs)
        backbone = tf.keras.applications.MobileNetV3Large(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            minimalistic=False,
            weights=None if not pretrained else "imagenet",
        )
    elif cfg.backbone == "mobilenetv3small":
        # MobileNetV3Small expects [0, 255] input (has internal preprocessing).
        x = tf.keras.layers.Rescaling(scale=255.0, offset=0.0, name="to_0_255")(inputs)
        backbone = tf.keras.applications.MobileNetV3Small(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            minimalistic=False,
            weights=None if not pretrained else "imagenet",
        )
    else:  # efficientnetb2
        x = tf.keras.layers.Rescaling(scale=255.0, offset=0.0, name="to_0_255")(inputs)
        backbone = tf.keras.applications.EfficientNetB2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights=None if not pretrained else "imagenet",
        )
    backbone.trainable = False
    x = backbone(x, training=False)

    if cfg.head_type == "heatmap":
        # Deconv upsampling + heatmap + soft-argmax (preserves spatial info).
        heatmaps, coords = _build_deconv_head(
            x, NUM_LANDMARKS, cfg.heatmap_channels,
            dropout=cfg.heatmap_dropout, num_deconv=cfg.num_deconv_layers,
            softargmax_beta=cfg.softargmax_beta,
        )
        coords = tf.keras.layers.Identity(name="landmarks_xy")(coords)
        if cfg.pure_heatmap_supervision:
            # Pure heatmap supervision: output only heatmaps for training.
            # SoftArgmax2D coords are still in the graph for TFLite export.
            return tf.keras.Model(
                inputs=inputs, outputs=heatmaps,
                name="cat_face_landmark_regressor",
            )
        elif cfg.heatmap_supervision:
            # Multi-output for training: heatmaps + coordinates
            return tf.keras.Model(
                inputs=inputs,
                outputs={"hm": heatmaps, "xy": coords},
                name="cat_face_landmark_regressor",
            )
        else:
            return tf.keras.Model(
                inputs=inputs, outputs=coords,
                name="cat_face_landmark_regressor",
            )
    else:
        # Original dense head (GAP destroys spatial info).
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        outputs = tf.keras.layers.Dense(
            NUM_LANDMARKS * 2, activation="sigmoid", name="landmarks_xy"
        )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="cat_face_landmark_regressor")


def _make_wing_loss(omega: float, epsilon: float):
    """Return a Wing loss function with baked-in parameters (Keras-serializable)."""
    @tf.keras.utils.register_keras_serializable(package="CatFLW")
    def wing_loss_fn(y_true, y_pred):
        return wing_loss(y_true, y_pred, omega=omega, epsilon=epsilon)
    return wing_loss_fn


def _make_weighted_mse(weights_list: list):
    """Return a per-landmark weighted MSE loss.

    weights_list: 48 floats, one per landmark. Internally normalized so
    sum(weights) == 48 (preserving same loss scale as unweighted MSE).
    The weight for landmark i is applied to both its x and y coordinates.
    """
    w = np.array(weights_list, dtype=np.float32)
    w = w / w.mean()  # normalize so mean weight = 1.0
    # Repeat each weight for x,y -> shape [96]
    w_flat = np.repeat(w, 2).astype(np.float32)
    w_tensor = tf.constant(w_flat, dtype=tf.float32)  # [96]

    @tf.keras.utils.register_keras_serializable(package="CatFLW")
    def weighted_mse_fn(y_true, y_pred):
        sq_diff = tf.square(y_true - y_pred)  # [B, 96]
        weighted = sq_diff * w_tensor  # [B, 96]
        return tf.reduce_mean(weighted)
    return weighted_mse_fn


def compile_model(model: tf.keras.Model, lr, cfg: ExperimentConfig) -> None:
    if cfg.optimizer == "adamw":
        try:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=cfg.weight_decay)
        except AttributeError:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        try:
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
        except AttributeError:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    if cfg.landmark_weights is not None:
        coord_loss_fn = _make_weighted_mse(cfg.landmark_weights)
    elif cfg.loss == "mse":
        coord_loss_fn = tf.keras.losses.MeanSquaredError()
    else:
        coord_loss_fn = _make_wing_loss(cfg.wing_omega, cfg.wing_epsilon)

    if cfg.pure_heatmap_supervision:
        # Pure heatmap supervision: only heatmap MSE loss, no coord metrics during training.
        if cfg.per_landmark_heatmap_loss:
            # Per-landmark normalized MSE: average loss within each landmark channel,
            # then average across landmarks. This prevents background pixels from
            # dominating the loss gradient — each landmark contributes equally regardless
            # of how many background pixels exist.
            @tf.keras.utils.register_keras_serializable(package="CatFLW")
            def per_landmark_heatmap_mse(y_true, y_pred):
                # y_true, y_pred: (B, H, W, K) where K=48 landmarks
                sq_diff = tf.square(y_true - y_pred)  # (B, H, W, K)
                # Average over spatial dims per landmark, then average across landmarks
                per_lm = tf.reduce_mean(sq_diff, axis=[1, 2])  # (B, K)
                return tf.reduce_mean(per_lm)
            hm_loss = per_landmark_heatmap_mse
        else:
            hm_loss = tf.keras.losses.MeanSquaredError()
        model.compile(
            optimizer=optimizer,
            loss=hm_loss,
            run_eagerly=False,
        )
    elif cfg.heatmap_supervision:
        model.compile(
            optimizer=optimizer,
            loss={"hm": tf.keras.losses.MeanSquaredError(), "xy": coord_loss_fn},
            loss_weights={"hm": 0.1, "xy": cfg.coord_loss_weight},
            metrics={"xy": [landmark_nme, landmark_nme_iod]},
            run_eagerly=False,
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss=coord_loss_fn,
            metrics=[landmark_nme, landmark_nme_iod],
            run_eagerly=False,
        )


def get_backbone(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and (
            layer.name.startswith("efficientnet") or layer.name.startswith("densenet")
            or layer.name.startswith("Mobilenet") or layer.name.startswith("mobilenet")
        ):
            return layer
    raise RuntimeError("Backbone not found")


@tf.keras.utils.register_keras_serializable(package="CatFLW")
class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup then delegates to an inner schedule (or constant)."""

    def __init__(self, inner, warmup_steps: int):
        super().__init__()
        self.inner = inner
        self.warmup_steps = warmup_steps
        self._peak_lr = float(inner) if isinstance(inner, (int, float)) else inner.initial_learning_rate

    def __call__(self, step):
        step_f = tf.cast(step, tf.float32)
        warmup = tf.minimum(step_f / tf.maximum(tf.cast(self.warmup_steps, tf.float32), 1.0), 1.0)
        if isinstance(self.inner, (int, float)):
            return self._peak_lr * warmup
        return self.inner(step) * warmup

    def get_config(self):
        return {"inner": self.inner, "warmup_steps": self.warmup_steps}


def build_lr_schedule(cfg: ExperimentConfig, num_train: int, total_epochs: int):
    """Build a learning rate or schedule from config."""
    steps_per_epoch = math.ceil(num_train / cfg.batch_size)

    if cfg.lr_schedule == "cosine":
        total_steps = steps_per_epoch * total_epochs
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg.learning_rate,
            decay_steps=total_steps,
            alpha=cfg.lr_min,
        )
    else:
        schedule = cfg.learning_rate

    # Only apply warmup when fine-tuning pretrained backbone layers.
    if cfg.unfreeze_backbone:
        warmup_steps = steps_per_epoch * 5
        return WarmupSchedule(schedule, warmup_steps)

    return schedule


def _monitor_metric(cfg: ExperimentConfig) -> str:
    if cfg.pure_heatmap_supervision:
        return "val_loss"
    base = "landmark_nme_iod" if cfg.nme_mode == "iod" else "landmark_nme"
    if cfg.heatmap_supervision:
        # Keras prefixes with the output layer name, not the dict key.
        return f"val_landmarks_xy_{base}"
    return f"val_{base}"


def _score_key(cfg: ExperimentConfig) -> str:
    if cfg.pure_heatmap_supervision:
        return "loss"
    base = "landmark_nme_iod" if cfg.nme_mode == "iod" else "landmark_nme"
    if cfg.heatmap_supervision:
        return f"landmarks_xy_{base}"
    return base


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    out_dir: Path,
    cfg: ExperimentConfig,
    num_train: int,
) -> tf.keras.Model:
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor = _monitor_metric(cfg)
    score_key = _score_key(cfg)

    if cfg.finetune_epochs == 0:
        # === SINGLE-PHASE TRAINING ===
        if cfg.unfreeze_backbone:
            backbone = get_backbone(model)
            backbone.trainable = True
            for layer in backbone.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False

        lr = build_lr_schedule(cfg, num_train, cfg.epochs)
        compile_model(model, lr=lr, cfg=cfg)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor, mode="min",
                patience=cfg.patience, restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(out_dir / "train_log.csv"), append=False),
        ]
        # ReduceLROnPlateau is only compatible with constant LR (not schedules).
        if cfg.lr_schedule == "constant":
            callbacks.insert(1, tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor, mode="min",
                factor=0.5, patience=5, min_lr=1e-6, verbose=1,
            ))

        swa_cb = None
        if cfg.use_swa:
            swa_start = int(cfg.epochs * cfg.swa_start_frac)
            swa_cb = SWACallback(start_epoch=swa_start)
            callbacks.append(swa_cb)

        print(f"\n=== Single-phase training: {cfg.epochs} epochs ===")
        model.fit(
            train_ds, validation_data=val_ds,
            epochs=max(cfg.epochs, 0),
            callbacks=callbacks, verbose=2,
        )
        best_metrics = evaluate_model(model, val_ds)
        best_score = float(best_metrics.get(score_key, float("inf")))
        best_source = "single_phase"
        best_state = [np.array(w, copy=True) for w in model.get_weights()]

        # Try SWA if enabled
        if swa_cb is not None:
            swa_weights = swa_cb.get_averaged_weights()
            if swa_weights is not None:
                model.set_weights(swa_weights)
                swa_metrics = evaluate_model(model, val_ds)
                swa_score = float(swa_metrics.get(score_key, float("inf")))
                print(f"SWA {score_key}={swa_score:.6f}")
                if swa_score <= best_score:
                    best_score = swa_score
                    best_source = "swa"
                    best_state = [np.array(w, copy=True) for w in swa_weights]
    else:
        # === TWO-PHASE TRAINING ===
        phase1_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor, mode="min",
                patience=min(cfg.patience, 6), restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor, mode="min",
                factor=0.5, patience=3, min_lr=1e-6, verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(out_dir / "train_log.csv"), append=False),
        ]

        print(f"\n=== Phase 1: frozen backbone, {cfg.epochs} epochs ===")
        compile_model(model, lr=cfg.learning_rate, cfg=cfg)
        model.fit(
            train_ds, validation_data=val_ds,
            epochs=max(cfg.epochs, 0),
            callbacks=phase1_callbacks, verbose=2,
        )
        p1_metrics = evaluate_model(model, val_ds)
        best_score = float(p1_metrics.get(score_key, float("inf")))
        best_source = "phase1"
        best_state = [np.array(w, copy=True) for w in model.get_weights()]

        print(f"\n=== Phase 2: fine-tune backbone tail, {cfg.finetune_epochs} epochs ===")
        backbone = get_backbone(model)
        backbone.trainable = True
        if cfg.finetune_last_layers > 0:
            for layer in backbone.layers[:-cfg.finetune_last_layers]:
                layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        # Use warmup for backbone fine-tuning to avoid destroying pretrained weights.
        steps_per_epoch = math.ceil(num_train / cfg.batch_size)
        warmup_steps = steps_per_epoch * 5  # 5-epoch warmup for Phase 2
        if cfg.lr_schedule == "cosine":
            # Cosine decay for Phase 2 fine-tuning
            total_ft_steps = steps_per_epoch * cfg.finetune_epochs
            cosine_sched = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=cfg.finetune_learning_rate,
                decay_steps=total_ft_steps,
                alpha=cfg.lr_min,
            )
            ft_lr = WarmupSchedule(cosine_sched, warmup_steps)
        else:
            ft_lr = WarmupSchedule(cfg.finetune_learning_rate, warmup_steps)
        compile_model(model, lr=ft_lr, cfg=cfg)

        swa_start = int(cfg.finetune_epochs * cfg.swa_start_frac)
        swa_cb = SWACallback(start_epoch=swa_start) if cfg.use_swa else None
        phase2_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor, mode="min",
                patience=cfg.patience, restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(out_dir / "train_log.csv"), append=True),
        ]
        if swa_cb is not None:
            phase2_callbacks.append(swa_cb)

        model.fit(
            train_ds, validation_data=val_ds,
            epochs=max(cfg.finetune_epochs, 0),
            callbacks=phase2_callbacks, verbose=2,
        )
        p2_metrics = evaluate_model(model, val_ds)
        p2_score = float(p2_metrics.get(score_key, float("inf")))
        if p2_score <= best_score:
            best_score = p2_score
            best_source = "phase2"
            best_state = [np.array(w, copy=True) for w in model.get_weights()]

        if swa_cb is not None:
            swa_weights = swa_cb.get_averaged_weights()
            if swa_weights is not None:
                model.set_weights(swa_weights)
                swa_metrics = evaluate_model(model, val_ds)
                swa_score = float(swa_metrics.get(score_key, float("inf")))
                print(f"SWA {score_key}={swa_score:.6f}")
                if swa_score <= best_score:
                    best_score = swa_score
                    best_source = "swa"
                    best_state = [np.array(w, copy=True) for w in swa_weights]

    print(f"Selecting {best_source} model ({score_key}={best_score:.6f})")
    model.set_weights(best_state)
    model.save(out_dir / "best.keras")
    model.save_weights(out_dir / "best.weights.h5")
    return model


def evaluate_model(model: tf.keras.Model, val_ds: tf.data.Dataset) -> dict[str, float]:
    values = model.evaluate(val_ds, verbose=0)
    if isinstance(values, (int, float)):
        # Single loss, no extra metrics (e.g. pure_heatmap_supervision)
        metrics = {"loss": float(values)}
    else:
        metrics = {n: float(v) for n, v in zip(model.metrics_names, values)}
    print("Validation metrics:", metrics)
    return metrics


# ---------------------------------------------------------------------------
# TFLite export + sanity check
# ---------------------------------------------------------------------------

def export_tflite(model: tf.keras.Model, out_path: Path) -> None:
    """Convert to float16 TFLite from the Keras model (dynamic batch dimension).

    Do not "fix" this to convert from a batch-1 concrete function without
    re-measuring first. The obvious-looking argument for doing so is wrong on the
    runtime these models actually ship on, and it was tried:

    Converting this way leaves the batch dimension dynamic, so Conv2DTranspose
    builds each deconv's output shape at run time out of SHAPE/STRIDED_SLICE/PACK
    and TFLite logs "Attempting to use a delegate that only supports static-sized
    tensors ...". Pinning the batch to 1 removes all of that (295 ops -> 279) and
    produces identical predictions. It looks like a free win, and on
    flutter_litert 3.6.0 it was one. On 3.7.0 it is a 2x regression:

      | litert | export  | median invoke | val NME_IOD |
      |--------|---------|---------------|-------------|
      | 3.6.0  | dynamic |      94.5 ms  |   3.4857    |
      | 3.6.0  | static  |      59.8 ms  |   3.4857    |
      | 3.7.0  | dynamic |      28.2 ms  |   3.4857    |
      | 3.7.0  | static  |      59.9 ms  |   3.4857    |

    3.7.0 got much faster on the dynamic graph specifically, and the static graph
    did not move. Accuracy is identical in all four cells, so this is purely about
    which graph the delegate handles better in a given release. The dog model
    reproduces the same inversion.

    The lesson is not "dynamic is better" either. It is that this choice is a
    property of the runtime version, not of the model, and it has flipped once
    already. Re-measure both against the version the packages resolve, with
    dogs-in-the-wild-ml/scripts/bench_litert_macos.py, before changing this.
    scripts/reexport_static.py produces the static variant from a trained
    best.keras without retraining.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_bytes = converter.convert()
    out_path.write_bytes(tflite_bytes)
    print(f"Saved TFLite: {out_path} ({len(tflite_bytes)/1024/1024:.2f} MB)")


def tflite_sanity_check(
    tflite_path: Path,
    val_records: list[Record],
    img_size: int,
    crop_margin: float,
    num_samples: int = 16,
) -> dict[str, float]:
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    print("TFLite input:", in_det["shape"], in_det["dtype"],
          "output:", out_det["shape"], out_det["dtype"])

    nmes: list[float] = []
    for rec in val_records[:num_samples]:
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        lm_flat = tf.constant(
            [coord for pt in rec.landmarks_abs for coord in pt], dtype=tf.float32
        )
        crop, lm_norm = crop_and_normalize(
            image,
            tf.constant(rec.bbox_xyxy_abs, tf.float32),
            lm_flat,
            img_size,
            crop_margin,
            square_pad=cfg.square_pad,
        )
        inp = tf.expand_dims(crop, 0).numpy().astype(in_det["dtype"])
        interp.set_tensor(in_det["index"], inp)
        interp.invoke()
        pred_flat = interp.get_tensor(out_det["index"])[0].astype(np.float32)
        pred_flat = np.clip(pred_flat, 0.0, 1.0)

        diff = lm_norm.numpy() - pred_flat  # [96]
        diff_2d = diff.reshape(NUM_LANDMARKS, 2)
        nme = float(np.mean(np.sqrt(np.sum(diff_2d**2, axis=-1))))
        nmes.append(nme)

    result = {"tflite_mean_nme_sample": float(np.mean(nmes)) if nmes else math.nan}
    print("TFLite sanity:", result)
    return result


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def save_metadata(
    out_dir: Path,
    cfg: ExperimentConfig,
    data_root: Path,
    train_records: list[Record],
    val_records: list[Record],
    val_metrics: dict[str, float],
    tflite_path: Path,
    tflite_sanity: dict[str, float],
) -> None:
    meta = {
        "model_name": "cat_face_landmark_regressor",
        "num_landmarks": NUM_LANDMARKS,
        "data_root": str(data_root),
        "dataset": "CatFLW",
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "input": {
            "shape": [1, cfg.img_size, cfg.img_size, 3],
            "dtype": "float32",
            "range": [0.0, 1.0],
            "preprocessing": f"crop to GT/pred bbox + margin, resize to square, rescale to [0,255] for {cfg.backbone} in-model",
        },
        "output": {
            "name": "landmarks_xy",
            "format": "flattened [x0,y0, x1,y1, ..., x47,y47] normalized in [0,1] relative to crop",
            "shape": [1, NUM_LANDMARKS * 2],
        },
        "experiment_config": asdict(cfg),
        "validation_metrics": val_metrics,
        "tflite_sanity": tflite_sanity,
        "artifacts": {
            "tflite": str(tflite_path),
            "keras_best": str(out_dir / "best.keras"),
            "train_log_csv": str(out_dir / "train_log.csv"),
        },
    }
    (out_dir / "model_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    configure_ca_bundle()
    set_seed(cfg.seed)
    tf.config.optimizer.set_jit(False)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"TensorFlow: {tf.__version__}")
    print(f"Experiment: {cfg.name} | backbone={cfg.backbone} loss={cfg.loss} "
          f"epochs={cfg.epochs} ft_epochs={cfg.finetune_epochs} batch={cfg.batch_size}")
    if not args.data_root.exists():
        raise FileNotFoundError(f"CatFLW not found at {args.data_root}")

    all_records = load_all_records(args.data_root, cfg.lm_margin,
                                   use_dataset_bbox=cfg.use_dataset_bbox)
    if not all_records:
        raise RuntimeError("No usable records found.")
    train_records, val_records = split_records(all_records, args.test_fraction, cfg.seed)
    if not train_records or not val_records:
        raise RuntimeError("Empty train or val records after split.")

    train_ds = build_tf_dataset(train_records, cfg=cfg, training=True)
    val_ds = build_tf_dataset(val_records, cfg=cfg, training=False)

    model = build_model(cfg)
    model.summary()

    if not args.tflite_only:
        model = train_model(model, train_ds, val_ds, out_dir=args.out_dir, cfg=cfg,
                            num_train=len(train_records))
    else:
        best_w = args.out_dir / "best.weights.h5"
        if not best_w.exists():
            raise FileNotFoundError(f"--tflite-only: {best_w} not found")
        model.load_weights(str(best_w))

    compile_model(model, lr=cfg.learning_rate, cfg=cfg)
    val_metrics = evaluate_model(model, val_ds)

    # For heatmap models, extract coord-only model for TFLite export.
    if cfg.pure_heatmap_supervision:
        # Pure heatmap model outputs heatmaps. Rebuild with SoftArgmax2D for export.
        hm_out = model.output  # (B, H, W, 48)
        coords = SoftArgmax2D(beta=cfg.softargmax_beta, name="soft_argmax_export")(hm_out)
        export_model = tf.keras.Model(inputs=model.inputs, outputs=coords)
    elif cfg.heatmap_supervision:
        export_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.get_layer("landmarks_xy").output,
        )
    else:
        export_model = model

    tflite_path = args.out_dir / f"cat_face_landmarks_{cfg.img_size}_float16.tflite"
    export_tflite(export_model, tflite_path)

    tflite_sanity = tflite_sanity_check(
        tflite_path, val_records,
        img_size=cfg.img_size, crop_margin=cfg.crop_margin,
    )
    save_metadata(
        out_dir=args.out_dir,
        cfg=cfg,
        data_root=args.data_root,
        train_records=train_records,
        val_records=val_records,
        val_metrics=val_metrics,
        tflite_path=tflite_path,
        tflite_sanity=tflite_sanity,
    )

    print("\nDone.")
    print(f"TFLite: {tflite_path}")
    print(f"Metadata: {args.out_dir / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
