# Cat Facial Landmark Detection: Progress Journal

**Current best: NME_IOD = 3.43 (384px single model) | Target: 2.91 (paper ELD ensemble) | Started at: 3.72**

This document is a living journal of our work on cat facial landmark detection using the CatFLW dataset. It's designed for future LLMs/developers to pick up where we left off and continue pushing toward (and beyond) the paper's target.

---

## Quick Reference

### Current Best Model
- **Best single model**: `tight_margin_384` — **3.43 NME_IOD** (no TTA)
- **Architecture**: EfficientNetV2S + 4-deconv heatmap head + SoftArgmax2D
- **Input**: 384x384 (192x192 heatmaps)
- **Artifacts**: `artifacts/tight_margin_384/`
- **Train-val gap**: ~1.0 (very low — overfitting is NOT the bottleneck here)

### All Single-Model Results
| Model | NME_IOD | Train NME_IOD | Gap | Notes |
|---|---|---|---|---|
| tight_margin_384 | **3.43** | 2.47 | 0.96 | **Current best** |
| tight_margin_320 | 3.53 | 2.51 | 1.02 | |
| tight_margin_256_long (400 ft epochs) | 3.61 | 2.46 | 1.15 | Longer training helped |
| tight_margin_256 | 3.72 | 2.83 | 0.89 | First run, baseline |

### In-Progress Runs
| Model | Status | Current Val NME_IOD | Notes |
|---|---|---|---|
| tight_margin_448 (448px, batch 4) | Phase 2, ep ~23/200 | ~3.70 | Pushing resolution further |
| tight_margin_384_long (400 ft, 80 layers) | Phase 2, ep ~62/400 | ~3.51 | Deeper + longer fine-tuning |

### Key Commands
```bash
# Train at 384px (current best single-model resolution)
python scripts/train_cat_face_landmarks.py --experiment tight_margin_384 --out artifacts/tight_margin_384

# Train at 320px
python scripts/train_cat_face_landmarks.py --experiment tight_margin_320 --out artifacts/tight_margin_320

# Train at 256px
python scripts/train_cat_face_landmarks.py --experiment tight_margin_256 --out artifacts/tight_margin_256

# Train at 448px (experimental)
python scripts/train_cat_face_landmarks.py --experiment tight_margin_384 --img-size 448 --batch-size 4 --out artifacts/tight_margin_448

# Longer fine-tuning with more unfrozen layers
python scripts/train_cat_face_landmarks.py --experiment tight_margin_384 --finetune-epochs 400 --finetune-last-layers 80 --out artifacts/tight_margin_384_long

# Comprehensive evaluation of all models + ensembles + TTA
python scripts/eval_all_models.py

# Multi-scale TTA evaluation
python scripts/eval_multiscale_tta.py
```

---

## Reference Paper

**"Automated Detection of Cat Facial Landmarks"** — Martvel, G., Farjon, G., & Kovalenko, B. (2024)
- **Dataset**: [CatFLW on Kaggle](https://www.kaggle.com/datasets/georgemartvel/catflw)
- **Paper**: Published in *International Journal of Computer Vision*
- **48 facial landmarks** on 2,079 images (we split 85/15 → 1,768 train / 311 test)
- **Best result**: NME_IOD = 2.91 using ELD (Ensemble of Landmark Detectors) + EfficientNetV2S
- **Benchmark**: [Papers with Code](https://paperswithcode.com/sota/facial-landmark-detection-on-catflw)

---

## How This Project Was Created

This repo was bootstrapped directly from the dogs-in-the-wild-ml project (DogFLW), which achieved NME_IOD = 8.04 on dog facial landmarks through extensive experimentation across 7 rounds. Rather than repeating all those experiments, we applied the proven best recipe directly:

**Architecture transferred from dogs (proven best):**
- EfficientNetV2S backbone + 4 deconv layers + SoftArgmax2D (heatmap head)
- Two-phase training: 100 epochs frozen → 200 epochs fine-tuning (last 50 layers, lr=1e-5)
- AdamW (weight_decay=1e-4), MSE loss, NME_IOD early stopping (patience=50)
- Tight crop margins: lm_margin=0.05, crop_margin=0.10
- Augmentations: horizontal flip, 15° rotation, scale jitter, crop jitter, brightness/contrast/saturation

**Dead-end approaches NOT transferred (proven failures on dogs):**
- Dense/GAP heads (destroy spatial info → NME ~40)
- Wing loss (worse than MSE)
- Heatmap supervision / dual loss (gradient interference)
- Mixup augmentation (underfitting on small datasets)
- Ear-weighted loss (robs from other landmarks)
- Heavy regularization (gap is structural, not solvable by dropout)
- Beta tuning for SoftArgmax (only helps if trained with same beta)

**CatFLW-specific adaptations:**
- 48 landmarks (vs 46 for dogs)
- Outer eye corner indices: 4 and 8 (vs 18 and 19 for dogs)
- Cat-specific FLIP_INDEX computed from average landmark positions
- No train/test split in dataset — we split 85/15 deterministically (seed=42)
- Label JSON key: "labels" (vs "landmarks" for dogs)

---

## Dataset: CatFLW

- **Source**: [Kaggle](https://www.kaggle.com/datasets/georgemartvel/catflw) (version 2)
- **Total images**: 2,079 (flat directory, no predefined split)
- **Our split**: 1,768 train / 311 test (15% test, seed=42)
- **Landmarks**: 48 per image, JSON format with key "labels"
- **Bounding boxes**: Provided in JSON key "bounding_boxes" [x1, y1, x2, y2]
- **Image format**: PNG

### Landmark Layout (48 landmarks)
Based on average normalized positions across all samples:
- **Right eye** (indices 3-7, 36-38): x ~ 0.31-0.42, y ~ 0.52-0.62
- **Left eye** (indices 1, 8-11, 39-41): x ~ 0.57-0.69, y ~ 0.52-0.62
- **Outer eye corners**: index 4 (right, x=0.31) and index 8 (left, x=0.69) — used for IOD
- **Right ear** (indices 22-26): x ~ 0.16-0.37, y ~ 0.15-0.46
- **Left ear** (indices 27-31): x ~ 0.62-0.83, y ~ 0.15-0.47
- **Nose** (indices 12-15, 32-35, 42-45): center, y ~ 0.71-0.77
- **Mouth/chin** (indices 0, 2, 16-21, 46-47): center/lower, y > 0.78

### Cat vs Dog Comparison
| | CatFLW | DogFLW |
|---|---|---|
| Images | 2,079 | 4,333 |
| Landmarks | 48 | 46 |
| Paper best NME_IOD | 2.91 | 6.52 |
| Our best (single model) | 3.43 | 8.77 |
| Train-val gap | ~1.0 | ~3.5 |

Cat faces are significantly easier to localize than dog faces. This is likely because:
- Cat faces have much less breed variation (no pug vs greyhound extremes)
- Cat facial structure is more uniform (consistent ear shape, eye placement)
- The train-val gap is 3.5x smaller, suggesting better generalization

---

## Experiment History

### Round 1: Resolution Sweep (NME_IOD 3.72 → 3.43)

Applied the proven best recipe from dogs at four resolutions. Every step up in resolution improved results — same pattern as dogs.

| # | Model | Resolution | Heatmap Size | NME_IOD | Train NME | Gap | Time |
|---|---|---|---|---|---|---|---|
| 1 | tight_margin_256 | 256x256 | 128x128 | 3.72 | 2.83 | 0.89 | ~1.5h |
| 2 | tight_margin_256_long | 256x256 | 128x128 | 3.61 | 2.46 | 1.15 | ~3h |
| 3 | tight_margin_320 | 320x320 | 160x160 | 3.53 | 2.51 | 1.02 | ~2.5h |
| 4 | **tight_margin_384** | 384x384 | 192x192 | **3.43** | 2.47 | 0.96 | ~3h |

**Key learnings:**
- Resolution is the #1 lever, same as dogs: 256→320 gave 0.19, 320→384 gave 0.10
- Returns are diminishing but haven't plateaued
- Longer fine-tuning helped at 256px (3.72→3.61), suggesting models aren't fully converged at 200 epochs
- Train-val gap is very small (~1.0) — overfitting is NOT the problem
- No run triggered early stopping — all ran to completion, still improving

### Round 2: Pushing Further (IN PROGRESS)

Two experiments to push the single model lower:

| # | Model | Config | Status | Current Val NME_IOD |
|---|---|---|---|---|
| 5 | tight_margin_448 | 448px, batch 4 | Phase 2 training | ~3.70 (early) |
| 6 | tight_margin_384_long | 384px, 400 ft epochs, 80 unfrozen layers | Phase 2 training | ~3.51 |

**Estimated final NME_IOD**: 448px → ~3.2, 384px long → ~3.3

---

## Not Yet Tried

### High confidence (worked on dogs):
- **Multi-scale + flip TTA** — gave ~0.5-0.8 free gain on dogs, no retraining needed
- **3-model ensemble** (256+320+384) — gave ~0.2 gain on dogs
- **Ensemble + ms+flip TTA** — compound of above, biggest win on dogs (8.77 → 8.04)

### Medium confidence:
- **512px resolution** — if 448px still shows gains, worth pushing further
- **Cosine annealing LR** for Phase 2 — untested, may find better optima
- **More aggressive fine-tuning** — unfreeze 100+ backbone layers

### Low confidence / not recommended:
- Higher regularization — train-val gap is only ~1.0, not worth addressing
- Mixup — failed on dogs with similar dataset size
- ELD (region-based specialists) — failed on dogs (9.15 vs 8.77 single model)
- Heatmap supervision — failed on dogs (gradient interference)
- Ear-weighted loss — failed on dogs (robs from other landmarks)

---

## Architecture Details

```
EfficientNetV2S (frozen phase 1, fine-tune last 50 layers phase 2)
    Input: [B, img_size, img_size, 3] (float32, [0,1] range)
    -> Rescaling(255) (in-model, for EfficientNet preprocessing)
    -> EfficientNetV2S backbone -> [B, img_size/32, img_size/32, 1280]
    -> Conv2DTranspose(256, 4x4, stride 2) + BN + ReLU + SpatialDropout(0.1)  -> 2x
    -> Conv2DTranspose(256, 4x4, stride 2) + BN + ReLU + SpatialDropout(0.1)  -> 4x
    -> Conv2DTranspose(256, 4x4, stride 2) + BN + ReLU + SpatialDropout(0.1)  -> 8x
    -> Conv2DTranspose(256, 4x4, stride 2) + BN + ReLU + SpatialDropout(0.1)  -> 16x
    -> Conv2D(48, 1x1) -> heatmaps [B, H, W, 48]
    -> SoftArgmax2D(beta=1.0) -> [B, 96] coordinates (x0,y0,...,x47,y47) in [0,1]

Heatmap sizes by resolution:
    224px -> 112x112
    256px -> 128x128
    320px -> 160x160
    384px -> 192x192
    448px -> 224x224

TFLite export: float16 quantization, ~55MB
```

### Training Recipe
```
Phase 1 (frozen backbone):
    Epochs: 100 (early stopping patience=6)
    LR: 1e-4 (ReduceLROnPlateau: factor=0.5, patience=3)
    Optimizer: AdamW (weight_decay=1e-4)

Phase 2 (fine-tune backbone tail):
    Epochs: 200 (early stopping patience=50)
    LR: 1e-5 with 5-epoch linear warmup
    Unfreeze: last 50 layers (BN layers stay frozen)
    Optimizer: AdamW (weight_decay=1e-4)

Loss: MSE on [0,1] normalized coordinates
Metric: NME_IOD (inter-ocular distance normalized)
```
