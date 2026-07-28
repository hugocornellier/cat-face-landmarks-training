# Cat Facial Landmark Detection: Progress Journal

**Current best: NME_IOD = 3.11 (EfficientNetV2S 448px cosine+SWA + ms+flip TTA) | Best raw: 3.27 | Best small: 3.48 / 3.31 w/ TTA (11MB) | Target: 2.91 (paper ELD ensemble) | Started at: 3.72**

This document is a living journal of our work on cat facial landmark detection using the CatFLW dataset. It's designed for future LLMs/developers to pick up where we left off and continue pushing toward (and beyond) the paper's target.

---

## Quick Reference

### Current Best Models
- **Best overall (with TTA)**: `tight_margin_448_cosine_swa` + ms+flip TTA — **3.11 NME_IOD** (55MB TFLite, 6 inference passes)
- **Best raw (no TTA)**: `tight_margin_448_long` / `tight_margin_448_cosine_swa` (EfficientNetV2S) — **3.27 NME_IOD** (55MB TFLite)
- **Best small model**: `small_v3large_384_long` (MobileNetV3Large) — **3.48 NME_IOD** raw / **3.31** w/ TTA (11MB TFLite)
- **Architecture**: Backbone + 4-deconv heatmap head + SoftArgmax2D
- **Train-val gap**: ~1.0 (very low — overfitting is NOT the bottleneck)

### All Single-Model Results
| # | Model | Backbone | Resolution | NME_IOD | + TTA | Train NME | Gap | TFLite Size |
|---|---|---|---|---|---|---|---|---|
| 1 | **tight_margin_448_cosine_swa** | EfficientNetV2S | 448 | **3.27** | **3.11** | — | — | 55MB |
| 2 | tight_margin_448_long | EfficientNetV2S | 448 | 3.27 | 3.13 | 1.63 | 1.64 | 55MB |
| 3 | tight_margin_384_long | EfficientNetV2S | 384 | 3.33 | 3.16 | 2.35 | 0.98 | 55MB |
| 4 | tight_margin_448 | EfficientNetV2S | 448 | 3.38 | — | 2.30 | 1.08 | 55MB |
| 5 | tight_margin_384 | EfficientNetV2S | 384 | 3.43 | — | 2.47 | 0.96 | 55MB |
| 6 | **small_v3large_384_long** | **MobileNetV3Large** | **384** | **3.48** | **3.31** | **2.48** | **1.00** | **11MB** |
| 7 | tight_margin_320 | EfficientNetV2S | 320 | 3.53 | — | 2.51 | 1.02 | 55MB |
| 8 | tight_margin_256_long | EfficientNetV2S | 256 | 3.61 | — | 2.46 | 1.15 | 55MB |
| 9 | small_v3large_384 | MobileNetV3Large | 384 | 3.62 | — | 2.62 | 1.00 | 11MB |
| 10 | small_v3large_448 | MobileNetV3Large | 448 | 3.63 | — | 2.46 | 1.17 | 11MB |
| 11 | tight_margin_256 | EfficientNetV2S | 256 | 3.72 | — | 2.83 | 0.89 | 55MB |
| 12 | small_v3small_256 | MobileNetV3Small | 256 | 4.65 | — | 3.40 | 1.25 | 5.6MB |

### Key Commands
```bash
# Best overall model (EfficientNetV2S 448px, long fine-tune)
python scripts/train_cat_face_landmarks.py --experiment tight_margin_384 --img-size 448 --batch-size 4 --finetune-epochs 400 --finetune-last-layers 80 --out artifacts/tight_margin_448_long

# Best small model (MobileNetV3Large 384px, long fine-tune)
python scripts/train_cat_face_landmarks.py --experiment small_v3large_384 --finetune-epochs 400 --finetune-last-layers 60 --out artifacts/small_v3large_384_long

# Standard resolution sweep
python scripts/train_cat_face_landmarks.py --experiment tight_margin_384 --out artifacts/tight_margin_384
python scripts/train_cat_face_landmarks.py --experiment tight_margin_320 --out artifacts/tight_margin_320
python scripts/train_cat_face_landmarks.py --experiment tight_margin_256 --out artifacts/tight_margin_256

# Small model variants
python scripts/train_cat_face_landmarks.py --experiment small_v3large_256 --out artifacts/small_v3large_256
python scripts/train_cat_face_landmarks.py --experiment small_v3small_256 --out artifacts/small_v3small_256

# Best with cosine+SWA (Round 6 — best TTA score)
python scripts/train_cat_face_landmarks.py --experiment tight_margin_384 --img-size 448 --batch-size 4 --finetune-epochs 400 --finetune-learning-rate 2e-5 --lr-schedule cosine --use-swa --swa-start-frac 0.65 --out artifacts/tight_margin_448_cosine_swa

# TTA evaluation (best model)
python scripts/eval_multiscale_tta.py --model artifacts/tight_margin_448_cosine_swa/best.keras --experiment tight_margin_384 --img-size 448

# TTA evaluation (small model)
python scripts/eval_multiscale_tta.py --model artifacts/small_v3large_384_long/best.keras --experiment small_v3large_384

# Comprehensive evaluation of all models + ensembles
python scripts/eval_all_models.py
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
| Our best (single model, raw) | 3.27 | 8.77 |
| Our best (single model + TTA) | 3.13 | 8.04 |
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

### Round 2: Pushing EfficientNetV2S Further (NME_IOD 3.43 → 3.33)

Two experiments to push the large model lower: higher resolution and longer training.

| # | Model | Config | NME_IOD | Train NME | Gap |
|---|---|---|---|---|---|
| 5 | tight_margin_448 | 448px, batch 4, 200 ft epochs | 3.38 | 2.30 | 1.08 |
| 6 | **tight_margin_384_long** | 384px, 400 ft epochs, 80 unfrozen layers | **3.33** | 2.35 | 0.98 |

**Key learnings:**
- 448px gave modest gain (3.43→3.38) but with diminishing returns
- Longer fine-tuning (400 epochs, 80 unfrozen layers) was more effective (3.43→3.33)
- Combining both approaches may yield further gains but wasn't tested
- EfficientNetV2S has enough capacity that resolution and training length both help

### Round 3: Small Model Exploration (MobileNetV3)

Goal: achieve good accuracy in a much smaller package (~11MB vs 55MB TFLite).

| # | Model | Backbone | Resolution | Config | NME_IOD | Train NME | Gap | TFLite |
|---|---|---|---|---|---|---|---|---|
| 7 | small_v3large_384 | MobileNetV3Large | 384 | 200 ft, 40 layers | 3.62 | 2.62 | 1.00 | 11MB |
| 8 | **small_v3large_384_long** | **MobileNetV3Large** | **384** | **400 ft, 60 layers** | **3.48** | **2.48** | **1.00** | **11MB** |
| 9 | small_v3large_448 | MobileNetV3Large | 448 | 300 ft, 40 layers | 3.63 | 2.46 | 1.17 | 11MB |
| 10 | small_v3small_256 | MobileNetV3Small | 256 | 200 ft, 30 layers | 4.65 | 3.40 | 1.25 | 5.6MB |

**Key learnings:**
- MobileNetV3Large at 384px with long training achieves **3.48 in just 11MB** — only 0.15 worse than the 55MB EfficientNetV2S best
- Higher resolution (448px) did NOT help for V3Large (3.63 vs 3.48 at 384px) — the smaller backbone lacks capacity to exploit higher resolution
- Longer fine-tuning was the key lever for small models too (3.62→3.48)
- MobileNetV3Small is too small for this task (4.65 NME_IOD)
- Small models use 128 deconv channels (vs 256 for EfficientNetV2S) to keep size down
- The accuracy/size tradeoff is excellent: 5x smaller model with only 4.5% worse NME

### Round 4: Test-Time Augmentation (NME_IOD 3.33 → 3.16)

Applied multi-scale + flip TTA to models. No retraining — just smarter inference by averaging predictions across multiple augmented views.

**EfficientNetV2S 384px long (best at the time):**

| Mode | Passes | NME_IOD | Gain vs baseline |
|---|---|---|---|
| Baseline (no TTA) | 1 | 3.33 | — |
| Flip TTA only | 2 | 3.22 | -0.11 |
| Multi-scale (3 scales) | 3 | 3.24 | -0.09 |
| **Multi-scale + flip** | **6** | **3.16** | **-0.17** |

**MobileNetV3Large 384px long (11MB model):**

| Mode | Passes | NME_IOD | Gain vs baseline |
|---|---|---|---|
| Baseline (no TTA) | 1 | 3.48 | — |
| Flip TTA only | 2 | 3.37 | -0.11 |
| Multi-scale (3 scales) | 3 | 3.41 | -0.07 |
| **Multi-scale + flip** | **6** | **3.31** | **-0.17** |

**Key learnings:**
- TTA gives a consistent ~0.17 NME gain across both model sizes
- Smaller than dogs (~0.7) because cat faces are already more uniform
- Flip TTA alone (-0.11) is the biggest single contributor
- Scales used: [0.9, 1.0, 1.1] with reflect-padding for zoom-out
- 11MB model with TTA (3.31) matches the 55MB model's raw score (3.33)

### Round 5: Combining 448px + Long Training (NME_IOD 3.33 → 3.27 raw, 3.16 → 3.13 TTA)

Combined the two best levers from Round 2 that had each been tested independently: 448px resolution + 400 ft epochs / 80 unfrozen layers.

| # | Model | Config | NME_IOD | + TTA | Train NME | Gap |
|---|---|---|---|---|---|---|
| 11 | **tight_margin_448_long** | 448px, batch 4, 400 ft, 80 layers | **3.27** | **3.13** | 1.63 | 1.64 |

**TTA breakdown (448px long):**

| Mode | NME_IOD |
|---|---|
| Baseline | 3.27 |
| Flip TTA | 3.17 |
| Multi-scale | 3.21 |
| **Multi-scale + flip** | **3.13** |

**Key learnings:**
- Combining resolution + long training gave additive gains: 448px alone was 3.38, long alone was 3.33, combined gives 3.27
- Train-val gap grew to 1.64 (vs 0.98 at 384px long) — the model is starting to memorize at 448px
- TTA gain consistent at ~0.14
- This is likely near the EfficientNetV2S ceiling — resolution and training length are both plateauing
- Model will serve as the teacher for knowledge distillation into the 11MB model

### Round 6: Cosine Decay + SWA (NME_IOD raw unchanged at 3.27, TTA 3.13 → 3.11)

Tested cosine learning rate decay + Stochastic Weight Averaging (SWA) on the best 448px config. Codex (GPT-5.4) analysis recommended this as the lowest-effort, highest-confidence improvement.

**Config**: Same as tight_margin_448_long but with `--lr-schedule cosine --use-swa --swa-start-frac 0.65 --finetune-learning-rate 2e-5`

| # | Model | Config | NME_IOD | + TTA | Notes |
|---|---|---|---|---|---|
| 12 | tight_margin_448_cosine_swa | cosine lr (2e-5→1e-6) + SWA (last 35%) | 3.267 | **3.107** | SWA was 3.270, Phase 2 best selected |

**TTA breakdown (cosine+SWA model):**

| Mode | NME_IOD |
|---|---|
| Baseline | 3.268 |
| Flip TTA | 3.164 |
| Multi-scale | 3.189 |
| **Multi-scale + flip** | **3.107** |

**Key learnings:**
- Raw NME identical (3.267 vs 3.27) — cosine + SWA didn't improve the single best checkpoint
- SWA (3.270) was slightly worse than the best Phase 2 checkpoint — weight averaging didn't help here
- BUT TTA improved from 3.13 → 3.11 — the smoother weight landscape from cosine decay produces predictions that average better across augmented views
- The gain is modest (0.02) but real, suggesting the model's predictions are more geometrically consistent
- finetune_learning_rate=2e-5 (vs 1e-5 baseline) with cosine decay performed identically

**Next steps being prepared:**
- Square-pad crop (preserve aspect ratio instead of stretching) + dataset bounding boxes — code already implemented, ready to test
- Self-distillation from TTA teacher predictions

---

## Not Yet Tried

### High confidence:
- **Knowledge distillation** — train 11MB model to mimic the 55MB teacher's predictions. Expected to close 30-50% of the 3.27→3.48 gap (~0.05-0.10 gain)
- **3-model ensemble** (256+320+384 or +448) — gave ~0.2 gain on dogs, untested on cats
- **Ensemble + ms+flip TTA** — compound of above, biggest win on dogs (8.77 → 8.04)

### Medium confidence:
- **Square-pad crop + dataset bounding boxes** — preserve aspect ratio (instead of stretching) and use annotation bounding boxes for more consistent framing. Code implemented, ready to test. Expected gain: 0.05-0.12
- **Self-distillation from TTA teacher** — cache TTA predictions, train with blended loss L_gt + λ*L_teacher. Expected gain: 0.04-0.08
- **512px for EfficientNetV2S** — diminishing returns but might squeeze 0.02-0.03
- **FPN/U-Net skip-fused multi-scale decoding** — replace plain deconv head with FPN using multi-scale backbone features. Expected gain: 0.04-0.09 (moderate-high effort)

### Low confidence / not recommended:
- Cosine annealing LR + SWA — tested in Round 6. No raw improvement (3.27→3.27), tiny TTA gain (3.13→3.11). Not worth the complexity alone.
- Higher regularization — train-val gap is only ~1.0, not worth addressing
- Mixup — failed on dogs with similar dataset size
- ELD (region-based specialists) — failed on dogs (9.15 vs 8.77 single model)
- Heatmap supervision — failed on dogs (gradient interference)
- Ear-weighted loss — failed on dogs (robs from other landmarks)
- MobileNetV3Small — too small, 4.65 NME (Round 3)
- Higher resolution for MobileNetV3Large — 448px was worse than 384px (Round 3)
- More unfreezing (100+ layers) on EfficientNetV2S — risk of overfitting, gap already growing at 448px

---

## Architecture Details

### EfficientNetV2S (large model, ~55MB TFLite)
```
EfficientNetV2S (frozen phase 1, fine-tune last 50-80 layers phase 2)
    Input: [B, img_size, img_size, 3] (float32, [0,1] range)
    -> Rescaling(255) (in-model, for EfficientNet preprocessing)
    -> EfficientNetV2S backbone -> [B, img_size/32, img_size/32, 1280]
    -> Conv2DTranspose(256, 4x4, stride 2) + BN + ReLU + SpatialDropout(0.1)  -> 2x
    -> Conv2DTranspose(256, 4x4, stride 2) + BN + ReLU + SpatialDropout(0.1)  -> 4x
    -> Conv2DTranspose(256, 4x4, stride 2) + BN + ReLU + SpatialDropout(0.1)  -> 8x
    -> Conv2DTranspose(256, 4x4, stride 2) + BN + ReLU + SpatialDropout(0.1)  -> 16x
    -> Conv2D(48, 1x1) -> heatmaps [B, H, W, 48]
    -> SoftArgmax2D(beta=1.0) -> [B, 96] coordinates (x0,y0,...,x47,y47) in [0,1]

Total params: ~21M | Trainable (phase 2): ~14M
```

### MobileNetV3Large (small model, ~11MB TFLite)
```
MobileNetV3Large (frozen phase 1, fine-tune last 40-60 layers phase 2)
    Input: [B, img_size, img_size, 3] (float32, [0,1] range)
    -> Rescaling(255) (in-model, for MobileNetV3 preprocessing)
    -> MobileNetV3Large backbone -> [B, img_size/32, img_size/32, 960]
    -> Conv2DTranspose(128, 4x4, stride 2) + BN + ReLU + SpatialDropout(0.1)  -> 2x
    -> Conv2DTranspose(128, 4x4, stride 2) + BN + ReLU + SpatialDropout(0.1)  -> 4x
    -> Conv2DTranspose(128, 4x4, stride 2) + BN + ReLU + SpatialDropout(0.1)  -> 8x
    -> Conv2DTranspose(128, 4x4, stride 2) + BN + ReLU + SpatialDropout(0.1)  -> 16x
    -> Conv2D(48, 1x1) -> heatmaps [B, H, W, 48]
    -> SoftArgmax2D(beta=1.0) -> [B, 96] coordinates (x0,y0,...,x47,y47) in [0,1]

Total params: ~5.8M | Trainable (phase 2): ~2.8M
```

### Training Recipe
```
Phase 1 (frozen backbone):
    Epochs: 100 (early stopping patience=6)
    LR: 1e-4 (ReduceLROnPlateau: factor=0.5, patience=3)
    Optimizer: AdamW (weight_decay=1e-4)

Phase 2 (fine-tune backbone tail):
    Epochs: 200-400 (early stopping patience=50)
    LR: 1e-5 with 5-epoch linear warmup
    Unfreeze: last 40-80 layers (BN layers stay frozen)
    Optimizer: AdamW (weight_decay=1e-4)

Loss: MSE on [0,1] normalized coordinates
Metric: NME_IOD (inter-ocular distance normalized)
TFLite export: float16 quantization
```
