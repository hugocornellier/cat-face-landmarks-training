#!/usr/bin/env python3
"""Run the full landmark pipeline on a single image and save visualization.

Usage:
    python scripts/infer_single_image.py \
        --image path/to/cat.jpg \
        --bbox x1,y1,x2,y2 \
        [--tflite artifacts/tight_margin_256/cat_face_landmarks_256_float16.tflite] \
        [--out output.png]

If --bbox is omitted, a simple center-crop heuristic is used (assumes face
is roughly centered and occupies ~40-60% of the image).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

NUM_LANDMARKS = 48
IMG_SIZE = 256
CROP_MARGIN = 0.10

# Region colors for visualization
REGION_COLORS = {
    "right_eye":   (0, 255, 0),    # green
    "left_eye":    (0, 255, 0),    # green
    "right_ear":   (255, 165, 0),  # orange
    "left_ear":    (255, 165, 0),  # orange
    "nose":        (0, 200, 255),  # cyan
    "mouth_chin":  (255, 80, 80),  # red
}

REGIONS = {
    "right_eye": [3, 4, 5, 6, 7, 36, 37, 38],
    "left_eye": [1, 8, 9, 10, 11, 39, 40, 41],
    "right_ear": [22, 23, 24, 25, 26],
    "left_ear": [27, 28, 29, 30, 31],
    "nose": [12, 13, 14, 15, 32, 33, 34, 35, 42, 43, 44, 45],
    "mouth_chin": [0, 2, 16, 17, 18, 19, 20, 21, 46, 47],
}

# Build index -> color lookup
IDX_TO_COLOR = {}
for region, indices in REGIONS.items():
    for idx in indices:
        IDX_TO_COLOR[idx] = REGION_COLORS[region]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--bbox", type=str, default=None,
                   help="Face bounding box as x1,y1,x2,y2 in pixel coords")
    p.add_argument("--tflite", type=Path,
                   default=Path("artifacts/tight_margin_256/cat_face_landmarks_256_float16.tflite"))
    p.add_argument("--out", type=Path, default=None,
                   help="Output path (default: <image>_landmarks.png)")
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--crop-margin", type=float, default=CROP_MARGIN)
    return p.parse_args()


def crop_and_normalize_np(image_np: np.ndarray, bbox: tuple[float, float, float, float],
                          img_size: int, crop_margin: float):
    """Crop image to bbox + margin, resize to (img_size, img_size).
    Returns: (crop_rgb_float32 [0,1], crop_coords (cx1, cy1, cx2, cy2))
    """
    h, w = image_np.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    mx = bw * crop_margin
    my = bh * crop_margin
    cx1 = max(0, int(np.floor(x1 - mx)))
    cy1 = max(0, int(np.floor(y1 - my)))
    cx2 = min(w, int(np.ceil(x2 + mx)))
    cy2 = min(h, int(np.ceil(y2 + my)))

    cropped = image_np[cy1:cy2, cx1:cx2]
    pil_crop = Image.fromarray(cropped).resize((img_size, img_size), Image.LANCZOS)
    crop_float = np.array(pil_crop, dtype=np.float32) / 255.0
    return crop_float, (cx1, cy1, cx2, cy2)


def run_tflite(tflite_path: Path, crop_float: np.ndarray):
    """Run TFLite inference. Input: [H, W, 3] float32 in [0,1]. Output: [96] coords."""
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    inp = np.expand_dims(crop_float, 0).astype(in_det["dtype"])
    interp.set_tensor(in_det["index"], inp)
    interp.invoke()
    pred = interp.get_tensor(out_det["index"])[0].astype(np.float32)
    return np.clip(pred, 0.0, 1.0)


def landmarks_to_image_coords(pred_flat: np.ndarray,
                               crop_box: tuple[int, int, int, int]):
    """Map [0,1] crop-normalized landmarks back to original image pixel coords."""
    cx1, cy1, cx2, cy2 = crop_box
    cw, ch = cx2 - cx1, cy2 - cy1
    pts = pred_flat.reshape(NUM_LANDMARKS, 2)
    img_pts = np.zeros_like(pts)
    img_pts[:, 0] = pts[:, 0] * cw + cx1  # x
    img_pts[:, 1] = pts[:, 1] * ch + cy1  # y
    return img_pts


def draw_landmarks(image: Image.Image, landmarks: np.ndarray, crop_box=None):
    """Draw colored landmarks on the image. Returns a new image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Draw crop box
    if crop_box is not None:
        cx1, cy1, cx2, cy2 = crop_box
        draw.rectangle([cx1, cy1, cx2, cy2], outline=(255, 255, 0), width=2)

    # Dot radius scales with image size
    r = max(3, min(image.width, image.height) // 200)

    for i in range(NUM_LANDMARKS):
        x, y = landmarks[i]
        color = IDX_TO_COLOR.get(i, (255, 255, 255))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(0, 0, 0))

    # Draw connecting lines for eyes
    for region, indices in REGIONS.items():
        color = REGION_COLORS[region]
        if len(indices) >= 2:
            for a, b in zip(indices[:-1], indices[1:]):
                ax, ay = landmarks[a]
                bx, by = landmarks[b]
                draw.line([(ax, ay), (bx, by)], fill=color, width=max(1, r // 2))

    return img


def main():
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.tflite.exists():
        raise FileNotFoundError(f"TFLite model not found: {args.tflite}")

    # Load image
    pil_img = Image.open(args.image).convert("RGB")
    image_np = np.array(pil_img)
    h, w = image_np.shape[:2]
    print(f"Image: {args.image} ({w}x{h})")

    # Get bounding box
    if args.bbox:
        bbox = tuple(float(v) for v in args.bbox.split(","))
        assert len(bbox) == 4, "bbox must be x1,y1,x2,y2"
    else:
        # Heuristic: assume face is roughly centered, ~50% of image
        cx, cy = w * 0.5, h * 0.4
        size = min(w, h) * 0.35
        bbox = (cx - size, cy - size, cx + size, cy + size)
        print(f"No --bbox provided, using heuristic: {bbox}")

    print(f"Bounding box: ({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})")

    # Crop + normalize
    crop_float, crop_box = crop_and_normalize_np(
        image_np, bbox, args.img_size, args.crop_margin
    )
    print(f"Crop region: {crop_box}")

    # Run TFLite
    print(f"Running TFLite model: {args.tflite}")
    pred_flat = run_tflite(args.tflite, crop_float)

    # Map back to image coordinates
    landmarks = landmarks_to_image_coords(pred_flat, crop_box)
    print(f"Predicted {NUM_LANDMARKS} landmarks")

    # Print some stats
    for region, indices in REGIONS.items():
        region_pts = landmarks[indices]
        cx = np.mean(region_pts[:, 0])
        cy = np.mean(region_pts[:, 1])
        print(f"  {region:<16s}  center=({cx:.0f}, {cy:.0f})")

    # Draw and save
    result_img = draw_landmarks(pil_img, landmarks, crop_box)
    out_path = args.out or args.image.with_name(args.image.stem + "_landmarks.png")
    result_img.save(out_path)
    print(f"\nSaved: {out_path}")

    # Also save the crop with landmarks for inspection
    crop_pil = Image.fromarray((crop_float * 255).astype(np.uint8))
    crop_landmarks = pred_flat.reshape(NUM_LANDMARKS, 2)
    crop_landmarks_px = crop_landmarks * args.img_size
    crop_result = draw_landmarks_on_crop(crop_pil, crop_landmarks_px)
    crop_out = out_path.with_name(out_path.stem + "_crop.png")
    crop_result.save(crop_out)
    print(f"Saved crop: {crop_out}")


def draw_landmarks_on_crop(crop_img: Image.Image, landmarks_px: np.ndarray):
    """Draw landmarks on the cropped face image."""
    img = crop_img.copy()
    draw = ImageDraw.Draw(img)
    r = 2
    for i in range(NUM_LANDMARKS):
        x, y = landmarks_px[i]
        color = IDX_TO_COLOR.get(i, (255, 255, 255))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(0, 0, 0))
    return img


if __name__ == "__main__":
    main()
