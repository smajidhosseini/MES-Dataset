#!/usr/bin/env python3
"""Reproducible signal-quality analysis for the finalized MES manual release."""

import csv
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(".")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Operational challenging-image thresholds, fixed before running this script.
# These are screening criteria, not clinical acceptability thresholds.
SHARPNESS_LOW = 10.0                 # variance of OpenCV default Laplacian
BRIGHTNESS_LOW = 50.0                # mean 8-bit grayscale intensity
BRIGHTNESS_HIGH = 150.0
GLARE_HIGH = 0.05                    # >5% of pixels
GLARE_V_MIN = 245                    # HSV value
GLARE_S_MAX = 45                     # HSV saturation


def video_id(stem):
    return stem.rsplit("-", 1)[0] if "-" in stem else stem


def shannon_entropy(gray):
    counts = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def measure(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to decode {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = float(gray.mean())
    contrast = float(gray.std(ddof=0))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    entropy = shannon_entropy(gray)
    glare = (hsv[:, :, 2] >= GLARE_V_MIN) & (hsv[:, :, 1] <= GLARE_S_MAX)
    glare_ratio = float(glare.mean())
    low_sharpness = sharpness < SHARPNESS_LOW
    extreme_brightness = brightness < BRIGHTNESS_LOW or brightness > BRIGHTNESS_HIGH
    elevated_glare = glare_ratio > GLARE_HIGH
    return {
        "image_id": path.stem,
        "video_id": video_id(path.stem),
        "width": image.shape[1],
        "height": image.shape[0],
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "entropy": entropy,
        "glare_ratio": glare_ratio,
        "low_sharpness": int(low_sharpness),
        "extreme_brightness": int(extreme_brightness),
        "elevated_glare": int(elevated_glare),
        "challenging_any": int(low_sharpness or extreme_brightness or elevated_glare),
    }


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Root of the extracted Figshare release")
    args = parser.parse_args()
    image_dir = args.root / "manual" / "images"
    out_dir = args.root / "analysis" / "signal_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    rows = [measure(path) for path in paths]
    fields = list(rows[0])
    with (out_dir / "signal_quality_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metrics = ["brightness", "contrast", "sharpness", "entropy", "glare_ratio"]
    summary = {metric: summarize([row[metric] for row in rows]) for metric in metrics}
    flags = {
        name: {
            "count": int(sum(row[name] for row in rows)),
            "percent": float(100 * sum(row[name] for row in rows) / len(rows)),
        }
        for name in ("low_sharpness", "extreme_brightness", "elevated_glare", "challenging_any")
    }
    result = {
        "images": len(rows),
        "image_dimensions": dict(Counter(f"{row['width']}x{row['height']}" for row in rows)),
        "metric_definitions": {
            "brightness": "Mean 8-bit grayscale intensity over the complete cropped frame.",
            "contrast": "Population standard deviation of 8-bit grayscale intensity.",
            "sharpness": "Variance of the OpenCV default-kernel Laplacian of the grayscale image.",
            "entropy": "Base-2 Shannon entropy of the 256-bin grayscale histogram, in bits.",
            "glare_ratio": f"Fraction of pixels with HSV value >= {GLARE_V_MIN} and saturation <= {GLARE_S_MAX}.",
        },
        "thresholds": {
            "low_sharpness": f"sharpness < {SHARPNESS_LOW}",
            "extreme_brightness": f"brightness < {BRIGHTNESS_LOW} or brightness > {BRIGHTNESS_HIGH}",
            "elevated_glare": f"glare_ratio > {GLARE_HIGH}",
            "challenging_any": "At least one operational criterion met.",
        },
        "summary": summary,
        "flags": flags,
    }
    (out_dir / "signal_quality_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    with (out_dir / "signal_quality_per_video.csv").open("w", newline="", encoding="utf-8") as handle:
        video_fields = ["video_id", "n"] + [f"{m}_mean" for m in metrics] + ["challenging_count", "challenging_percent"]
        writer = csv.DictWriter(handle, fieldnames=video_fields)
        writer.writeheader()
        for vid in sorted(by_video):
            items = by_video[vid]
            row = {"video_id": vid, "n": len(items)}
            row.update({f"{m}_mean": np.mean([item[m] for item in items]) for m in metrics})
            count = sum(item["challenging_any"] for item in items)
            row.update({"challenging_count": count, "challenging_percent": 100 * count / len(items)})
            writer.writerow(row)

    labels = {
        "brightness": "Brightness (mean grayscale intensity)",
        "contrast": "Contrast (grayscale SD)",
        "sharpness": "Sharpness (Laplacian variance)",
        "entropy": "Entropy (bits)",
        "glare_ratio": "Glare ratio",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        values = [row[metric] for row in rows]
        ax.hist(values, bins=40, color="#3465A4", edgecolor="white", linewidth=0.4)
        ax.axvline(np.mean(values), color="#B2182B", linestyle="--", linewidth=1.5, label="Mean")
        ax.set_xlabel(labels[metric])
        ax.set_ylabel("Frames")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    axes[-1].axis("off")
    fig.suptitle(f"Signal-quality distributions across {len(rows):,} manually annotated frames", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "signal_quality_distributions.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "signal_quality_distributions.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
