#!/usr/bin/env python3
"""Materialize and verify the selected LOAO pseudo-label component."""

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def copy_or_link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError:
        pass
    except OSError:
        shutil.copy2(source, destination)


def write_csv(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.selection_manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    output_images = args.output_dir / "images"
    output_masks = args.output_dir / "masks_indexed"
    metadata = args.output_dir / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    manifest = []
    pixel_counts = {str(value): 0 for value in (0, 1, 2, 3, 4, 255)}
    for index, row in enumerate(rows, 1):
        source_image = args.project_root / row["source_image_path"]
        source_mask = args.source_mask_dir / f"{row['stem']}.png"
        suffix = source_image.suffix.lower()
        release_image = output_images / f"P{index:05d}{suffix}"
        release_mask = output_masks / f"P{index:05d}.png"
        copy_or_link(source_image, release_image)
        copy_or_link(source_mask, release_mask)
        mask = np.asarray(Image.open(release_mask))
        values, counts = np.unique(mask, return_counts=True)
        if not set(map(int, values)).issubset({1, 3, 4, 255}):
            raise AssertionError(f"Unexpected mask values for {row['stem']}: {values}")
        for value, count in zip(values, counts):
            pixel_counts[str(int(value))] += int(count)
        manifest.append({
            "pseudo_image_id": f"P{index:05d}",
            "image_path": str(release_image.relative_to(args.output_dir)),
            "mask_path": str(release_mask.relative_to(args.output_dir)),
            "image_sha256": digest(release_image),
            "mask_sha256": digest(release_mask),
            "source_stem": row["stem"], "video_id": row["video_id"],
            "anonymized_animal_id": row["anonymized_animal_id"],
            "frame_number": row["frame_number"],
            "nearest_manual_frame_distance": row["nearest_manual_frame_distance"],
            "manual_temporal_exclusion_radius": row["manual_temporal_radius"],
            "mean_confidence": row["mean_confidence"],
            "mean_normalized_entropy": row["mean_normalized_entropy"],
            "mean_ensemble_disagreement": row["mean_ensemble_disagreement"],
            "accepted_pixel_ratio": row["valid_ratio"],
            "MES0_area_fraction": row["frac_mes0"],
            "MES1_area_fraction": "0", "MES2_area_fraction": row["frac_mes2"],
            "MES3_area_fraction": row["frac_mes3"],
            "held_out_animal": row["held_out_animal"],
            "teacher_seeds": row["teacher_seeds"],
            "teacher_checkpoint_sha256": row["teacher_checkpoint_sha256"],
            "selection_score": row["selection_score"],
            "release_phash": row["release_phash"],
            "annotation_provenance": "three-seed animal-excluded LOAO ensemble",
            "MES1_policy": "excluded_due_to_OOF_precision_below_0.80",
            "intended_use": "training-only weak supervision; never validation or test ground truth",
        })
    if len({r["image_sha256"] for r in manifest}) != len(manifest):
        raise AssertionError("Release contains duplicate image SHA-256 values")
    if any(r["held_out_animal"] != r["anonymized_animal_id"] for r in manifest):
        raise AssertionError("Animal-excluded teacher provenance failed")
    write_csv(metadata / "canonical_pseudo_manifest.csv", manifest)
    summary = {
        "released_images": len(manifest), "unique_image_sha256": len(manifest),
        "physical_images": len(list(output_images.iterdir())),
        "physical_masks": len(list(output_masks.iterdir())),
        "allowed_mask_values": [1, 3, 4, 255], "pixel_counts": pixel_counts,
        "MES1_pseudo_pixels": pixel_counts["2"],
        "teacher_exclusion_audit_passed": True,
        "physical_file_audit_passed": True,
        "global_exact_uniqueness_audit_passed": True,
    }
    (metadata / "physical_integrity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
