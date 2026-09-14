#!/usr/bin/env python3
"""Select a leakage-audited LOAO pseudo-label release without a target count."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def truthy(value):
    return str(value).lower() in {"1", "true", "yes"}


def read_rows(paths):
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def phash(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    coeff = cv2.dct(small.astype(np.float32))[:8, :8]
    bits = coeff > np.median(coeff[1:, :])
    return int(np.packbits(bits.ravel()).view(np.uint64)[0])


def score(row):
    return (float(row["mean_confidence"])
            - float(row["mean_normalized_entropy"])
            - 5.0 * float(row["mean_ensemble_disagreement"])
            + 0.2 * float(row["valid_ratio"]))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--visual-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-area", type=float, default=0.015)
    parser.add_argument("--phash-gap", type=int, default=4)
    args = parser.parse_args()

    rows = read_rows(sorted(args.metrics_dir.glob("A*.csv")))
    audit = read_rows([args.visual_audit])
    audit_by_stem = {row["stem"]: row for row in audit}
    audit_rows = []
    candidates = []
    for row in rows:
        reasons = []
        if not truthy(row["eligible_after_calibration"]):
            reasons.append(row.get("exclusion_reason") or "failed_calibration")
        if max(float(row["frac_mes0"]), float(row["frac_mes2"]),
               float(row["frac_mes3"])) < args.min_area:
            reasons.append("insufficient_released_mes_area")
        previous = audit_by_stem.get(row["stem"])
        if previous is None:
            reasons.append("manual_visual_audit_missing")
        elif truthy(previous.get("excluded_visual_manual_overlap")):
            reasons.append("visual_manual_overlap")
        image = args.project_root / row["source_image_path"]
        mask = args.mask_dir / f"{row['stem']}.png"
        if not image.is_file():
            reasons.append("source_image_missing")
        if not mask.is_file():
            reasons.append("pseudo_mask_missing")
        record = {**row, "selection_status": "rejected" if reasons else "candidate",
                  "selection_reason": ";".join(reasons)}
        audit_rows.append(record)
        if not reasons:
            candidates.append(record)

    ranked = sorted(candidates, key=lambda row: (-score(row), row["stem"]))
    unique = []
    seen_sha = {}
    for row in ranked:
        digest = row["image_sha256"]
        if digest in seen_sha:
            row["selection_status"] = "rejected"
            row["selection_reason"] = "duplicate_sha256"
            row["duplicate_of"] = seen_sha[digest]
            continue
        seen_sha[digest] = row["stem"]
        unique.append(row)

    selected = []
    hashes = []
    for row in unique:
        digest = phash(args.project_root / row["source_image_path"])
        if any((digest ^ old).bit_count() < args.phash_gap for old in hashes):
            row["selection_status"] = "rejected"
            row["selection_reason"] = "global_perceptual_near_duplicate"
            continue
        row["selection_status"] = "selected"
        row["selection_reason"] = "passed_all_prespecified_criteria"
        row["selection_score"] = f"{score(row):.10f}"
        row["release_phash"] = f"{digest:016x}"
        row["anonymized_animal_id"] = row.pop("animal_id")
        row["held_out_animal"] = row["anonymized_animal_id"]
        row["frac_1"] = "0"
        selected.append(row)
        hashes.append(digest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "selection_audit.csv", audit_rows)
    write_csv(args.output_dir / "selected_manifest.csv", selected)
    summary = {
        "selection_count_not_fixed_in_advance": True,
        "input_rows": len(rows), "calibrated_and_area_eligible": len(candidates),
        "unique_sha_candidates": len(unique), "selected_images": len(selected),
        "minimum_released_class_area_fraction": args.min_area,
        "global_phash_minimum_hamming_distance": args.phash_gap,
        "mes1_pseudo_pixels_released": 0,
        "selected_images_by_class_presence": {
            f"MES{c}": sum(float(r[f"frac_mes{c}"]) >= args.min_area for r in selected)
            for c in (0, 2, 3)
        },
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
