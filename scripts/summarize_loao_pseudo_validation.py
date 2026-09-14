#!/usr/bin/env python3
"""Combine OOF calibration and selected-release uncertainty evidence."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def describe(values):
    data = np.asarray(values, dtype=float)
    return {"mean": float(data.mean()), "sd": float(data.std(ddof=1)),
            "median": float(np.median(data)), "q05": float(np.quantile(data, .05)),
            "q95": float(np.quantile(data, .95))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--physical-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    leakage = json.loads(args.leakage_audit.read_text(encoding="utf-8"))
    physical = json.loads(args.physical_audit.read_text(encoding="utf-8"))
    summary = {
        "validation_scope": "No new expert annotations were used. OOF manual performance calibrates thresholds; ensemble uncertainty summarizes selected pseudo-labels.",
        "released_images": len(rows),
        "selected_release_uncertainty": {
            "mean_confidence": describe([r["mean_confidence"] for r in rows]),
            "mean_normalized_entropy": describe([r["mean_normalized_entropy"] for r in rows]),
            "mean_ensemble_disagreement": describe([r["mean_ensemble_disagreement"] for r in rows]),
            "accepted_pixel_ratio": describe([r["accepted_pixel_ratio"] for r in rows]),
        },
        "out_of_fold_calibration": calibration,
        "MES1_release_policy": "Excluded because OOF precision was 0.643 at confidence 0.99, below the prespecified 0.80 target.",
        "leakage_audit": leakage,
        "physical_integrity_audit": physical,
        "all_release_gates_passed": bool(leakage["all_required_audits_passed"]
                                          and physical["physical_file_audit_passed"]
                                          and physical["global_exact_uniqueness_audit_passed"]
                                          and physical["MES1_pseudo_pixels"] == 0),
        "interpretation_limit": "These checks quantify calibration, uncertainty, consistency among independent-seed teachers, and leakage controls; they do not replace independent expert validation of pseudo-label accuracy.",
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"released_images": len(rows),
                      "all_release_gates_passed": summary["all_release_gates_passed"],
                      "selected_release_uncertainty": summary["selected_release_uncertainty"]}, indent=2))
    if not summary["all_release_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
