#!/usr/bin/env python3
"""Fail-closed audit of the finalized LOAO pseudo-label manifest."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    with args.selection_audit.open(newline="", encoding="utf-8-sig") as handle:
        audit = {r["stem"]: r for r in csv.DictReader(handle)}
    failures = []
    selected_audit = []
    for row in rows:
        source = row["source_stem"]
        record = audit.get(source)
        if record is None:
            failures.append(f"missing_selection_audit:{source}")
            continue
        selected_audit.append(record)
        if int(float(row["nearest_manual_frame_distance"])) <= int(row["manual_temporal_exclusion_radius"]):
            failures.append(f"temporal_overlap:{source}")
        if row["held_out_animal"] != row["anonymized_animal_id"]:
            failures.append(f"teacher_animal_mismatch:{source}")
        if len(row["teacher_seeds"].split(";")) != 3:
            failures.append(f"teacher_seed_count:{source}")
        if len(row["teacher_checkpoint_sha256"].split(";")) != 3:
            failures.append(f"teacher_hash_count:{source}")
    result = {
        "released_images": len(rows),
        "manual_temporal_exclusion_passed": not any(x.startswith("temporal_overlap") for x in failures),
        "teacher_animal_exclusion_passed": not any(x.startswith("teacher_animal") for x in failures),
        "three_teacher_provenance_passed": not any("teacher_" in x and "mismatch" not in x for x in failures),
        "selection_audit_linkage_passed": len(selected_audit) == len(rows),
        "all_required_audits_passed": not failures,
        "failure_count": len(failures), "failures": failures[:100],
        "minimum_manual_frame_distance": min(int(float(r["nearest_manual_frame_distance"])) for r in rows),
        "temporal_exclusion_radius": min(int(r["manual_temporal_exclusion_radius"]) for r in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
