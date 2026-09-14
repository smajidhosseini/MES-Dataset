#!/usr/bin/env python3
"""Pure helpers shared by the leakage-safe LOAO pseudo-label pipeline."""

import numpy as np


def build_video_animal_map(rows):
    mapping = {}
    for row in rows:
        video = row["video_id"]
        animal = row["anonymized_animal_id"]
        previous = mapping.setdefault(video, animal)
        if previous != animal:
            raise ValueError(f"Video {video} maps to both {previous} and {animal}")
    return mapping


def teacher_assignment(video_id, video_animal_map, all_animals, seeds=(42, 1337, 2026)):
    if video_id not in video_animal_map:
        raise KeyError(f"Candidate video is not mapped: {video_id}")
    held_out = video_animal_map[video_id]
    training = sorted(set(all_animals) - {held_out})
    if held_out in training:
        raise AssertionError("Held-out animal leaked into teacher training animals")
    return {
        "held_out_animal": held_out,
        "training_animals": training,
        "seeds": list(seeds),
    }


def unique_by_sha256(rows):
    seen = {}
    unique = []
    rejected = []
    for row in rows:
        digest = row["image_sha256"]
        if digest in seen:
            rejected.append({
                "stem": row["stem"],
                "duplicate_of": seen[digest],
                "reason": "duplicate_sha256",
            })
            continue
        seen[digest] = row["stem"]
        unique.append(row)
    return unique, rejected


def rank_score(row):
    """Deterministic quality score used before diversity selection."""
    return (
        float(row["mean_confidence"])
        - float(row["mean_normalized_entropy"])
        - 5.0 * float(row["mean_disagreement"])
        + 0.2 * float(row["valid_ratio"])
    )


def assert_release_integrity(rows, expected_count=None):
    """Fail closed on the invariants required for publication release."""
    if not rows:
        raise AssertionError("Pseudo-label release is empty")
    if expected_count is not None and len(rows) != expected_count:
        raise AssertionError(f"Expected {expected_count} rows, found {len(rows)}")
    for key in ("image_sha256", "release_image_path", "release_mask_path"):
        values = [row[key] for row in rows]
        if len(values) != len(set(values)):
            raise AssertionError(f"Duplicate {key} in release")
    if any(float(row.get("frac_1", 0) or 0) != 0 for row in rows):
        raise AssertionError("MES 1 pseudo-pixels must be excluded")
    if any(row.get("held_out_animal") != row.get("anonymized_animal_id") for row in rows):
        raise AssertionError("Teacher-exclusion provenance mismatch")


def teacher_jobs(animals, seeds=(42, 1337, 2026)):
    return [
        {"animal": animal, "seed": int(seed)}
        for animal in sorted(animals)
        for seed in seeds
    ]


def ensemble_statistics(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim < 3 or probabilities.shape[1] != 4:
        raise ValueError("Expected shape (models, 4, ...)")
    mean_probability = probabilities.mean(axis=0)
    prediction = mean_probability.argmax(axis=0)
    confidence = mean_probability.max(axis=0)
    clipped = np.clip(mean_probability, 1e-12, 1.0)
    normalized_entropy = -(clipped * np.log(clipped)).sum(axis=0) / np.log(4.0)
    disagreement = probabilities.var(axis=0).mean(axis=0)
    return {
        "mean_probability": mean_probability,
        "prediction": prediction,
        "confidence": confidence,
        "normalized_entropy": normalized_entropy,
        "disagreement": disagreement,
    }


def calibrated_publication_mask(
    prediction, confidence, normalized_entropy, disagreement,
    class_thresholds, entropy_max, disagreement_max,
):
    prediction = np.asarray(prediction)
    accepted = (
        (np.asarray(normalized_entropy) <= entropy_max)
        & (np.asarray(disagreement) <= disagreement_max)
    )
    output = np.full(prediction.shape, 255, dtype=np.uint8)
    for class_id, threshold in class_thresholds.items():
        keep = accepted & (prediction == int(class_id)) & (np.asarray(confidence) >= threshold)
        output[keep] = int(class_id) + 1
    return output
