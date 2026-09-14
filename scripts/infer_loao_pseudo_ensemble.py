#!/usr/bin/env python3
"""Generate calibrated pseudo-label candidates with animal-excluded ensembles."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def parse_video(value):
    digits = "".join(character for character in str(value).split("_")[0] if character.isdigit())
    return f"V{int(digits):03d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manual-release", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--temporal-radius", type=int, default=30)
    args = parser.parse_args()

    import cv2
    import numpy as np
    import pandas as pd
    import torch
    import segmentation_models_pytorch as smp
    from PIL import Image

    gv = load_module("grouped_validation", args.project_root / "grouped_validation.py")
    core = load_module("loao_pseudo_release_core", args.project_root / "publication_package/scripts/loao_pseudo_release_core.py")
    calibration = json.loads((args.teacher_root / "validation/calibration_summary.json").read_text())
    class_thresholds = {
        int(class_id): details["confidence_min"]
        for class_id, details in calibration["class_confidence_thresholds"].items()
        if int(class_id) != 1
    }
    entropy_max = calibration["normalized_entropy_max"]
    disagreement_max = calibration["ensemble_disagreement_max"]

    manual = pd.read_csv(args.manual_release / "metadata/canonical_manual_manifest.csv")
    video_animal = dict(zip(manual.video_id, manual.anonymized_animal_id))
    manual_hashes = set(manual.image_sha256)
    manual_frames = {
        video: np.asarray(sorted(group.frame_number.astype(int)), dtype=int)
        for video, group in manual.groupby("video_id")
    }
    candidates = pd.read_csv(args.candidate_manifest)
    candidates["release_video_id"] = candidates.video_id.map(parse_video)
    candidates["animal"] = candidates.release_video_id.map(video_animal)
    if candidates.animal.isna().any():
        missing = sorted(candidates.loc[candidates.animal.isna(), "video_id"].unique())
        raise ValueError(f"Unmapped candidate videos: {missing}")
    animals = sorted(candidates.animal.unique())[args.shard_index::args.num_shards]
    device = torch.device("cuda:0")
    output_masks = args.output / "candidate_masks"
    output_metrics = args.output / "candidate_metrics"
    output_masks.mkdir(parents=True, exist_ok=True)
    output_metrics.mkdir(parents=True, exist_ok=True)

    for animal in animals:
        metrics_path = output_metrics / f"{animal}.csv"
        done_path = output_metrics / f"{animal}.complete"
        if done_path.exists() and metrics_path.exists():
            print(f"SKIP complete {animal}", flush=True)
            continue
        models, checkpoint_hashes = [], []
        for seed in (42, 1337, 2026):
            checkpoint = args.teacher_root / "teachers" / animal / f"seed_{seed}" / "model.pt"
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            if payload["held_out_animal"] != animal or animal in payload["training_animals"]:
                raise AssertionError(f"Teacher leakage in {checkpoint}")
            checkpoint_hashes.append(hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=4).to(device)
            model.load_state_dict(payload["model_state_dict"])
            model.eval(); models.append(model)

        subset = candidates[candidates.animal == animal].sort_values("stem").reset_index(drop=True)
        rows = []
        print(f"INFER {animal} candidates={len(subset)}", flush=True)
        for start in range(0, len(subset), args.batch_size):
            batch_rows = [row for _, row in subset.iloc[start:start + args.batch_size].iterrows()]
            images, originals, raw_bytes = [], [], []
            for row in batch_rows:
                path = Path(row.image_path)
                if not path.is_absolute():
                    path = args.project_root / path
                data = path.read_bytes()
                image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Cannot decode {path}")
                originals.append(image); raw_bytes.append(data)
                images.append(gv.preprocess(image, 512, 512))
            tensor = torch.from_numpy(np.stack(images).astype(np.float32)).to(device)
            sum_probability = torch.zeros((len(batch_rows), 4, 512, 512), device=device)
            sum_squared = torch.zeros_like(sum_probability)
            with torch.no_grad():
                for model in models:
                    with torch.amp.autocast("cuda"):
                        probability = torch.softmax(model(tensor), dim=1).float()
                    sum_probability += probability
                    sum_squared += probability.square()
            mean_probability = sum_probability / len(models)
            prediction = mean_probability.argmax(dim=1).cpu().numpy()
            confidence = mean_probability.max(dim=1).values.cpu().numpy()
            entropy = (-(mean_probability.clamp_min(1e-12) * mean_probability.clamp_min(1e-12).log()).sum(dim=1) / np.log(4.0)).cpu().numpy()
            disagreement = ((sum_squared / len(models) - mean_probability.square()).mean(dim=1)).cpu().numpy()

            for index, row in enumerate(batch_rows):
                video = row.release_video_id
                frame = int(row.frame_idx)
                nearest_distance = int(np.min(np.abs(manual_frames[video] - frame)))
                digest = sha256_bytes(raw_bytes[index])
                exclusion = ""
                if digest in manual_hashes:
                    exclusion = "exact_manual_sha256"
                elif nearest_distance <= args.temporal_radius:
                    exclusion = "within_temporal_radius_of_manual"
                mask_512 = core.calibrated_publication_mask(
                    prediction[index], confidence[index], entropy[index], disagreement[index],
                    class_thresholds, entropy_max, disagreement_max,
                )
                image = originals[index]
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                artifact = ((hsv[:, :, 2] >= 245) & (hsv[:, :, 1] <= 45)) | (hsv[:, :, 2] <= 38)
                mask = cv2.resize(mask_512, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask[artifact] = 255
                valid = mask != 255
                valid_ratio = float(valid.mean())
                accepted_512 = mask_512 != 255
                mean_confidence = float(confidence[index][accepted_512].mean()) if accepted_512.any() else 0.0
                mean_entropy = float(entropy[index][accepted_512].mean()) if accepted_512.any() else 1.0
                mean_disagreement = float(disagreement[index][accepted_512].mean()) if accepted_512.any() else 1.0
                fractions = {class_id: float((mask == class_id + 1).mean()) for class_id in (0, 2, 3)}
                eligible = not exclusion and valid_ratio >= 0.20 and mean_confidence >= 0.80
                if eligible:
                    Image.fromarray(mask).save(output_masks / f"{row.stem}.png")
                rows.append({
                    "stem": row.stem, "video_id": video, "animal_id": animal,
                    "frame_number": frame, "source_image_path": row.image_path,
                    "image_sha256": digest, "nearest_manual_frame_distance": nearest_distance,
                    "manual_temporal_radius": args.temporal_radius, "exclusion_reason": exclusion,
                    "eligible_after_calibration": eligible, "valid_ratio": valid_ratio,
                    "mean_confidence": mean_confidence, "mean_normalized_entropy": mean_entropy,
                    "mean_ensemble_disagreement": mean_disagreement,
                    "frac_mes0": fractions[0], "frac_mes1": 0.0,
                    "frac_mes2": fractions[2], "frac_mes3": fractions[3],
                    "mes1_policy": "excluded_due_to_oof_precision_below_0.80",
                    "teacher_seeds": "42;1337;2026",
                    "teacher_checkpoint_sha256": ";".join(checkpoint_hashes),
                })
            if start % (args.batch_size * 100) == 0:
                print(f"{animal} {min(start + args.batch_size, len(subset))}/{len(subset)}", flush=True)
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        done_path.write_text(f"rows={len(rows)}\n")
        print(f"DONE {animal} eligible={sum(bool(row['eligible_after_calibration']) for row in rows)}", flush=True)
        del models
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
