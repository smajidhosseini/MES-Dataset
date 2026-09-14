#!/usr/bin/env python3
"""Calibrate animal-excluded three-seed teachers on the 953 expert labels."""

import argparse
import importlib.util
import json
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-precision", type=float, default=0.80)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    import segmentation_models_pytorch as smp
    from torch.utils.data import DataLoader

    gv = load_module("grouped_validation", args.project_root / "grouped_validation.py")
    core = load_module("loao_pseudo_release_core", args.project_root / "publication_package/scripts/loao_pseudo_release_core.py")
    canonical = pd.read_csv(args.release / "metadata/canonical_manual_manifest.csv")
    mapping = dict(zip(canonical.video_id, canonical.anonymized_animal_id))
    video_to_animal = {video.replace("V", "", 1) + "_mp4": animal for video, animal in mapping.items()}
    frames = gv.build_manifest(args.release / "manual/images", args.release / "manual/masks_indexed")
    frames["animal"] = frames.video_id.map(video_to_animal)
    device = torch.device("cuda:0")
    all_gt, all_pred, all_conf, all_entropy, all_disagreement = [], [], [], [], []
    animal_rows = []

    for animal in sorted(frames.animal.unique()):
        models = []
        for seed in (42, 1337, 2026):
            checkpoint = args.teacher_root / "teachers" / animal / f"seed_{seed}" / "model.pt"
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            if payload["held_out_animal"] != animal or animal in payload["training_animals"]:
                raise AssertionError(f"Teacher leakage in {checkpoint}")
            model = smp.DeepLabV3Plus(
                encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=4,
            ).to(device)
            model.load_state_dict(payload["model_state_dict"])
            model.eval()
            models.append(model)
        held = frames[frames.animal == animal].reset_index(drop=True)
        loader = DataLoader(gv.MESDataset(held, 512, 512, "eval"), batch_size=12, shuffle=False, num_workers=0)
        confusion = np.zeros((4, 4), dtype=np.int64)
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                probabilities = []
                for model in models:
                    with torch.amp.autocast("cuda"):
                        probabilities.append(torch.softmax(model(images), dim=1).float().cpu().numpy())
                stats = core.ensemble_statistics(np.stack(probabilities).transpose(0, 2, 1, 3, 4))
                gt = targets.numpy()
                valid = gt != gv.IGNORE_INDEX
                predicted = stats["prediction"][valid]
                truth = gt[valid]
                confidence = stats["confidence"][valid]
                entropy = stats["normalized_entropy"][valid]
                disagreement = stats["disagreement"][valid]
                confusion += np.bincount(truth * 4 + predicted, minlength=16).reshape(4, 4)
                all_gt.append(truth); all_pred.append(predicted); all_conf.append(confidence)
                all_entropy.append(entropy); all_disagreement.append(disagreement)
        metrics = gv.metrics_from_conf(confusion, gv.present_classes_from_df(held))
        animal_rows.append({
            "animal": animal, "n_images": len(held), "annotated_pixels": int(confusion.sum()),
            "mdice": metrics["mdice"], "pixel_accuracy": metrics["pix_acc"],
            **{f"dice_{c}": float(metrics["dice_pc"][c]) for c in range(4)},
        })
        print(f"CALIBRATED {animal} mdice={metrics['mdice']:.4f}", flush=True)
        del models
        torch.cuda.empty_cache()

    gt = np.concatenate(all_gt); pred = np.concatenate(all_pred)
    confidence = np.concatenate(all_conf); entropy = np.concatenate(all_entropy)
    disagreement = np.concatenate(all_disagreement)
    thresholds = {}
    curves = []
    grid = np.arange(0.50, 1.00, 0.01)
    for class_id in range(4):
        candidates = []
        for threshold in grid:
            keep = (pred == class_id) & (confidence >= threshold)
            count = int(keep.sum())
            precision = float((gt[keep] == class_id).mean()) if count else 0.0
            candidates.append((threshold, count, precision))
            curves.append({"class_id": class_id, "confidence_threshold": threshold, "retained_pixels": count, "precision": precision})
        qualifying = [item for item in candidates if item[1] >= 1000 and item[2] >= args.target_precision]
        chosen = qualifying[0] if qualifying else max((item for item in candidates if item[1] >= 1000), key=lambda item: item[2])
        thresholds[str(class_id)] = {"confidence_min": float(chosen[0]), "oof_pixels": chosen[1], "oof_precision": chosen[2]}

    accepted = np.zeros(len(gt), dtype=bool)
    for class_id, details in thresholds.items():
        accepted |= (pred == int(class_id)) & (confidence >= details["confidence_min"])
    correct_accepted = accepted & (pred == gt)
    entropy_max = float(np.quantile(entropy[correct_accepted], 0.90))
    disagreement_max = float(np.quantile(disagreement[correct_accepted], 0.90))
    accepted &= (entropy <= entropy_max) & (disagreement <= disagreement_max)
    overall_confusion = np.bincount(gt * 4 + pred, minlength=16).reshape(4, 4)
    overall_metrics = gv.metrics_from_conf(overall_confusion, [0, 1, 2, 3])
    result = {
        "manual_images": 953,
        "protocol": "three-seed animal-excluded LOAO ensemble; each expert pixel predicted only by teachers trained on the other 13 animals",
        "target_precision": args.target_precision,
        "class_confidence_thresholds": thresholds,
        "normalized_entropy_max": entropy_max,
        "ensemble_disagreement_max": disagreement_max,
        "accepted_pixel_fraction": float(accepted.mean()),
        "accepted_pixel_accuracy": float((pred[accepted] == gt[accepted]).mean()),
        "unfiltered_mdice": overall_metrics["mdice"],
        "unfiltered_pixel_accuracy": overall_metrics["pix_acc"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(animal_rows).to_csv(args.output / "oof_per_animal.csv", index=False)
    pd.DataFrame(curves).to_csv(args.output / "selective_precision_curves.csv", index=False)
    (args.output / "calibration_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
