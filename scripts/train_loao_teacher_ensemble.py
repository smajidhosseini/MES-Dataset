#!/usr/bin/env python3
"""Train three fixed-epoch animal-excluded teachers for each MES animal."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import segmentation_models_pytorch as smp
    from torch.utils.data import DataLoader

    gv = load_module("grouped_validation", args.project_root / "grouped_validation.py")
    core = load_module("loao_pseudo_release_core", args.project_root / "publication_package/scripts/loao_pseudo_release_core.py")
    canonical = pd.read_csv(args.release / "metadata/canonical_manual_manifest.csv")
    mapping = dict(zip(canonical.video_id, canonical.anonymized_animal_id))
    animals = sorted(set(mapping.values()))
    jobs = core.teacher_jobs(animals)[args.shard_index::args.num_shards]
    frame_df = gv.build_manifest(args.release / "manual/images", args.release / "manual/masks_indexed")
    video_to_animal = {video.replace("V", "", 1) + "_mp4": animal for video, animal in mapping.items()}
    frame_df["animal"] = frame_df.video_id.map(video_to_animal)
    if frame_df.animal.isna().any():
        raise ValueError("At least one manual frame lacks an animal mapping")

    device = torch.device("cuda:0")
    for job in jobs:
        held_out, seed = job["animal"], job["seed"]
        fold_dir = args.output / "teachers" / held_out / f"seed_{seed}"
        checkpoint = fold_dir / "model.pt"
        metadata_path = fold_dir / "run_metadata.json"
        if checkpoint.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("checkpoint_sha256") == sha256(checkpoint):
                print(f"SKIP complete {held_out} seed={seed}", flush=True)
                continue

        train_df = frame_df[(frame_df.animal != held_out) & frame_df.has_scorable].reset_index(drop=True)
        test_df = frame_df[frame_df.animal == held_out].reset_index(drop=True)
        training_animals = sorted(set(train_df.animal))
        if held_out in training_animals or len(training_animals) != 13:
            raise AssertionError(f"Animal leakage for {held_out}: {training_animals}")
        gv.seed_all(seed)
        weights = gv.compute_class_weights(train_df).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=gv.IGNORE_INDEX)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        train_loader = DataLoader(
            gv.MESDataset(train_df, 512, 512, "train"), batch_size=args.batch_size,
            shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
        )
        test_loader = DataLoader(
            gv.MESDataset(test_df, 512, 512, "eval"), batch_size=args.batch_size,
            shuffle=False, num_workers=0, pin_memory=True,
        )
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50", encoder_weights="imagenet",
            in_channels=3, classes=4,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=3e-6,
        )
        print(f"TRAIN {held_out} seed={seed} n_train={len(train_df)} n_test={len(test_df)}", flush=True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for images, masks in train_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(images), masks)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.item()))
            scheduler.step(epoch)
            if epoch == 1 or epoch % 10 == 0:
                print(f"{held_out} seed={seed} epoch={epoch} loss={np.mean(losses):.5f}", flush=True)

        present = gv.present_classes_from_df(test_df)
        metrics, confusion = gv.evaluate(model, test_loader, device, present)
        fold_dir.mkdir(parents=True, exist_ok=True)
        tmp_checkpoint = checkpoint.with_suffix(".pt.tmp")
        torch.save({
            "model_state_dict": model.state_dict(),
            "architecture": "DeepLabV3+",
            "encoder": "resnet50",
            "seed": seed,
            "held_out_animal": held_out,
            "training_animals": training_animals,
            "epochs": args.epochs,
            "image_size": [512, 512],
        }, tmp_checkpoint)
        os.replace(tmp_checkpoint, checkpoint)
        pd.DataFrame(confusion).to_csv(fold_dir / "oof_confusion.csv", index=False)
        metadata = {
            "held_out_animal": held_out,
            "training_animals": training_animals,
            "seed": seed,
            "n_train": len(train_df),
            "n_oof_test": len(test_df),
            "epochs": args.epochs,
            "architecture": "DeepLabV3+",
            "encoder": "resnet50",
            "encoder_initialization": "ImageNet",
            "image_size": [512, 512],
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "selection_rule": "fixed 40 epochs; held-out animal evaluated once",
            "oof_mdice": metrics["mdice"],
            "oof_pixel_accuracy": metrics["pix_acc"],
            "checkpoint_sha256": sha256(checkpoint),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"DONE {held_out} seed={seed} mdice={metrics['mdice']:.4f}", flush=True)


if __name__ == "__main__":
    main()
