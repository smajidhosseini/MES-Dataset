import os
import shutil
import random
import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import segmentation_models_pytorch as smp


NUM_CLASSES   = 4
IGNORE_INDEX  = 255
CLASS_NAMES   = ["MES 0", "MES 1", "MES 2", "MES 3"]
MIN_TEST_FRAMES = 5
MASK_ENCODING = "release_indexed"


def decode_mask(mask):
    """Map stored mask values to training IDs 0--3 plus IGNORE_INDEX."""
    mask = mask.astype(np.uint8, copy=True)
    if MASK_ENCODING == "release_indexed":
        # Release: 0=background/unannotated, 1--4=MES 0--3, 255=ignore.
        decoded = np.full(mask.shape, IGNORE_INDEX, dtype=np.uint8)
        for stored_value in range(1, 5):
            decoded[mask == stored_value] = stored_value - 1
        return decoded
    if MASK_ENCODING == "legacy_zero_based":
        # Historical experiment masks: 0--3=MES 0--3, 255=ignore.
        return mask
    raise ValueError(f"Unknown mask encoding: {MASK_ENCODING}")


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def imread_color(path):
    arr = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None


def imread_gray(path):
    arr = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE) if arr.size else None


def extract_video_id(stem):
    return stem.split("_mp4")[0] + "_mp4"


def preprocess(img, h, w):
    img  = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb  = img[:, :, ::-1].astype(np.float32) / 255.0
    x    = np.transpose(rgb, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (x - mean[:, None, None]) / std[:, None, None]


def augment(img, mask, ignore):
    if random.random() < 0.5:
        img    = cv2.flip(img,    1)
        mask   = cv2.flip(mask,   1)
        ignore = cv2.flip(ignore, 1)
    angle = random.uniform(-8, 8)
    scale = random.uniform(0.95, 1.05)
    m     = cv2.getRotationMatrix2D(
        (img.shape[1] / 2.0, img.shape[0] / 2.0), angle, scale)
    img    = cv2.warpAffine(img,    m, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR,  borderMode=cv2.BORDER_REFLECT_101)
    mask   = cv2.warpAffine(mask,   m, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=IGNORE_INDEX)
    ignore = cv2.warpAffine(ignore, m, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=1)
    if random.random() < 0.3:
        img = np.clip(
            img.astype(np.float32) * (1 + random.uniform(-0.15, 0.15))
            + random.uniform(-0.08, 0.08) * 255, 0, 255).astype(np.uint8)
    if random.random() < 0.2:
        img = cv2.GaussianBlur(img, (random.choice([3, 5]),) * 2, 0)
    return img, mask, ignore


class MESDataset(Dataset):
    def __init__(self, df, img_h, img_w, mode="train"):
        self.df   = df.reset_index(drop=True)
        self.h    = img_h
        self.w    = img_w
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r    = self.df.iloc[idx]
        img  = imread_color(r.img_path)
        mask = imread_gray(r.mask_path)
        if img is None or mask is None:
            raise RuntimeError(f"Failed reading {r.stem}")
        mask = decode_mask(mask)
        img  = cv2.resize(img,  (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        ignore = (mask == IGNORE_INDEX).astype(np.uint8)
        if self.mode == "train":
            img, mask, ignore = augment(img, mask, ignore)
        mask = mask.astype(np.uint8)
        mask[ignore > 0] = IGNORE_INDEX
        x = preprocess(img, self.h, self.w)
        return (torch.from_numpy(x.astype(np.float32)),
                torch.from_numpy(mask.astype(np.int64)))


def build_manifest(img_dir, mask_dir):
    rows = []
    for img_path in sorted(Path(img_dir).glob("*.jpg")):
        mask_path = Path(mask_dir) / (img_path.stem + ".png")
        if not mask_path.exists():
            continue
        mask = decode_mask(np.array(Image.open(mask_path)))
        if (mask != IGNORE_INDEX).sum() == 0:
            continue
        rows.append({
            "stem":      img_path.stem,
            "video_id":  extract_video_id(img_path.stem),
            "img_path":  str(img_path),
            "mask_path": str(mask_path),
        })
        for c in range(NUM_CLASSES):
            rows[-1][f"px_{c}"] = int((mask == c).sum())
    return pd.DataFrame(rows)


def compute_class_weights(df):
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for _, r in df.iterrows():
        for c in range(NUM_CLASSES):
            counts[c] += r[f"px_{c}"]
    counts  = np.maximum(counts, 1.0)
    weights = 1.0 / np.log(1.02 + counts / counts.sum())
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def present_classes_from_df(df):
    return [c for c in range(NUM_CLASSES) if df[f"px_{c}"].sum() > 0]


def metrics_from_conf(conf, present_classes=None):
    conf = conf.astype(np.float64)
    tp   = np.diag(conf)
    fp   = conf.sum(0) - tp
    fn   = conf.sum(1) - tp
    iou  = tp / np.maximum(1.0, tp + fp + fn)
    dice = (2 * tp) / np.maximum(1.0, 2 * tp + fp + fn)
    prec = tp / np.maximum(1.0, tp + fp)
    rec  = tp / np.maximum(1.0, tp + fn)
    acc  = tp.sum() / np.maximum(1.0, conf.sum())

    if present_classes is None:
        present_classes = [c for c in range(NUM_CLASSES) if conf[c, :].sum() > 0]
    present_classes = list(present_classes)
    if not present_classes:
        present_classes = list(range(NUM_CLASSES))

    return {
        "pix_acc": float(acc),
        "miou":    float(iou[present_classes].mean()),
        "mdice":   float(dice[present_classes].mean()),
        "iou_pc":  iou,
        "dice_pc": dice,
        "prec_pc": prec,
        "rec_pc":  rec,
        "present_classes": present_classes,
    }


def metric_or_na(metrics, metric_name, class_idx):
    if class_idx not in metrics["present_classes"]:
        return np.nan
    return round(float(metrics[metric_name][class_idx]), 4)


def fmt_metric(value):
    if pd.isna(value):
        return "NA"
    return f"{value:.4f}"


@torch.no_grad()
def evaluate(model, loader, device, present_classes=None):
    model.eval()
    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            preds = model(xb).argmax(dim=1)
        valid = yb != IGNORE_INDEX
        yv    = yb[valid]
        pv    = preds[valid]
        if yv.numel() > 0:
            conf += torch.bincount(
                yv * NUM_CLASSES + pv,
                minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
    return metrics_from_conf(conf.cpu().numpy(), present_classes), conf.cpu().numpy()


def train_fold(train_df, test_df, args, device, fold_name, present_classes):
    seed_all(args.seed)

    weights   = compute_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)
    scaler    = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    train_ds = MESDataset(train_df, args.img_h, args.img_w, "train")
    test_ds  = MESDataset(test_df,  args.img_h, args.img_w, "eval")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True,
                              persistent_workers=args.num_workers > 0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=False,
                              persistent_workers=args.num_workers > 0)

    model     = smp.DeepLabV3Plus(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.t0, T_mult=2, eta_min=args.lr * 0.01)

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        loss_sum  = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum  += float(loss.item())
            n_batches += 1

        scheduler.step(epoch)
        # Do not inspect the held-out video during training.  In LOVO it is the
        # test set, so checkpoint selection or early stopping on its score leaks
        # test information.  The epoch count is fixed before the experiment.

    best_metrics, best_conf = evaluate(
        model, test_loader, device, present_classes)

    fold_dir = args.output_dir / f"fold_{fold_name.replace('_mp4', '')}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        best_conf,
        index=[f"gt_{c}" for c in range(NUM_CLASSES)],
        columns=[f"pred_{c}" for c in range(NUM_CLASSES)],
    ).to_csv(fold_dir / "best_confusion.csv")
    with open(fold_dir / "run_metadata.json", "w") as f:
        json.dump({
            "video_id": fold_name,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "present_classes": present_classes,
            "seed": args.seed,
            "encoder": args.encoder,
            "image_size": [args.img_h, args.img_w],
            "training_epochs": args.num_epochs,
            "selection_rule": "fixed epochs; held-out video evaluated once",
            "best_mdice": best_metrics["mdice"],
        }, f, indent=2)
    return best_metrics


def run_lovo(args, device):
    print(f"Loading manifest from {args.img_dir} / {args.mask_dir}")
    df = build_manifest(args.img_dir, args.mask_dir)
    print(f"Total frames: {len(df)} across {df['video_id'].nunique()} videos")

    video_ids = sorted(df["video_id"].unique())
    results   = []
    skipped   = []

    for fold_i, held_out in enumerate(video_ids):
        test_df  = df[df["video_id"] == held_out].reset_index(drop=True)
        train_df = df[df["video_id"] != held_out].reset_index(drop=True)

        n_test      = len(test_df)
        active_cls  = present_classes_from_df(test_df)
        has_mes3    = 3 in active_cls

        if n_test < MIN_TEST_FRAMES:
            print(f"\nFold {fold_i+1:02d}/{len(video_ids)} "
                  f"[{held_out}] SKIPPED: only {n_test} test frames")
            skipped.append(held_out)
            continue

        print(f"\nFold {fold_i+1:02d}/{len(video_ids)} "
              f"[{held_out}] train={len(train_df)} test={n_test} "
              f"classes={[CLASS_NAMES[c] for c in active_cls]}")

        t0      = time.time()
        metrics = train_fold(train_df, test_df, args, device, held_out, active_cls)
        elapsed = time.time() - t0

        row = {
            "fold":       fold_i + 1,
            "video_id":   held_out,
            "n_train":    len(train_df),
            "n_test":     n_test,
            "has_mes3":   has_mes3,
            "present_classes": ",".join(CLASS_NAMES[c] for c in active_cls),
            "elapsed_s":  round(elapsed, 1),
            "pix_acc":    round(metrics["pix_acc"], 4),
            "miou":       round(metrics["miou"],    4),
            "mdice":      round(metrics["mdice"],   4),
        }
        for c in range(NUM_CLASSES):
            row[f"iou_{c}"]  = metric_or_na(metrics, "iou_pc", c)
            row[f"dice_{c}"] = metric_or_na(metrics, "dice_pc", c)
            row[f"rec_{c}"]  = metric_or_na(metrics, "rec_pc", c)
            row[f"prec_{c}"] = metric_or_na(metrics, "prec_pc", c)

        results.append(row)
        df_results = pd.DataFrame(results)
        df_results.to_csv(args.output_csv, index=False)

        print(f"  mIoU={row['miou']:.4f} mDice={row['mdice']:.4f} "
              f"IoU3={fmt_metric(row['iou_3'])} Rec3={fmt_metric(row['rec_3'])} "
              f"({elapsed:.0f}s)")

    return pd.DataFrame(results), skipped


def summarize(df_results):
    print(f"\n{'='*65}")
    print("LOVO SUMMARY")
    print(f"{'='*65}")
    print(f"Valid folds: {len(df_results)}")

    all_folds = df_results
    mes3_folds = df_results[df_results["has_mes3"]]

    print(f"\nAll {len(all_folds)} valid folds:")
    print("  mIoU/mDice are macro-averaged over classes present in each held-out video.")
    for metric in ["miou", "mdice", "pix_acc"]:
        vals = all_folds[metric]
        print(f"  {metric:<12} mean={vals.mean():.4f}  "
              f"std={vals.std():.4f}  "
              f"min={vals.min():.4f}  max={vals.max():.4f}")

    print(f"\nPer-class IoU (all valid folds):")
    for c in range(NUM_CLASSES):
        vals = all_folds[f"iou_{c}"].dropna()
        if len(vals) == 0:
            print(f"  {CLASS_NAMES[c]:<10} NA")
        else:
            print(f"  {CLASS_NAMES[c]:<10} mean={vals.mean():.4f}  std={vals.std():.4f}")

    print(f"\nPer-class Recall (all valid folds):")
    for c in range(NUM_CLASSES):
        vals = all_folds[f"rec_{c}"].dropna()
        if len(vals) == 0:
            print(f"  {CLASS_NAMES[c]:<10} NA")
        else:
            print(f"  {CLASS_NAMES[c]:<10} mean={vals.mean():.4f}  std={vals.std():.4f}")

    if len(mes3_folds) > 0:
        print(f"\nMES 3 metrics ({len(mes3_folds)} folds containing MES 3):")
        for metric in ["iou_3", "dice_3", "rec_3", "prec_3"]:
            vals = mes3_folds[metric]
            print(f"  {metric:<12} mean={vals.mean():.4f}  std={vals.std():.4f}")

    print(f"\nPer-fold results:")
    print(f"{'Video':<12} {'n_test':>7} {'mIoU':>7} {'mDice':>7} "
          f"{'IoU0':>7} {'IoU1':>7} {'IoU2':>7} {'IoU3':>7} {'Rec3':>7}")
    print("-" * 72)
    for _, r in df_results.iterrows():
        print(f"{r['video_id']:<12} {r['n_test']:>7} {r['miou']:>7.4f} "
              f"{r['mdice']:>7.4f} {fmt_metric(r['iou_0']):>7} "
              f"{fmt_metric(r['iou_1']):>7} {fmt_metric(r['iou_2']):>7} "
              f"{fmt_metric(r['iou_3']):>7} {fmt_metric(r['rec_3']):>7}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base",        type=Path,
                   default=Path("dataset"))
    p.add_argument("--img_dir",     type=Path, default=None)
    p.add_argument("--mask_dir",    type=Path, default=None)
    p.add_argument("--output_csv",  type=Path, default=None)
    p.add_argument("--output_dir",  type=Path, default=None)
    p.add_argument("--encoder",     type=str,  default="resnet50")
    p.add_argument("--img_h",       type=int,  default=512)
    p.add_argument("--img_w",       type=int,  default=512)
    p.add_argument("--batch_size",  type=int,  default=16)
    p.add_argument("--num_epochs",  type=int,  default=40)
    p.add_argument("--t0",          type=int,  default=20)
    p.add_argument("--lr",          type=float,default=1e-4)
    p.add_argument("--num_workers", type=int,  default=4)
    p.add_argument("--gpu_id",      type=int,  default=0)
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--mask_encoding", choices=["release_indexed", "legacy_zero_based"],
                   default="release_indexed",
                   help="Use release_indexed for public masks (0 background, 1--4 MES); "
                        "legacy_zero_based reproduces historical 0--3 masks.")
    p.add_argument("--fold_start",  type=int,  default=1,
                   help="Start from this fold index (1-based, for resuming)")
    p.add_argument("--fold_end",    type=int,  default=999,
                   help="Stop after this fold index (for splitting across GPUs)")
    args = p.parse_args()

    if args.img_dir is None:
        args.img_dir = args.base / "train" / "images"
    if args.mask_dir is None:
        args.mask_dir = args.base / "train" / "masks"
    if args.output_csv is None:
        args.output_csv = args.base / f"lovo_results_f{args.fold_start}_{args.fold_end}.csv"
    if args.output_dir is None:
        args.output_dir = args.base / "results_lovo_pixel_pooled"

    return args


if __name__ == "__main__":
    args   = parse_args()
    MASK_ENCODING = args.mask_encoding
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    print(f"LOVO evaluation")
    print(f"Device  : {device}")
    print(f"Encoder : {args.encoder}")
    print(f"Images  : {args.img_dir}")
    print(f"Masks   : {args.mask_dir}")
    print(f"Folds   : {args.fold_start} to {args.fold_end}")
    print(f"Output  : {args.output_csv}")

    df = build_manifest(args.img_dir, args.mask_dir)
    video_ids = sorted(df["video_id"].unique())

    fold_start = args.fold_start - 1
    fold_end   = min(args.fold_end, len(video_ids))
    video_ids  = video_ids[fold_start:fold_end]

    all_results = []
    skipped     = []

    for fold_i, held_out in enumerate(video_ids):
        global_fold = fold_start + fold_i + 1
        test_df     = df[df["video_id"] == held_out].reset_index(drop=True)
        train_df    = df[df["video_id"] != held_out].reset_index(drop=True)

        n_test     = len(test_df)
        active_cls = present_classes_from_df(test_df)
        has_mes3   = 3 in active_cls

        if n_test < MIN_TEST_FRAMES:
            print(f"\nFold {global_fold:02d} [{held_out}] SKIPPED "
                  f"(only {n_test} frames)")
            skipped.append(held_out)
            continue

        print(f"\nFold {global_fold:02d} [{held_out}] "
              f"train={len(train_df)} test={n_test} "
              f"active={[CLASS_NAMES[c] for c in active_cls]}")

        t0      = time.time()
        metrics = train_fold(train_df, test_df, args, device, held_out, active_cls)
        elapsed = time.time() - t0

        row = {
            "fold":      global_fold,
            "video_id":  held_out,
            "n_train":   len(train_df),
            "n_test":    n_test,
            "has_mes3":  has_mes3,
            "present_classes": ",".join(CLASS_NAMES[c] for c in active_cls),
            "elapsed_s": round(elapsed, 1),
            "pix_acc":   round(metrics["pix_acc"], 4),
            "miou":      round(metrics["miou"],    4),
            "mdice":     round(metrics["mdice"],   4),
        }
        for c in range(NUM_CLASSES):
            row[f"iou_{c}"]  = metric_or_na(metrics, "iou_pc", c)
            row[f"dice_{c}"] = metric_or_na(metrics, "dice_pc", c)
            row[f"rec_{c}"]  = metric_or_na(metrics, "rec_pc", c)
            row[f"prec_{c}"] = metric_or_na(metrics, "prec_pc", c)

        all_results.append(row)
        df_out = pd.DataFrame(all_results)
        df_out.to_csv(args.output_csv, index=False)

        print(f"  mIoU={row['miou']:.4f}  mDice={row['mdice']:.4f}  "
              f"IoU3={fmt_metric(row['iou_3'])}  Rec3={fmt_metric(row['rec_3'])}  "
              f"({elapsed:.0f}s)")

    df_results = pd.DataFrame(all_results)
    if len(df_results) > 0:
        summarize(df_results)
        print(f"\nResults saved to {args.output_csv}")
    if skipped:
        print(f"Skipped videos (< {MIN_TEST_FRAMES} frames): {skipped}")
