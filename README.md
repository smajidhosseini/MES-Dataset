# Spatial MES dataset: reproducible code

This repository contains the code used to generate, filter, audit, characterize, split, train, and evaluate the spatial Mayo Endoscopic Subscore (MES) dataset described in the accompanying manuscript.

The 953-image expert-annotated manual release is the reference dataset for validation and testing. The separate 8,772-image pseudo-label component is optional, training-only weak supervision. It is model-generated, is not independently expert validated, and must never be used as validation or test ground truth. MES 1 pseudo-pixels are excluded.

The data are deposited separately on Figshare. This repository does not contain images, annotation labels, or masks.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Contents

- `scripts/train_loao_teacher_ensemble.py`: train the 42 three-seed, animal-excluded teachers.
- `scripts/infer_loao_pseudo_ensemble.py`: run animal-excluded ensemble inference.
- `scripts/calibrate_loao_teacher_ensemble.py`: calibrate class and uncertainty thresholds from out-of-fold manual predictions.
- `scripts/select_loao_pseudo_release.py`: apply quality, leakage, uniqueness, temporal-separation, and diversity rules.
- `scripts/materialize_loao_pseudo_release.py`: write the pseudo-label release.
- `scripts/audit_loao_pseudo_release.py`: verify integrity and leakage constraints.
- `scripts/summarize_loao_pseudo_validation.py`: summarize calibration and annotation-free checks.
- `scripts/run_grouped_validation.py`: DeepLabV3+ baseline runner used for held-out-video jobs; its training and metric functions are also imported by the protocol wrapper used for four-fold and LOAO runs.
- `scripts/optimize_animal_grouped_folds.py`: generate balanced folds while keeping each animal indivisible.
- `scripts/signal_quality_analysis.py`: reproduce manual-image signal-quality summaries.
- `scripts/verify_label_format.py`: validate YOLO polygons and indexed-mask values.
- `configs/pseudo_selection_config.json`: finalized pseudo-label selection criteria.

## Dataset layout

After extracting the Figshare deposit:

```text
MES_dataset_release_v1/
  manual/{images,labels_yolo,masks_indexed,metadata}/
  pseudo_labels/{images,masks_indexed,metadata}/
```

Run basic manual-release checks with:

```bash
python scripts/verify_label_format.py --release-root /path/to/MES_dataset_release_v1
python scripts/signal_quality_analysis.py --root /path/to/MES_dataset_release_v1
```

Pseudo-label command-line arguments and defaults are documented with `--help`. The finalized thresholds are in `configs/pseudo_selection_config.json`. During any cross-validation experiment, pseudo-labeled frames from the held-out video or animal must be excluded, and all reported evaluation targets must come from the expert-annotated manual component.
