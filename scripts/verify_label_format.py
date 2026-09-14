#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--release-root", type=Path, default=Path("."),
                    help="Root of the extracted Figshare release")
ROOT = parser.parse_args().release_root
labels = sorted((ROOT / "manual" / "labels_yolo").glob("*.txt"))
masks = sorted((ROOT / "manual" / "masks_indexed").glob("*.png"))

rows = 0
empty_files = 0
class_counts = Counter()
vertex_counts = Counter()
bad_rows = []
coord_min = float("inf")
coord_max = float("-inf")
closed_polygons = 0

for path in labels:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        empty_files += 1
        continue
    for line_no, line in enumerate(content.splitlines(), 1):
        fields = line.split()
        rows += 1
        try:
            cls = int(float(fields[0]))
            coords = [float(x) for x in fields[1:]]
        except Exception as exc:
            bad_rows.append([path.name, line_no, f"parse: {exc}"])
            continue
        class_counts[cls] += 1
        if len(coords) < 6 or len(coords) % 2:
            bad_rows.append([path.name, line_no, f"coordinate_count={len(coords)}"])
            continue
        vertex_counts[len(coords) // 2] += 1
        coord_min = min(coord_min, *coords)
        coord_max = max(coord_max, *coords)
        outside = [v for v in coords if v < 0 or v > 1]
        if outside:
            bad_rows.append([path.name, line_no, f"outside_0_1={len(outside)}"])
        if coords[0] == coords[-2] and coords[1] == coords[-1]:
            closed_polygons += 1

mask_values = Counter()
shape_counts = Counter()
for path in masks:
    arr = np.asarray(Image.open(path))
    shape_counts[str(tuple(arr.shape))] += 1
    values, counts = np.unique(arr, return_counts=True)
    mask_values.update({int(v): int(n) for v, n in zip(values, counts)})

result = {
    "label_files": len(labels),
    "empty_label_files": empty_files,
    "polygon_rows": rows,
    "class_counts": dict(sorted(class_counts.items())),
    "coordinate_min": coord_min,
    "coordinate_max": coord_max,
    "minimum_vertices": min(vertex_counts) if vertex_counts else None,
    "maximum_vertices": max(vertex_counts) if vertex_counts else None,
    "explicitly_closed_polygons": closed_polygons,
    "bad_rows": bad_rows[:100],
    "bad_row_count": len(bad_rows),
    "mask_files": len(masks),
    "mask_shapes": dict(shape_counts),
    "mask_values": dict(sorted(mask_values.items())),
}
print(json.dumps(result, indent=2))
