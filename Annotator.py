import re
import copy
from pathlib import Path

import cv2
import numpy as np


IMAGE_DIR = Path(r"C:\Users\majid\Documents\Projects\MES\Dataset\train\images")
LABEL_DIR = Path(r"C:\Users\majid\Documents\Projects\MES\Dataset\train\labels")
CORRECTED_LABEL_DIR = Path(r"C:\Users\majid\Documents\Projects\MES\Dataset\train\corrected_labels")

WINDOW_NAME = "MES Annotator"

CLASSES = [0, 1, 2, 3]
PANEL_W = 820
PANEL_H = 820
TOP_H = 90
MARGIN = 20

BG = (32, 32, 32)
BTN_BG = (60, 60, 60)
BTN_FG = (235, 235, 235)
TEXT = (230, 230, 230)
ACTIVE = (0, 255, 255)
DELETE_COLOR = (60, 60, 180)

CLASS_COLORS = {
    0: (80, 180, 255),
    1: (80, 255, 120),
    2: (255, 180, 80),
    3: (255, 80, 160),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FILE_RE = re.compile(r"^(?P<video>.+?)_mp4-(?P<frame>\d+)_jpg", re.IGNORECASE)


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def imread_any(path):
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def parse_info(img_path):
    name = Path(img_path).name
    m = FILE_RE.search(name)
    if m:
        return {
            "video": m.group("video"),
            "frame": int(m.group("frame")),
            "name": name,
        }
    stem = Path(img_path).stem
    return {
        "video": stem.split("_", 1)[0],
        "frame": -1,
        "name": name,
    }


def label_path_for_image(img_path, base_dir):
    return Path(base_dir) / f"{Path(img_path).stem}.txt"


def has_label_in_dir(img_path, base_dir):
    return label_path_for_image(img_path, base_dir).exists()


def has_any_label(img_path):
    return has_label_in_dir(img_path, CORRECTED_LABEL_DIR) or has_label_in_dir(img_path, LABEL_DIR)


def get_best_label_path(img_path):
    p1 = label_path_for_image(img_path, CORRECTED_LABEL_DIR)
    p2 = label_path_for_image(img_path, LABEL_DIR)
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    return None


def load_yolo_polygons(txt_path, w, h):
    polys = []
    if txt_path is None or not Path(txt_path).exists():
        return polys

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            try:
                cls = int(float(parts[0]))
            except Exception:
                continue

            nums = parts[1:]
            if len(nums) < 6 or len(nums) % 2 != 0:
                continue

            vals = []
            ok = True
            for x in nums:
                try:
                    vals.append(float(x))
                except Exception:
                    ok = False
                    break
            if not ok:
                continue

            pts = []
            for i in range(0, len(vals), 2):
                x = int(round(vals[i] * w))
                y = int(round(vals[i + 1] * h))
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                pts.append((x, y))

            if len(pts) >= 3:
                polys.append({"cls": cls, "pts": pts})

    return polys


def save_yolo_polygons(polys, w, h, out_path):
    ensure_dir(Path(out_path).parent)
    lines = []

    for poly in polys:
        pts = poly["pts"]
        cls = int(poly["cls"])
        if len(pts) < 3:
            continue

        vals = []
        for x, y in pts:
            vals.append(f"{max(0.0, min(1.0, x / max(1, w))):.6f}")
            vals.append(f"{max(0.0, min(1.0, y / max(1, h))):.6f}")

        lines.append(f"{cls} " + " ".join(vals))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def fit_image(img, panel_w, panel_h):
    h, w = img.shape[:2]
    scale = min(panel_w / w, panel_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ox = (panel_w - nw) // 2
    oy = (panel_h - nh) // 2
    return resized, scale, ox, oy


def draw_arrow_button(canvas, rect, direction, label):
    x1, y1, x2, y2 = rect
    cv2.rectangle(canvas, (x1, y1), (x2, y2), BTN_BG, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (120, 120, 120), 1)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    if direction == "left":
        pts = np.array([[cx + 10, cy - 12], [cx - 10, cy], [cx + 10, cy + 12]], np.int32)
    else:
        pts = np.array([[cx - 10, cy - 12], [cx + 10, cy], [cx - 10, cy + 12]], np.int32)

    cv2.fillConvexPoly(canvas, pts, BTN_FG)
    cv2.putText(canvas, label, (x1 + 8, y2 + 18), FONT, 0.45, TEXT, 1, cv2.LINE_AA)


def draw_copy_button(canvas, rect, label="Copy"):
    x1, y1, x2, y2 = rect
    cv2.rectangle(canvas, (x1, y1), (x2, y2), BTN_BG, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (120, 120, 120), 1)
    cv2.putText(canvas, ">>", (x1 + 10, y1 + 23), FONT, 0.7, BTN_FG, 2, cv2.LINE_AA)
    cv2.putText(canvas, label, (x1 + 4, y2 + 18), FONT, 0.45, TEXT, 1, cv2.LINE_AA)


def draw_delete_button(canvas, rect, label="Delete"):
    x1, y1, x2, y2 = rect
    cv2.rectangle(canvas, (x1, y1), (x2, y2), DELETE_COLOR, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), 1)
    cv2.putText(canvas, "X", (x1 + 18, y1 + 24), FONT, 0.7, BTN_FG, 2, cv2.LINE_AA)
    cv2.putText(canvas, label, (x1 + 2, y2 + 18), FONT, 0.45, TEXT, 1, cv2.LINE_AA)


def point_in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def point_to_segment_distance(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby

    if denom == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5

    t = (apx * abx + apy * aby) / denom
    t = max(0.0, min(1.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


class Annotator:
    def __init__(self):
        ensure_dir(CORRECTED_LABEL_DIR)

        self.records = self.build_records()
        if not self.records:
            raise RuntimeError("No images found.")

        self.target_indices = list(range(len(self.records)))
        self.left_indices = [i for i, r in enumerate(self.records) if has_any_label(r["img_path"])]

        self.target_pos = 0
        self.left_pos = 0

        self.left_img = None
        self.right_img = None
        self.left_polys = []
        self.right_polys = []

        self.active_poly_idx = -1
        self.active_class = 0
        self.message = ""
        self.layout = {}

        self.load_left_only()
        self.load_right_only()

    def build_records(self):
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"):
            imgs.extend(IMAGE_DIR.glob(ext))
            imgs.extend(IMAGE_DIR.glob(ext.upper()))

        rows = []
        for p in sorted(set(imgs)):
            info = parse_info(p)
            rows.append({
                "img_path": p,
                "video": info["video"],
                "frame": info["frame"],
                "name": info["name"],
            })

        rows.sort(key=lambda r: (r["video"], r["frame"], r["name"]))
        return rows

    def get_current_target_idx(self):
        if not self.target_indices:
            return None
        return self.target_indices[self.target_pos]

    def get_current_left_idx(self):
        if not self.left_indices:
            return None
        return self.left_indices[self.left_pos]

    def load_polys_for_record(self, rec_idx):
        rec = self.records[rec_idx]
        img = imread_any(rec["img_path"])
        if img is None:
            return [], None

        h, w = img.shape[:2]
        polys = load_yolo_polygons(get_best_label_path(rec["img_path"]), w, h)
        return polys, img

    def load_left_reference(self):
        ref_idx = self.get_current_left_idx()
        if ref_idx is None:
            if self.right_img is not None:
                self.left_img = np.zeros_like(self.right_img)
            else:
                self.left_img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            self.left_polys = []
            return

        self.left_polys, self.left_img = self.load_polys_for_record(ref_idx)

    def load_right_annotation(self):
        target_idx = self.get_current_target_idx()
        if target_idx is None:
            self.right_polys = []
            self.active_poly_idx = -1
            return

        rec = self.records[target_idx]
        corrected = label_path_for_image(rec["img_path"], CORRECTED_LABEL_DIR)
        original = label_path_for_image(rec["img_path"], LABEL_DIR)

        h, w = self.right_img.shape[:2]

        if corrected.exists():
            self.right_polys = load_yolo_polygons(corrected, w, h)
            self.message = f"Loaded corrected label: {corrected.name}"
        elif original.exists():
            self.right_polys = load_yolo_polygons(original, w, h)
            self.message = f"Loaded original label: {original.name}"
        else:
            self.right_polys = []
            self.message = "Target has no annotation yet."

        self.active_poly_idx = 0 if self.right_polys else -1
        if self.active_poly_idx >= 0:
            self.active_class = int(self.right_polys[self.active_poly_idx]["cls"])
        else:
            self.active_class = CLASSES[0]

    def load_left_only(self):
        self.load_left_reference()
        if self.right_img is None:
            self.right_img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        self.render()

    def load_right_only(self):
        target_idx = self.get_current_target_idx()
        if target_idx is None:
            self.right_img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            self.right_polys = []
            self.active_poly_idx = -1
            self.render()
            return

        rec = self.records[target_idx]
        self.right_img = imread_any(rec["img_path"])
        if self.right_img is None:
            raise RuntimeError(f"Could not read image: {rec['img_path']}")

        self.load_right_annotation()
        if self.left_img is None:
            self.load_left_reference()
        self.render()

    def prev_target(self):
        if self.target_indices and self.target_pos > 0:
            self.target_pos -= 1
            self.load_right_only()

    def next_target(self):
        if self.target_indices and self.target_pos < len(self.target_indices) - 1:
            self.target_pos += 1
            self.load_right_only()

    def prev_ref(self):
        if self.left_indices and self.left_pos > 0:
            self.left_pos -= 1
            self.load_left_only()
            ref_idx = self.get_current_left_idx()
            self.message = f"Left reference: {self.records[ref_idx]['name']}" if ref_idx is not None else "No reference"
            self.render()

    def next_ref(self):
        if self.left_indices and self.left_pos < len(self.left_indices) - 1:
            self.left_pos += 1
            self.load_left_only()
            ref_idx = self.get_current_left_idx()
            self.message = f"Left reference: {self.records[ref_idx]['name']}" if ref_idx is not None else "No reference"
            self.render()

    def cycle_active_polygon(self):
        if not self.right_polys:
            self.active_poly_idx = -1
            self.message = "No polygon on right image."
            self.render()
            return

        if self.active_poly_idx < 0:
            self.active_poly_idx = 0
        else:
            self.active_poly_idx = (self.active_poly_idx + 1) % len(self.right_polys)

        self.active_class = int(self.right_polys[self.active_poly_idx]["cls"])
        self.message = f"Active polygon switched to {self.active_poly_idx}, class {self.active_class}"
        self.render()

    def add_new_polygon(self):
        self.right_polys.append({"cls": self.active_class, "pts": []})
        self.active_poly_idx = len(self.right_polys) - 1
        self.message = f"New polygon created: {self.active_poly_idx}, class {self.active_class}"
        self.render()

    def delete_active_polygon(self):
        if self.active_poly_idx < 0 or self.active_poly_idx >= len(self.right_polys):
            self.message = "No active polygon to delete."
            self.render()
            return

        self.right_polys.pop(self.active_poly_idx)

        if not self.right_polys:
            self.active_poly_idx = -1
            self.message = "Active polygon deleted. No polygons left."
        else:
            self.active_poly_idx = min(self.active_poly_idx, len(self.right_polys) - 1)
            self.active_class = int(self.right_polys[self.active_poly_idx]["cls"])
            self.message = f"Active polygon deleted. Now active: {self.active_poly_idx}"

        self.render()

    def copy_left_to_right(self):
        self.right_polys = copy.deepcopy(self.left_polys)
        self.active_poly_idx = 0 if self.right_polys else -1
        if self.active_poly_idx >= 0:
            self.active_class = int(self.right_polys[self.active_poly_idx]["cls"])
        self.message = "Copied annotation from left to right."
        self.render()

    def save_current(self):
        target_idx = self.get_current_target_idx()
        if target_idx is None:
            self.message = "No target image."
            self.render()
            return

        rec = self.records[target_idx]
        h, w = self.right_img.shape[:2]
        out_path = label_path_for_image(rec["img_path"], CORRECTED_LABEL_DIR)
        save_yolo_polygons(self.right_polys, w, h, out_path)
        self.message = f"Saved: {out_path.name}"
        self.render()

    def delete_current_right_image_and_labels(self):
        target_idx = self.get_current_target_idx()
        if target_idx is None:
            self.message = "No target image to delete."
            self.render()
            return

        rec = self.records[target_idx]

        img_path = Path(rec["img_path"])
        orig_label = label_path_for_image(rec["img_path"], LABEL_DIR)
        corr_label = label_path_for_image(rec["img_path"], CORRECTED_LABEL_DIR)

        deleted = []

        if img_path.exists():
            img_path.unlink()
            deleted.append(img_path.name)

        if orig_label.exists():
            orig_label.unlink()
            deleted.append(orig_label.name)

        if corr_label.exists():
            corr_label.unlink()
            deleted.append(corr_label.name)

        deleted_target_pos = self.target_pos

        self.records.pop(target_idx)

        new_target_indices = []
        for idx in self.target_indices:
            if idx == target_idx:
                continue
            if idx > target_idx:
                new_target_indices.append(idx - 1)
            else:
                new_target_indices.append(idx)
        self.target_indices = new_target_indices

        new_left_indices = []
        for idx in self.left_indices:
            if idx == target_idx:
                continue
            if idx > target_idx:
                new_left_indices.append(idx - 1)
            else:
                new_left_indices.append(idx)
        self.left_indices = new_left_indices

        if self.left_indices:
            self.left_pos = min(self.left_pos, len(self.left_indices) - 1)
        else:
            self.left_pos = 0

        if not self.target_indices:
            self.right_polys = []
            self.left_polys = []
            self.right_img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            self.active_poly_idx = -1
            self.message = f"Deleted: {', '.join(deleted)}. No target images left."
            self.load_left_reference()
            self.render()
            return

        self.target_pos = min(deleted_target_pos, len(self.target_indices) - 1)
        self.right_polys = []
        self.active_poly_idx = -1
        self.message = f"Deleted: {', '.join(deleted)}"

        self.load_left_reference()
        self.load_right_only()

    def ensure_active_poly(self):
        if self.active_poly_idx < 0 or self.active_poly_idx >= len(self.right_polys):
            self.right_polys.append({"cls": self.active_class, "pts": []})
            self.active_poly_idx = len(self.right_polys) - 1

    def insert_point_into_active_polygon(self, x, y):
        self.ensure_active_poly()
        poly = self.right_polys[self.active_poly_idx]
        poly["cls"] = self.active_class
        pts = poly["pts"]

        if len(pts) < 2:
            pts.append((int(x), int(y)))
            self.render()
            return

        click = (int(x), int(y))
        best_insert_after = 0
        best_dist = None
        n = len(pts)

        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            d = point_to_segment_distance(click, a, b)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_insert_after = i

        pts.insert(best_insert_after + 1, click)
        self.render()

    def remove_closest_point(self, x, y):
        if self.active_poly_idx < 0 or self.active_poly_idx >= len(self.right_polys):
            return

        pts = self.right_polys[self.active_poly_idx]["pts"]
        if not pts:
            return

        dists = [((px - x) ** 2 + (py - y) ** 2, i) for i, (px, py) in enumerate(pts)]
        _, idx = min(dists, key=lambda t: t[0])
        pts.pop(idx)
        self.render()

    def draw_polys_on_panel(self, base_img, polys, panel_rect, active_idx=None, show_class_text=False):
        x1, y1, x2, y2 = panel_rect
        panel_w = x2 - x1
        panel_h = y2 - y1

        view = np.full((panel_h, panel_w, 3), BG, dtype=np.uint8)
        resized, scale, ox, oy = fit_image(base_img, panel_w, panel_h)
        view[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized

        for i, poly in enumerate(polys):
            color = CLASS_COLORS.get(int(poly["cls"]), (255, 255, 255))
            pts = poly["pts"]
            if not pts:
                continue

            pts2 = []
            for px, py in pts:
                rx = int(round(px * scale)) + ox
                ry = int(round(py * scale)) + oy
                pts2.append((rx, ry))

            if len(pts2) >= 2:
                thickness = 4 if i == active_idx else 2
                line_color = ACTIVE if i == active_idx else color
                cv2.polylines(view, [np.array(pts2, np.int32)], True, line_color, thickness, cv2.LINE_AA)

            for j, (rx, ry) in enumerate(pts2):
                point_color = ACTIVE if i == active_idx else color
                radius = 5 if i == active_idx else 3
                cv2.circle(view, (rx, ry), radius, point_color, -1, cv2.LINE_AA)
                if i == active_idx:
                    cv2.putText(view, str(j), (rx + 4, ry - 4), FONT, 0.35, ACTIVE, 1, cv2.LINE_AA)

            tx, ty = pts2[0]
            if show_class_text:
                label = f"class {int(poly['cls'])}"
                cv2.putText(view, label, (tx + 6, max(18, ty - 8)), FONT, 0.55, color, 2, cv2.LINE_AA)

            if i == active_idx:
                active_label = f"active #{i} c{int(poly['cls'])}"
                cv2.putText(view, active_label, (tx + 6, min(panel_h - 10, ty + 18)), FONT, 0.5, ACTIVE, 2, cv2.LINE_AA)

        return view, scale, ox, oy

    def render(self):
        canvas_w = PANEL_W * 2 + MARGIN * 3
        canvas_h = PANEL_H + TOP_H + MARGIN * 2 + 145
        canvas = np.full((canvas_h, canvas_w, 3), BG, dtype=np.uint8)

        left_panel = (MARGIN, TOP_H + MARGIN, MARGIN + PANEL_W, TOP_H + MARGIN + PANEL_H)
        right_panel = (MARGIN * 2 + PANEL_W, TOP_H + MARGIN, MARGIN * 2 + PANEL_W * 2, TOP_H + MARGIN + PANEL_H)

        lb = (MARGIN + 20, 20, MARGIN + 70, 50)
        ln = (MARGIN + 90, 20, MARGIN + 140, 50)
        cp = (MARGIN + 160, 20, MARGIN + 230, 50)
        db = (MARGIN + 250, 20, MARGIN + 330, 50)
        rb = (MARGIN * 2 + PANEL_W + 20, 20, MARGIN * 2 + PANEL_W + 70, 50)
        rn = (MARGIN * 2 + PANEL_W + 90, 20, MARGIN * 2 + PANEL_W + 140, 50)

        draw_arrow_button(canvas, lb, "left", "Ref")
        draw_arrow_button(canvas, ln, "right", "Ref")
        draw_copy_button(canvas, cp, "Copy")
        draw_delete_button(canvas, db, "Delete")
        draw_arrow_button(canvas, rb, "left", "Target")
        draw_arrow_button(canvas, rn, "right", "Target")

        left_base = self.left_img if self.left_img is not None else np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        right_base = self.right_img if self.right_img is not None else np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

        left_view, ls, lox, loy = self.draw_polys_on_panel(
            left_base, self.left_polys, left_panel, None, show_class_text=True
        )
        right_view, rs, rox, roy = self.draw_polys_on_panel(
            right_base, self.right_polys, right_panel, self.active_poly_idx, show_class_text=False
        )

        canvas[left_panel[1]:left_panel[3], left_panel[0]:left_panel[2]] = left_view
        canvas[right_panel[1]:right_panel[3], right_panel[0]:right_panel[2]] = right_view

        target_idx = self.get_current_target_idx()
        ref_idx = self.get_current_left_idx()

        left_title = f"Reference: {self.records[ref_idx]['name']}" if ref_idx is not None else "Reference: none"
        right_title = f"Target {self.target_pos + 1}/{len(self.target_indices)}: {self.records[target_idx]['name']}" if target_idx is not None else "Target: none"

        cv2.putText(canvas, left_title, (left_panel[0], left_panel[1] - 10), FONT, 0.6, TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, right_title, (right_panel[0], right_panel[1] - 10), FONT, 0.6, TEXT, 1, cv2.LINE_AA)

        info_y = TOP_H + PANEL_H + MARGIN + 35
        cv2.putText(
            canvas,
            f"Active class: {self.active_class} | Active polygon: {self.active_poly_idx} | Right polygons: {len(self.right_polys)}",
            (MARGIN, info_y),
            FONT,
            0.55,
            TEXT,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Left click adds point | Right click deletes nearest point | 0/1/2/3 set class | Tab switch polygon",
            (MARGIN, info_y + 28),
            FONT,
            0.48,
            TEXT,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "N new polygon | Backspace delete active polygon | S save | C copy | Delete removes image+labels | Q quit",
            (MARGIN, info_y + 56),
            FONT,
            0.48,
            TEXT,
            1,
            cv2.LINE_AA,
        )

        if self.message:
            cv2.putText(canvas, self.message, (MARGIN, info_y + 88), FONT, 0.52, ACTIVE, 1, cv2.LINE_AA)

        self.layout = {
            "left_panel": left_panel,
            "right_panel": right_panel,
            "left_btn_prev": lb,
            "left_btn_next": ln,
            "copy_btn": cp,
            "delete_btn": db,
            "right_btn_prev": rb,
            "right_btn_next": rn,
            "right_scale": rs,
            "right_ox": rox,
            "right_oy": roy,
        }

        cv2.imshow(WINDOW_NAME, canvas)

    def canvas_to_right_image(self, x, y):
        x1, y1, x2, y2 = self.layout["right_panel"]
        if not (x1 <= x < x2 and y1 <= y < y2):
            return None

        px = x - x1
        py = y - y1
        scale = self.layout["right_scale"]
        ox = self.layout["right_ox"]
        oy = self.layout["right_oy"]

        if px < ox or py < oy:
            return None
        if px >= ox + int(round(self.right_img.shape[1] * scale)):
            return None
        if py >= oy + int(round(self.right_img.shape[0] * scale)):
            return None

        rx = int(round((px - ox) / max(scale, 1e-9)))
        ry = int(round((py - oy) / max(scale, 1e-9)))
        rx = max(0, min(self.right_img.shape[1] - 1, rx))
        ry = max(0, min(self.right_img.shape[0] - 1, ry))
        return rx, ry

    def on_mouse(self, event, x, y, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            if point_in_rect(x, y, self.layout["left_btn_prev"]):
                self.prev_ref()
                return
            if point_in_rect(x, y, self.layout["left_btn_next"]):
                self.next_ref()
                return
            if point_in_rect(x, y, self.layout["copy_btn"]):
                self.copy_left_to_right()
                return
            if point_in_rect(x, y, self.layout["delete_btn"]):
                self.delete_current_right_image_and_labels()
                return
            if point_in_rect(x, y, self.layout["right_btn_prev"]):
                self.prev_target()
                return
            if point_in_rect(x, y, self.layout["right_btn_next"]):
                self.next_target()
                return

        p = self.canvas_to_right_image(x, y)
        if p is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.insert_point_into_active_polygon(p[0], p[1])
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.remove_closest_point(p[0], p[1])

    def loop(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1700, 1120)
        cv2.setMouseCallback(WINDOW_NAME, lambda e, x, y, f, p=None: self.on_mouse(e, x, y, f))

        while True:
            key = cv2.waitKeyEx(30)
            if key < 0:
                continue

            if key in (27, ord("q"), ord("Q")):
                break
            elif key in (ord("s"), ord("S"), 19):
                self.save_current()
            elif key in (ord("c"), ord("C")):
                self.copy_left_to_right()
            elif key in (ord("n"), ord("N")):
                self.add_new_polygon()
            elif key in (9,):
                self.cycle_active_polygon()
            elif key in (8,):
                self.delete_active_polygon()
            elif key in (ord("0"), ord("1"), ord("2"), ord("3")):
                self.active_class = int(chr(key))
                if 0 <= self.active_poly_idx < len(self.right_polys):
                    self.right_polys[self.active_poly_idx]["cls"] = self.active_class
                self.message = f"Active class set to {self.active_class}"
                self.render()
            elif key in (3014656, 127):
                self.delete_current_right_image_and_labels()

        cv2.destroyAllWindows()


def main():
    app = Annotator()
    app.loop()


if __name__ == "__main__":
    main()