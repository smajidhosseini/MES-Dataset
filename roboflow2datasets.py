import re
import copy
from pathlib import Path
import cv2
import numpy as np

# Updated Directories
REF_IMAGE_DIR = Path(r"C:\Users\majid\Documents\Projects\MES\Dataset\train\images")
REF_LABEL_DIR = Path(r"C:\Users\majid\Documents\Projects\MES\Dataset\train\labels")

TARGET_IMAGE_DIR = Path(r"C:\Users\majid\Documents\Projects\MES\Dataset\train\clean_dataset\unlabeled\train")
TARGET_SAVE_IMAGE_DIR = Path(r"C:\Users\majid\Documents\Projects\MES\Dataset\train\clean_dataset\labeled\train\images")
TARGET_LABEL_DIR = Path(r"C:\Users\majid\Documents\Projects\MES\Dataset\train\clean_dataset\labeled\train\labels")

WINDOW_NAME = "MES Annotator"

CLASSES = [0, 1, 2, 3]
PANEL_W, PANEL_H = 820, 820
TOP_H, MARGIN = 110, 20

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


def get_class_color(cls_id):
    if cls_id not in CLASS_COLORS:
        r = (cls_id * 50) % 200 + 55
        g = (cls_id * 80) % 200 + 55
        b = (cls_id * 110) % 200 + 55
        CLASS_COLORS[cls_id] = (b, g, r)
    return CLASS_COLORS[cls_id]


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def imread_any(path):
    arr = np.fromfile(str(path), dtype=np.uint8)
    return None if arr.size == 0 else cv2.imdecode(arr, cv2.IMREAD_COLOR)


def imwrite_any(path, img):
    ensure_dir(Path(path).parent)
    ext = Path(path).suffix.lower()
    encoded = cv2.imencode('.png' if ext == '.png' else '.jpg', img)[1]
    encoded.tofile(str(path))


def parse_info(img_path):
    name = Path(img_path).name
    m = FILE_RE.search(name)
    if m:
        return {"video": m.group("video"), "frame": int(m.group("frame")), "name": name}
    stem = Path(img_path).stem
    return {"video": stem.split("_", 1)[0], "frame": -1, "name": name}


def label_path_for_image(img_path, base_dir):
    return Path(base_dir) / f"{Path(img_path).stem}.txt"


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
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    if direction == "left":
        pts = np.array([[cx + 10, cy - 12], [cx - 10, cy], [cx + 10, cy + 12]], np.int32)
    else:
        pts = np.array([[cx - 10, cy - 12], [cx + 10, cy], [cx - 10, cy + 12]], np.int32)
    cv2.fillConvexPoly(canvas, pts, BTN_FG)
    cv2.putText(canvas, label, (x1 + 8, y2 + 18), FONT, 0.45, TEXT, 1, cv2.LINE_AA)


def draw_text_button(canvas, rect, label, active=False):
    x1, y1, x2, y2 = rect
    bg = (80, 140, 80) if active else BTN_BG
    cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), 1)
    cv2.putText(canvas, label, (x1 + 8, y2 - 12), FONT, 0.45, TEXT, 1, cv2.LINE_AA)


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
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx, cy = ax + t * abx, ay + t * aby
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


class Annotator:
    def __init__(self):
        ensure_dir(TARGET_SAVE_IMAGE_DIR)
        ensure_dir(TARGET_LABEL_DIR)

        self.ref_records = self.build_records(REF_IMAGE_DIR)
        self.target_records = self.build_records(TARGET_IMAGE_DIR)

        if not self.ref_records:
            print("Warning: No reference images found.")
        if not self.target_records:
            print("Warning: No target images found.")

        self.left_pos = 0
        self.target_pos = 0

        self.left_img, self.right_img = None, None
        self.left_polys, self.right_polys = [], []
        
        # Persistent border variables across dataset
        self.persistent_left_roi = None 
        self.persistent_right_roi = None
        
        # Active borders for current viewing frame
        self.left_roi = None  
        self.right_roi = None

        self.active_poly_idx = -1
        self.active_class = 0
        
        self.mode = "POLY"
        self.roi_start_pt = None
        
        self.is_editing_class = False
        self.class_input_buffer = ""
        
        self.message = ""
        self.layout = {}

        self.load_left_only()
        self.load_right_only()

    def build_records(self, directory):
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"):
            imgs.extend(directory.glob(ext))
            imgs.extend(directory.glob(ext.upper()))
        rows = []
        for p in sorted(set(imgs)):
            info = parse_info(p)
            rows.append({"img_path": p, "name": info["name"]})
        return rows

    def load_left_reference(self):
        if not self.ref_records:
            self.left_img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            self.left_polys = []
            self.left_roi = (0, 0, PANEL_W, PANEL_H)
            return

        rec = self.ref_records[self.left_pos]
        self.left_img = imread_any(rec["img_path"])
        if self.left_img is None:
            self.left_polys = []
            self.left_roi = (0, 0, PANEL_W, PANEL_H)
            return

        h, w = self.left_img.shape[:2]
        
        if self.persistent_left_roi:
            px, py, pw, ph = self.persistent_left_roi
            px = max(0, min(px, w - 1))
            py = max(0, min(py, h - 1))
            pw = max(1, min(pw, w - px))
            ph = max(1, min(ph, h - py))
            self.left_roi = (px, py, pw, ph)
        else:
            self.left_roi = (0, 0, w, h)
            
        label_path = label_path_for_image(rec["img_path"], REF_LABEL_DIR)
        self.left_polys = load_yolo_polygons(label_path, w, h)

    def load_right_annotation(self):
        if not self.target_records:
            self.right_polys = []
            self.active_poly_idx = -1
            return

        rec = self.target_records[self.target_pos]
        h, w = self.right_img.shape[:2]
        label_path = label_path_for_image(rec["img_path"], TARGET_LABEL_DIR)

        if label_path.exists():
            self.right_polys = load_yolo_polygons(label_path, w, h)
            self.message = f"Loaded label: {label_path.name}"
        else:
            self.right_polys = []
            self.message = "Target has no annotation yet."

        self.active_poly_idx = 0 if self.right_polys else -1
        self.active_class = int(self.right_polys[self.active_poly_idx]["cls"]) if self.active_poly_idx >= 0 else CLASSES[0]

    def load_left_only(self):
        self.load_left_reference()
        if self.right_img is None:
            self.right_img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            self.right_roi = (0, 0, PANEL_W, PANEL_H)
        self.render()

    def load_right_only(self):
        if not self.target_records:
            self.right_img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            self.right_roi = (0, 0, PANEL_W, PANEL_H)
            self.right_polys = []
            self.active_poly_idx = -1
            self.render()
            return

        rec = self.target_records[self.target_pos]
        
        # Check if we already have a cropped/saved version of this image
        saved_img_path = TARGET_SAVE_IMAGE_DIR / rec["name"]
        if saved_img_path.exists():
            self.right_img = imread_any(saved_img_path)
            if self.right_img is None:
                self.right_img = imread_any(rec["img_path"])
        else:
            self.right_img = imread_any(rec["img_path"])

        if self.right_img is None:
            raise RuntimeError(f"Could not read image: {rec['img_path']}")

        h, w = self.right_img.shape[:2]
        
        # If loading a previously saved/cropped image, the persistent ROI shouldn't apply physically to it again 
        # (since it's already cropped), but for consistency we assign the whole image bounds.
        if saved_img_path.exists():
            self.right_roi = (0, 0, w, h)
        elif self.persistent_right_roi:
            px, py, pw, ph = self.persistent_right_roi
            px = max(0, min(px, w - 1))
            py = max(0, min(py, h - 1))
            pw = max(1, min(pw, w - px))
            ph = max(1, min(ph, h - py))
            self.right_roi = (px, py, pw, ph)
        else:
            self.right_roi = (0, 0, w, h)
            
        self.load_right_annotation()
        
        if self.left_img is None:
            self.load_left_reference()
        self.render()

    def prev_target(self):
        if self.target_records and self.target_pos > 0:
            self.target_pos -= 1
            self.load_right_only()

    def next_target(self):
        if self.target_records and self.target_pos < len(self.target_records) - 1:
            self.target_pos += 1
            self.load_right_only()

    def fast_prev_target(self):
        if self.target_records and self.target_pos > 0:
            self.target_pos = max(0, self.target_pos - 100)
            self.load_right_only()

    def fast_next_target(self):
        if self.target_records and self.target_pos < len(self.target_records) - 1:
            self.target_pos = min(len(self.target_records) - 1, self.target_pos + 100)
            self.load_right_only()

    def prev_ref(self):
        if self.ref_records and self.left_pos > 0:
            self.left_pos -= 1
            self.load_left_only()
            self.message = f"Left reference: {self.ref_records[self.left_pos]['name']}"
            self.render()

    def next_ref(self):
        if self.ref_records and self.left_pos < len(self.ref_records) - 1:
            self.left_pos += 1
            self.load_left_only()
            self.message = f"Left reference: {self.ref_records[self.left_pos]['name']}"
            self.render()

    def fast_prev_ref(self):
        if self.ref_records and self.left_pos > 0:
            self.left_pos = max(0, self.left_pos - 100)
            self.load_left_only()
            self.message = f"Left reference jumped -100: {self.ref_records[self.left_pos]['name']}"
            self.render()

    def fast_next_ref(self):
        if self.ref_records and self.left_pos < len(self.ref_records) - 1:
            self.left_pos = min(len(self.ref_records) - 1, self.left_pos + 100)
            self.load_left_only()
            self.message = f"Left reference jumped +100: {self.ref_records[self.left_pos]['name']}"
            self.render()

    def cycle_active_polygon(self):
        if not self.right_polys:
            self.active_poly_idx = -1
            self.message = "No polygon on right image."
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
        lx, ly, lw, lh = self.left_roi
        rx, ry, rw, rh = self.right_roi
        
        # Scaling variables between left and right ROIs
        scale_x = rw / lw if lw > 0 else 1
        scale_y = rh / lh if lh > 0 else 1

        self.right_polys = []
        for poly in self.left_polys:
            new_pts = []
            for px, py in poly["pts"]:
                nx = rx + (px - lx) * scale_x
                ny = ry + (py - ly) * scale_y
                new_pts.append((int(nx), int(ny)))
            self.right_polys.append({"cls": poly["cls"], "pts": new_pts})

        self.active_poly_idx = 0 if self.right_polys else -1
        if self.active_poly_idx >= 0:
            self.active_class = int(self.right_polys[self.active_poly_idx]["cls"])
            
        self.message = "Copied and transformed annotation relative to borders."
        self.render()

    def save_current(self):
        if not self.target_records:
            self.message = "No target image."
            self.render()
            return

        rec = self.target_records[self.target_pos]
        img_h, img_w = self.right_img.shape[:2]
        rx, ry, rw, rh = self.right_roi
        
        img_to_save = self.right_img
        
        # If Right ROI is distinct, perform physical cropping
        if rw < img_w or rh < img_h or rx > 0 or ry > 0:
            img_to_save = self.right_img[ry:ry+rh, rx:rx+rw]
            
            # Map polys to cropped dimension bounds
            for poly in self.right_polys:
                for i, (px, py) in enumerate(poly["pts"]):
                    nx = max(0, min(rw - 1, px - rx))
                    ny = max(0, min(rh - 1, py - ry))
                    poly["pts"][i] = (nx, ny)
            
            self.right_img = img_to_save
            self.right_roi = (0, 0, rw, rh)
            img_w, img_h = rw, rh

        # Write to TARGET_SAVE_IMAGE_DIR
        out_img_path = TARGET_SAVE_IMAGE_DIR / rec["name"]
        imwrite_any(out_img_path, img_to_save)

        # Write to TARGET_LABEL_DIR
        out_label_path = label_path_for_image(rec["img_path"], TARGET_LABEL_DIR)
        save_yolo_polygons(self.right_polys, img_w, img_h, out_label_path)
        
        self.message = f"Saved Image & Label: {out_label_path.name}"
        self.render()

    def delete_current_right_image_and_labels(self):
        if not self.target_records:
            self.message = "No target image to delete."
            self.render()
            return

        rec = self.target_records.pop(self.target_pos)
        
        # Paths to potentially delete
        img_path_orig = Path(rec["img_path"])
        img_path_saved = TARGET_SAVE_IMAGE_DIR / rec["name"]
        label_path = label_path_for_image(rec["img_path"], TARGET_LABEL_DIR)

        deleted = []
        if img_path_orig.exists():
            img_path_orig.unlink()
            deleted.append(img_path_orig.name)
            
        if img_path_saved.exists():
            img_path_saved.unlink()
            if img_path_saved.name not in deleted:
                deleted.append(img_path_saved.name + " (cropped)")
                
        if label_path.exists():
            label_path.unlink()
            deleted.append(label_path.name)

        if not self.target_records:
            self.right_polys = []
            self.right_img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            self.active_poly_idx = -1
            self.message = f"Deleted: {', '.join(deleted)}. No target images left."
            self.target_pos = 0
            self.render()
            return

        self.target_pos = min(self.target_pos, len(self.target_records) - 1)
        self.message = f"Deleted: {', '.join(deleted)}"
        self.load_right_only()

    def commit_class_input(self):
        if self.class_input_buffer.isdigit():
            cls_id = int(self.class_input_buffer)
            self.active_class = cls_id
            if cls_id not in CLASSES:
                CLASSES.append(cls_id)
                self.message = f"Created new class: {cls_id}"
            else:
                self.message = f"Active class set to {cls_id}"
            if 0 <= self.active_poly_idx < len(self.right_polys):
                self.right_polys[self.active_poly_idx]["cls"] = self.active_class
        self.is_editing_class = False
        self.class_input_buffer = ""
        self.render()

    def ensure_active_poly(self):
        if self.active_poly_idx < 0 or self.active_poly_idx >= len(self.right_polys):
            self.right_polys.append({"cls": self.active_class, "pts": []})
            self.active_poly_idx = len(self.right_polys) - 1

    def insert_point_into_active_polygon(self, x, y):
        self.ensure_active_poly()
        poly = self.right_polys[self.active_poly_idx]
        poly["cls"] = self.active_class
        pts = poly["pts"]
        click = (int(x), int(y))

        if len(pts) < 2:
            pts.append(click)
        else:
            best_insert_after = 0
            best_dist = None
            n = len(pts)
            for i in range(n):
                d = point_to_segment_distance(click, pts[i], pts[(i + 1) % n])
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

    def draw_polys_on_panel(self, base_img, polys, panel_rect, active_idx=None, show_class_text=False, roi=None):
        x1, y1, x2, y2 = panel_rect
        panel_w, panel_h = x2 - x1, y2 - y1

        view = np.full((panel_h, panel_w, 3), BG, dtype=np.uint8)
        resized, scale, ox, oy = fit_image(base_img, panel_w, panel_h)
        view[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
        
        # Render ROI if specified
        if roi:
            rx, ry, rw, rh = roi
            px1, py1 = int(rx * scale) + ox, int(ry * scale) + oy
            px2, py2 = int((rx + rw) * scale) + ox, int((ry + rh) * scale) + oy
            cv2.rectangle(view, (px1, py1), (px2, py2), (80, 255, 80), 2, cv2.LINE_AA)
            
            # Display width and height label above the ROI
            label = f"{rw}x{rh}"
            cv2.putText(view, label, (px1, max(15, py1 - 5)), FONT, 0.5, (80, 255, 80), 1, cv2.LINE_AA)

        for i, poly in enumerate(polys):
            color = get_class_color(int(poly["cls"]))
            pts = poly["pts"]
            if not pts:
                continue

            pts2 = [(int(round(px * scale)) + ox, int(round(py * scale)) + oy) for px, py in pts]
            if len(pts2) >= 2:
                thickness = 4 if i == active_idx else 2
                line_color = ACTIVE if i == active_idx else color
                cv2.polylines(view, [np.array(pts2, np.int32)], True, line_color, thickness, cv2.LINE_AA)

            for j, (rx_pt, ry_pt) in enumerate(pts2):
                point_color = ACTIVE if i == active_idx else color
                radius = 5 if i == active_idx else 3
                cv2.circle(view, (rx_pt, ry_pt), radius, point_color, -1, cv2.LINE_AA)
                if i == active_idx:
                    cv2.putText(view, str(j), (rx_pt + 4, ry_pt - 4), FONT, 0.35, ACTIVE, 1, cv2.LINE_AA)

            tx, ty = pts2[0]
            if show_class_text:
                cv2.putText(view, f"class {int(poly['cls'])}", (tx + 6, max(18, ty - 8)), FONT, 0.55, color, 2, cv2.LINE_AA)
            if i == active_idx:
                cv2.putText(view, f"active #{i} c{int(poly['cls'])}", (tx + 6, min(panel_h - 10, ty + 18)), FONT, 0.5, ACTIVE, 2, cv2.LINE_AA)

        return view, scale, ox, oy

    def render(self):
        canvas_w = PANEL_W * 2 + MARGIN * 3
        canvas_h = PANEL_H + TOP_H + MARGIN * 2 + 145
        canvas = np.full((canvas_h, canvas_w, 3), BG, dtype=np.uint8)

        left_panel = (MARGIN, TOP_H + MARGIN, MARGIN + PANEL_W, TOP_H + MARGIN + PANEL_H)
        right_panel = (MARGIN * 2 + PANEL_W, TOP_H + MARGIN, MARGIN * 2 + PANEL_W * 2, TOP_H + MARGIN + PANEL_H)

        # Row 1 Navigation and Action Buttons
        base_lx = MARGIN + 20
        lb_fast = (base_lx, 20, base_lx + 60, 50)
        lb      = (base_lx + 70, 20, base_lx + 120, 50)
        ln      = (base_lx + 130, 20, base_lx + 180, 50)
        ln_fast = (base_lx + 190, 20, base_lx + 250, 50)
        
        cp = (base_lx + 270, 20, base_lx + 340, 50)
        db = (base_lx + 360, 20, base_lx + 440, 50)

        base_rx = MARGIN * 2 + PANEL_W + 20
        rb_fast = (base_rx, 20, base_rx + 60, 50)
        rb      = (base_rx + 70, 20, base_rx + 120, 50)
        rn      = (base_rx + 130, 20, base_rx + 180, 50)
        rn_fast = (base_rx + 190, 20, base_rx + 250, 50)
        
        cb = (base_rx + 270, 20, base_rx + 390, 50)
        
        # Row 2 Buttons for ROI
        lroi_btn = (MARGIN + 20, 65, MARGIN + 140, 95)
        rroi_btn = (MARGIN * 2 + PANEL_W + 20, 65, MARGIN * 2 + PANEL_W + 140, 95)

        draw_text_button(canvas, lb_fast, "-100")
        draw_arrow_button(canvas, lb, "left", "Ref")
        draw_arrow_button(canvas, ln, "right", "Ref")
        draw_text_button(canvas, ln_fast, "+100")

        draw_copy_button(canvas, cp, "Copy")
        draw_delete_button(canvas, db, "Delete")
        
        draw_text_button(canvas, rb_fast, "-100")
        draw_arrow_button(canvas, rb, "left", "Target")
        draw_arrow_button(canvas, rn, "right", "Target")
        draw_text_button(canvas, rn_fast, "+100")

        draw_text_button(canvas, lroi_btn, "Set L-Border", self.mode == "DRAW_L_ROI")
        draw_text_button(canvas, rroi_btn, "Set R-Border", self.mode == "DRAW_R_ROI")

        cx1, cy1, cx2, cy2 = cb
        box_bg_color = (120, 120, 160) if self.is_editing_class else BTN_BG
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), box_bg_color, -1)
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (180, 180, 180), 1)
        text = f"Set Cls: {self.class_input_buffer}_" if self.is_editing_class else f"Class: {self.active_class}"
        cv2.putText(canvas, text, (cx1 + 8, cy2 - 14), FONT, 0.55, TEXT, 1, cv2.LINE_AA)

        left_base = self.left_img if self.left_img is not None else np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        right_base = self.right_img if self.right_img is not None else np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

        left_view, ls, lox, loy = self.draw_polys_on_panel(left_base, self.left_polys, left_panel, None, show_class_text=True, roi=self.left_roi)
        right_view, rs, rox, roy = self.draw_polys_on_panel(right_base, self.right_polys, right_panel, self.active_poly_idx, show_class_text=False, roi=self.right_roi)

        canvas[left_panel[1]:left_panel[3], left_panel[0]:left_panel[2]] = left_view
        canvas[right_panel[1]:right_panel[3], right_panel[0]:right_panel[2]] = right_view

        left_title = f"Reference {self.left_pos + 1}/{len(self.ref_records)}: {self.ref_records[self.left_pos]['name']}" if self.ref_records else "Reference: none"
        right_title = f"Target {self.target_pos + 1}/{len(self.target_records)}: {self.target_records[self.target_pos]['name']}" if self.target_records else "Target: none"

        cv2.putText(canvas, left_title, (left_panel[0], left_panel[1] - 10), FONT, 0.6, TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, right_title, (right_panel[0], right_panel[1] - 10), FONT, 0.6, TEXT, 1, cv2.LINE_AA)

        info_y = TOP_H + PANEL_H + MARGIN + 35
        cv2.putText(canvas, f"Active class: {self.active_class} | Active polygon: {self.active_poly_idx} | Right polygons: {len(self.right_polys)}", (MARGIN, info_y), FONT, 0.55, TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Left click adds point | Right click deletes nearest | Set L/R Border -> drag box -> copy to map relative pos.", (MARGIN, info_y + 28), FONT, 0.48, TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, "N new poly | Backspace delete active | S save & crop to new folder | C copy relative | Delete destroys | Q quit", (MARGIN, info_y + 56), FONT, 0.48, TEXT, 1, cv2.LINE_AA)

        if self.message:
            cv2.putText(canvas, self.message, (MARGIN, info_y + 88), FONT, 0.52, ACTIVE, 1, cv2.LINE_AA)

        self.layout = {
            "left_panel": left_panel, "right_panel": right_panel,
            "lb_fast": lb_fast, "left_btn_prev": lb, "left_btn_next": ln, "ln_fast": ln_fast,
            "copy_btn": cp, "delete_btn": db,
            "rb_fast": rb_fast, "right_btn_prev": rb, "right_btn_next": rn, "rn_fast": rn_fast,
            "class_box": cb, "lroi_btn": lroi_btn, "rroi_btn": rroi_btn,
            "left_scale": ls, "left_ox": lox, "left_oy": loy,
            "right_scale": rs, "right_ox": rox, "right_oy": roy,
        }
        cv2.imshow(WINDOW_NAME, canvas)

    def canvas_to_left_image(self, x, y):
        x1, y1, x2, y2 = self.layout["left_panel"]
        if not (x1 <= x < x2 and y1 <= y < y2):
            return None
        px, py = x - x1, y - y1
        scale, ox, oy = self.layout["left_scale"], self.layout["left_ox"], self.layout["left_oy"]
        if px < ox or py < oy or px >= ox + int(round(self.left_img.shape[1] * scale)) or py >= oy + int(round(self.left_img.shape[0] * scale)):
            return None
        rx = int(round((px - ox) / max(scale, 1e-9)))
        ry = int(round((py - oy) / max(scale, 1e-9)))
        return max(0, min(self.left_img.shape[1] - 1, rx)), max(0, min(self.left_img.shape[0] - 1, ry))

    def canvas_to_right_image(self, x, y):
        x1, y1, x2, y2 = self.layout["right_panel"]
        if not (x1 <= x < x2 and y1 <= y < y2):
            return None
        px, py = x - x1, y - y1
        scale, ox, oy = self.layout["right_scale"], self.layout["right_ox"], self.layout["right_oy"]
        if px < ox or py < oy or px >= ox + int(round(self.right_img.shape[1] * scale)) or py >= oy + int(round(self.right_img.shape[0] * scale)):
            return None
        rx = int(round((px - ox) / max(scale, 1e-9)))
        ry = int(round((py - oy) / max(scale, 1e-9)))
        return max(0, min(self.right_img.shape[1] - 1, rx)), max(0, min(self.right_img.shape[0] - 1, ry))

    def on_mouse(self, event, x, y, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            if point_in_rect(x, y, self.layout["class_box"]):
                self.is_editing_class = True
                self.class_input_buffer = ""
                self.render()
                return
            if self.is_editing_class:
                self.commit_class_input()
                
            # Mode Toggles
            if point_in_rect(x, y, self.layout["lroi_btn"]):
                self.mode = "POLY" if self.mode == "DRAW_L_ROI" else "DRAW_L_ROI"
                self.roi_start_pt = None
                self.render(); return
            if point_in_rect(x, y, self.layout["rroi_btn"]):
                self.mode = "POLY" if self.mode == "DRAW_R_ROI" else "DRAW_R_ROI"
                self.roi_start_pt = None
                self.render(); return

            if point_in_rect(x, y, self.layout["lb_fast"]): return self.fast_prev_ref()
            if point_in_rect(x, y, self.layout["left_btn_prev"]): return self.prev_ref()
            if point_in_rect(x, y, self.layout["left_btn_next"]): return self.next_ref()
            if point_in_rect(x, y, self.layout["ln_fast"]): return self.fast_next_ref()

            if point_in_rect(x, y, self.layout["copy_btn"]): return self.copy_left_to_right()
            if point_in_rect(x, y, self.layout["delete_btn"]): return self.delete_current_right_image_and_labels()
            
            if point_in_rect(x, y, self.layout["rb_fast"]): return self.fast_prev_target()
            if point_in_rect(x, y, self.layout["right_btn_prev"]): return self.prev_target()
            if point_in_rect(x, y, self.layout["right_btn_next"]): return self.next_target()
            if point_in_rect(x, y, self.layout["rn_fast"]): return self.fast_next_target()
            
            # Initiate Border Draw
            if self.mode == "DRAW_L_ROI":
                self.roi_start_pt = self.canvas_to_left_image(x, y)
                return
            if self.mode == "DRAW_R_ROI":
                self.roi_start_pt = self.canvas_to_right_image(x, y)
                return

        # Continuing Border Draw dynamically
        if event == cv2.EVENT_MOUSEMOVE:
            if self.mode == "DRAW_L_ROI" and self.roi_start_pt:
                p = self.canvas_to_left_image(x, y)
                if p:
                    sx, sy = self.roi_start_pt
                    cx, cy = p
                    self.left_roi = (min(sx, cx), min(sy, cy), abs(cx - sx), abs(cy - sy))
                    self.render()
                return
            if self.mode == "DRAW_R_ROI" and self.roi_start_pt:
                p = self.canvas_to_right_image(x, y)
                if p:
                    sx, sy = self.roi_start_pt
                    cx, cy = p
                    self.right_roi = (min(sx, cx), min(sy, cy), abs(cx - sx), abs(cy - sy))
                    self.render()
                return

        # Finalizing Border Draw
        if event == cv2.EVENT_LBUTTONUP:
            if self.mode in ("DRAW_L_ROI", "DRAW_R_ROI") and self.roi_start_pt:
                if self.mode == "DRAW_L_ROI":
                    if self.left_roi[2] < 10 or self.left_roi[3] < 10:
                        self.left_roi = (0, 0, self.left_img.shape[1], self.left_img.shape[0])
                        self.persistent_left_roi = None
                    else:
                        self.persistent_left_roi = self.left_roi
                        
                elif self.mode == "DRAW_R_ROI":
                    if self.right_roi[2] < 10 or self.right_roi[3] < 10:
                        self.right_roi = (0, 0, self.right_img.shape[1], self.right_img.shape[0])
                        self.persistent_right_roi = None
                    else:
                        self.persistent_right_roi = self.right_roi
                    
                self.mode = "POLY"
                self.roi_start_pt = None
                self.render()
                return

        p = self.canvas_to_right_image(x, y)
        if p is None or self.mode != "POLY": return
        if event == cv2.EVENT_LBUTTONDOWN: self.insert_point_into_active_polygon(*p)
        elif event == cv2.EVENT_RBUTTONDOWN: self.remove_closest_point(*p)

    def loop(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1700, 1120)
        cv2.setMouseCallback(WINDOW_NAME, lambda e, x, y, f, p=None: self.on_mouse(e, x, y, f))

        while True:
            key = cv2.waitKeyEx(30)
            if key < 0: continue
            
            if self.is_editing_class:
                if 48 <= key <= 57:  # 0-9
                    self.class_input_buffer += chr(key)
                    self.render()
                elif key in (8, 127):  # Backspace
                    self.class_input_buffer = self.class_input_buffer[:-1]
                    self.render()
                elif key in (13, 10, 141, 271):  # Enter
                    self.commit_class_input()
                elif key == 27:  # Esc
                    self.is_editing_class = False
                    self.class_input_buffer = ""
                    self.render()
                continue

            if key in (27, ord("q"), ord("Q")): break
            elif key in (ord("s"), ord("S"), 19): self.save_current()
            elif key in (ord("c"), ord("C")): self.copy_left_to_right()
            elif key in (ord("n"), ord("N")): self.add_new_polygon()
            elif key in (9,): self.cycle_active_polygon()
            elif key in (8,): self.delete_active_polygon()
            elif key in (ord("0"), ord("1"), ord("2"), ord("3")):
                self.active_class = int(chr(key))
                if 0 <= self.active_poly_idx < len(self.right_polys):
                    self.right_polys[self.active_poly_idx]["cls"] = self.active_class
                self.message = f"Active class set to {self.active_class}"
                self.render()
            elif key in (3014656, 127): self.delete_current_right_image_and_labels()
            
        cv2.destroyAllWindows()


def main():
    app = Annotator()
    app.loop()

if __name__ == "__main__":
    main()