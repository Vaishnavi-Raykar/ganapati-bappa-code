"""
generate_vector_json.py

Generates 100% clean, silky smooth vector_data.json for ganapati_python/ directly from reference_image.png.
Applies double-pass Gaussian curve smoothing for anti-aliased, ultra-smooth line drawing.
Supports 3x Supersample space (2700x3300).
"""

import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
REF_IMG_PATH = ROOT / "assets" / "reference_image.png"
OUT_JSON = ROOT / "ganapati_python" / "vector_data.json"

SCALE = 2700.0 / 498.0
CENTER_REF_X = 236.5
CENTER_REF_Y = 268.0
CANVAS_CENTER_X = 1350.0
CANVAS_CENTER_Y = 1650.0

def ref_to_canvas(rx, ry):
    cx = (rx - CENTER_REF_X) * SCALE + CANVAS_CENTER_X
    cy = (ry - CENTER_REF_Y) * SCALE + CANVAS_CENTER_Y
    return (round(cx, 2), round(cy, 2))

def smooth_points(pts_list, passes=2):
    """Applies multi-pass Gaussian 1D kernel smoothing to create ultra-smooth vector curves."""
    if len(pts_list) <= 3:
        return pts_list
    arr = np.array(pts_list, dtype=float)
    kernel = np.array([0.1, 0.8, 0.1])
    
    for _ in range(passes):
        smoothed = np.zeros_like(arr)
        smoothed[:, 0] = np.convolve(arr[:, 0], kernel, mode='same')
        smoothed[:, 1] = np.convolve(arr[:, 1], kernel, mode='same')
        smoothed[0] = arr[0]
        smoothed[-1] = arr[-1]
        arr = smoothed

    return [(round(row[0], 2), round(row[1], 2)) for row in arr]

def fit_cubic_bezier(points):
    pts = np.array(points, dtype=float)
    n = len(pts)
    if n <= 1:
        p = tuple(pts[0]) if n == 1 else (CANVAS_CENTER_X, CANVAS_CENTER_Y)
        return [p, p, p, p]
    p0, p3 = pts[0], pts[-1]
    dists = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    cum_dists = np.insert(np.cumsum(dists), 0, 0)
    total_len = cum_dists[-1]
    if total_len == 0:
        return [tuple(p0), tuple(p0), tuple(p0), tuple(p3)]
    t = cum_dists / total_len
    u = 1.0 - t
    b1 = 3.0 * (u**2) * t
    b2 = 3.0 * u * (t**2)
    rhs = pts - np.outer((u**3), p0) - np.outer((t**3), p3)
    A = np.column_stack([b1, b2])
    try:
        res, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
        p1, p2 = res[0], res[1]
    except Exception:
        p1 = p0 + (p3 - p0) / 3.0
        p2 = p0 + 2.0 * (p3 - p0) / 3.0
    return [tuple(p0), tuple(p1), tuple(p2), tuple(p3)]

def fit_bezier_spline(pts, max_pts=10):
    bezier_segs = []
    for i in range(0, len(pts) - 1, max_pts - 1):
        chunk = pts[i : i + max_pts]
        if len(chunk) >= 2:
            seg = fit_cubic_bezier([[p[0], p[1]] for p in chunk])
            bezier_segs.append(seg)
    return bezier_segs

def classify_path(pts):
    avg_x = sum(p[0] for p in pts) / len(pts)
    avg_y = sum(p[1] for p in pts) / len(pts)
    min_x = min(p[0] for p in pts)
    max_x = max(p[0] for p in pts)
    min_y = min(p[1] for p in pts)
    max_y = max(p[1] for p in pts)

    if min_y < 120 and avg_x > 180 and max_x > 210 and not (min_x < 170 and max_y > 110):
        return (1, 'Crown')
    if max_x < 220 and min_y > 75 and max_y < 235 and avg_x < 185:
        return (3, 'Left Ear')
    if min_x > 280 and min_y > 130 and max_y < 310:
        return (4, 'Right Ear')
    if 180 <= avg_x <= 220 and 145 <= avg_y <= 200:
        return (7, 'Left Eye')
    if 255 <= avg_x <= 300 and 165 <= avg_y <= 220:
        return (8, 'Right Eye')
    if 210 <= avg_x <= 320 and 100 <= min_y <= 165 and max_y < 175:
        return (6, 'Forehead')
    if 150 <= min_x and max_x <= 340 and 115 <= min_y <= 250 and avg_y < 260 and not (avg_y > 200 and min_x < 175):
        return (2, 'Head Outline')
    if max_x < 175 and min_y > 190 and max_y < 310 and min_x < 140:
        if min_x < 85:
            return (11, 'Left Hand')
        return (10, 'Left Arm')
    if min_x > 270 and min_y > 260 and max_y < 360:
        if max_x > 345:
            return (13, 'Right Hand')
        return (12, 'Right Arm')
    if 160 <= min_x and max_x <= 265 and 190 <= min_y and max_y <= 340:
        return (9, 'Trunk')
    if 150 <= min_x and max_x <= 265 and 235 <= min_y <= 345 and avg_y < 350:
        return (14, 'Necklace')
    if 135 <= min_x and max_x <= 200 and 335 <= min_y and max_y <= 435:
        if avg_y < 375:
            return (17, 'Waist Knot')
        return (18, 'Hanging Cloth')
    if 140 <= min_x and max_x <= 290 and 240 <= min_y and max_y <= 370:
        return (15, 'Torso/Upper Clothing')
    if max_y > 470:
        return (23, 'Feet')
    if min_y > 440 and max_y <= 485:
        return (21, 'Legs')
    if min_y > 350 and max_y <= 480:
        if (max_x - min_x) < 40 and avg_y > 390:
            return (20, 'Dhoti Folds')
        return (19, 'Dhoti')

    return (24, 'Small Details')

def generate():
    img = cv2.imread(str(REF_IMG_PATH), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Missing {REF_IMG_PATH}")

    h, w = img.shape

    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    processed_paths = []
    for idx, c in enumerate(contours):
        if cv2.arcLength(c, False) < 10:
            continue

        pts_ref = [(float(pt[0][0]), float(pt[0][1])) for pt in c]
        min_x = min(p[0] for p in pts_ref)
        max_x = max(p[0] for p in pts_ref)
        min_y = min(p[1] for p in pts_ref)
        max_y = max(p[1] for p in pts_ref)

        # Discard ONLY the outer image bounding box rectangle (spans width > 350 AND height > 450)
        if (max_x - min_x) > 350 and (max_y - min_y) > 450:
            continue

        # Discard top/left edge noise lines
        if max_y < 12 or max_x < 25 or min_x > 450:
            continue

        step_num, category = classify_path(pts_ref)
        canvas_pts = [ref_to_canvas(p[0], p[1]) for p in pts_ref]

        smoothed = smooth_points(canvas_pts, passes=2)
        bezier_segs = fit_bezier_spline(smoothed)

        width_super = 7.5 if category in ['Head Outline', 'Crown', 'Left Ear', 'Right Ear', 'Dhoti', 'Feet', 'Legs'] else 5.5
        name_str = f"{category.lower().replace(' ', '_')}_{idx}"

        processed_paths.append({
            'id': idx,
            'name': name_str,
            'category': category,
            'step_order': step_num,
            'bezier_segments': bezier_segs,
            'points': smoothed,
            'width': width_super,
            'is_closed': False,
            'is_filled': False,
            'fill_color': None
        })

    processed_paths.sort(key=lambda p: (p['step_order'], p['id']))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(processed_paths, f, indent=2)

    print(f"Generated {len(processed_paths)} silky-smooth vector paths -> {OUT_JSON}")

if __name__ == "__main__":
    generate()
