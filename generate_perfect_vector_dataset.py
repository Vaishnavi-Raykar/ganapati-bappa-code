"""
generate_perfect_vector_dataset.py

High-accuracy vector dataset generator for Ganapati Pygame application.
Root-cause vector fixes:
1. Zhang-Suen 1-pixel skeletonization for Bappa (no double-lines, no random polygon shortcuts).
2. Tangent-matching branch merger: joins 1-pixel stroke fragments across junction nodes into long continuous polylines.
3. Complete anatomical preservation: crown, head, ears, face, trunk, arms, hands, fingers, dhoti, legs, feet, toes.
4. Independent Mooshak Mouse (from new_mouse_reference.png) with clear white-space gap from Bappa's feet/dhoti.
5. Mooshak features: round ears, eye, nose, whiskers, paws holding modak, seated leg, curved tail.
"""

import json
import math
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
BAPPA_REF_PATH = ROOT / "assets" / "reference_image.png"
MOUSE_REF_PATH = ROOT / "assets" / "new_mouse_reference.png"
OUT_JSON = ROOT / "ganapati_python" / "vector_data.json"

SCALE = 2700.0 / 498.0
CENTER_REF_X = 236.5
CENTER_REF_Y = 268.0
CANVAS_CENTER_X = 1350.0
CANVAS_CENTER_Y = 1650.0

def bappa_ref_to_canvas(rx, ry):
    cx = (rx - CENTER_REF_X) * SCALE + CANVAS_CENTER_X
    cy = (ry - CENTER_REF_Y) * SCALE + CANVAS_CENTER_Y
    return (round(cx, 2), round(cy, 2))

def mouse_ref_to_canvas(mx, my, mouse_w=115.0, mouse_h=123.0):
    # Position mouse beside Bappa's right foot at X_ref = 370, Y_ref = 430 (clear white space gap!)
    scale_m = 0.65
    rx = 370.0 + (mx - mouse_w / 2.0) * scale_m
    ry = 430.0 + (my - mouse_h / 2.0) * scale_m
    return bappa_ref_to_canvas(rx, ry)

def zhang_suen_thinning(img):
    """Zhang-Suen 1-pixel thinning for perfect centerline extraction."""
    im = (img > 0).astype(np.uint8)
    prev = np.zeros_like(im)

    while True:
        p2 = np.roll(im, -1, axis=0)
        p3 = np.roll(np.roll(im, -1, axis=0), 1, axis=1)
        p4 = np.roll(im, 1, axis=1)
        p5 = np.roll(np.roll(im, 1, axis=0), 1, axis=1)
        p6 = np.roll(im, 1, axis=0)
        p7 = np.roll(np.roll(im, 1, axis=0), -1, axis=1)
        p8 = np.roll(im, -1, axis=1)
        p9 = np.roll(np.roll(im, -1, axis=0), -1, axis=1)

        A = ((p2 == 0) & (p3 == 1)).astype(int) + \
            ((p3 == 0) & (p4 == 1)).astype(int) + \
            ((p4 == 0) & (p5 == 1)).astype(int) + \
            ((p5 == 0) & (p6 == 1)).astype(int) + \
            ((p6 == 0) & (p7 == 1)).astype(int) + \
            ((p7 == 0) & (p8 == 1)).astype(int) + \
            ((p8 == 0) & (p9 == 1)).astype(int) + \
            ((p9 == 0) & (p2 == 1)).astype(int)

        B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

        m1 = (2 <= B) & (B <= 6) & (A == 1) & (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
        im[m1 & (im == 1)] = 0

        p2 = np.roll(im, -1, axis=0)
        p3 = np.roll(np.roll(im, -1, axis=0), 1, axis=1)
        p4 = np.roll(im, 1, axis=1)
        p5 = np.roll(np.roll(im, 1, axis=0), 1, axis=1)
        p6 = np.roll(im, 1, axis=0)
        p7 = np.roll(np.roll(im, 1, axis=0), -1, axis=1)
        p8 = np.roll(im, -1, axis=1)
        p9 = np.roll(np.roll(im, -1, axis=0), -1, axis=1)

        A = ((p2 == 0) & (p3 == 1)).astype(int) + \
            ((p3 == 0) & (p4 == 1)).astype(int) + \
            ((p4 == 0) & (p5 == 1)).astype(int) + \
            ((p5 == 0) & (p6 == 1)).astype(int) + \
            ((p6 == 0) & (p7 == 1)).astype(int) + \
            ((p7 == 0) & (p8 == 1)).astype(int) + \
            ((p8 == 0) & (p9 == 1)).astype(int) + \
            ((p9 == 0) & (p2 == 1)).astype(int)

        B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

        m2 = (2 <= B) & (B <= 6) & (A == 1) & (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
        im[m2 & (im == 1)] = 0

        if np.array_equal(im, prev):
            break
        prev = im.copy()

    return im * 255

def get_end_tangent(pts, at_start=True, step=4):
    if len(pts) <= 1:
        return (1.0, 0.0)
    step = min(step, len(pts)-1)
    if at_start:
        p0 = pts[0]
        p1 = pts[step]
        dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    else:
        p0 = pts[-step]
        p1 = pts[-1]
        dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    mag = math.hypot(dx, dy)
    if mag == 0:
        return (1.0, 0.0)
    return (dx/mag, dy/mag)

def angle_deg(v1, v2):
    dot = max(-1.0, min(1.0, v1[0]*v2[0] + v1[1]*v2[1]))
    return math.degrees(math.acos(dot))

def smooth_points(pts_list, passes=2):
    if len(pts_list) <= 3:
        return pts_list
    arr = np.array(pts_list, dtype=float)
    kernel = np.array([0.15, 0.7, 0.15])
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

def fit_bezier_spline(pts, max_pts=12):
    bezier_segs = []
    for i in range(0, len(pts) - 1, max_pts - 1):
        chunk = pts[i : i + max_pts]
        if len(chunk) >= 2:
            seg = fit_cubic_bezier([[p[0], p[1]] for p in chunk])
            bezier_segs.append(seg)
    return bezier_segs

def classify_bappa_path(pts_ref):
    avg_x = sum(p[0] for p in pts_ref) / len(pts_ref)
    avg_y = sum(p[1] for p in pts_ref) / len(pts_ref)
    min_x = min(p[0] for p in pts_ref)
    max_x = max(p[0] for p in pts_ref)
    min_y = min(p[1] for p in pts_ref)
    max_y = max(p[1] for p in pts_ref)

    if min_y < 120 and avg_x > 180 and max_x > 210 and not (min_x < 170 and max_y > 110):
        return (1, 'Crown')
    if max_x < 220 and min_y > 75 and max_y < 235 and avg_x < 185:
        return (3, 'Left Ear')
    if min_x > 280 and min_y > 130 and max_y < 310:
        return (4, 'Right Ear')
    if 180 <= avg_x <= 225 and 145 <= avg_y <= 205:
        return (7, 'Left Eye')
    if 255 <= avg_x <= 300 and 165 <= avg_y <= 225:
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

def extract_bappa_paths():
    img = cv2.imread(str(BAPPA_REF_PATH), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Missing {BAPPA_REF_PATH}")

    h, w = img.shape
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    skel = zhang_suen_thinning(thresh)
    skel[:15, :] = 0
    skel[h-15:, :] = 0
    skel[:, :30] = 0
    skel[:, w-30:] = 0

    skel_pts = np.argwhere(skel > 0)
    skel_set = set((pt[1], pt[0]) for pt in skel_pts)

    def get_8nbrs(p):
        x, y = p
        return [(x+dx, y+dy) for dx in [-1,0,1] for dy in [-1,0,1] if (dx!=0 or dy!=0) and (x+dx, y+dy) in skel_set]

    deg = {p: len(get_8nbrs(p)) for p in skel_set}
    junc_pixels = set(p for p, d in deg.items() if d > 2)
    regular_pixels = set(p for p in skel_set if p not in junc_pixels)

    visited_reg = set()
    branches = []

    for p in regular_pixels:
        if p in visited_reg:
            continue
        path = [p]
        visited_reg.add(p)
        
        curr = p
        while True:
            reg_nbrs = [n for n in get_8nbrs(curr) if n in regular_pixels and n not in visited_reg]
            if len(reg_nbrs) == 1:
                next_p = reg_nbrs[0]
                visited_reg.add(next_p)
                path.append(next_p)
                curr = next_p
            else:
                break
                
        curr = p
        while True:
            reg_nbrs = [n for n in get_8nbrs(curr) if n in regular_pixels and n not in visited_reg]
            if len(reg_nbrs) == 1:
                next_p = reg_nbrs[0]
                visited_reg.add(next_p)
                path.insert(0, next_p)
                curr = next_p
            else:
                break

        if len(path) >= 3:
            branches.append(path)

    # Tangent-matching branch merger
    merged_branches = [list(b) for b in branches]
    changed = True
    while changed:
        changed = False
        n = len(merged_branches)
        to_delete = set()
        
        for i in range(n):
            if i in to_delete: continue
            b1 = merged_branches[i]
            
            for j in range(i+1, n):
                if j in to_delete: continue
                b2 = merged_branches[j]
                
                d1 = math.hypot(b1[-1][0]-b2[0][0], b1[-1][1]-b2[0][1])
                if d1 <= 8.0:
                    t1 = get_end_tangent(b1, at_start=False)
                    t2 = get_end_tangent(b2, at_start=True)
                    if angle_deg(t1, t2) <= 45.0:
                        b1.extend(b2[1:])
                        to_delete.add(j)
                        changed = True
                        break
                        
                d2 = math.hypot(b1[-1][0]-b2[-1][0], b1[-1][1]-b2[-1][1])
                if d2 <= 8.0:
                    t1 = get_end_tangent(b1, at_start=False)
                    t2 = get_end_tangent(b2, at_start=False)
                    t2_rev = (-t2[0], -t2[1])
                    if angle_deg(t1, t2_rev) <= 45.0:
                        b1.extend(reversed(b2[:-1]))
                        to_delete.add(j)
                        changed = True
                        break
                        
                d3 = math.hypot(b1[0][0]-b2[-1][0], b1[0][1]-b2[-1][1])
                if d3 <= 8.0:
                    t1 = get_end_tangent(b1, at_start=True)
                    t1_rev = (-t1[0], -t1[1])
                    t2 = get_end_tangent(b2, at_start=False)
                    if angle_deg(t1_rev, t2) <= 45.0:
                        merged_branches[i] = list(b2[:-1]) + b1
                        to_delete.add(j)
                        changed = True
                        break
                        
                d4 = math.hypot(b1[0][0]-b2[0][0], b1[0][1]-b2[0][1])
                if d4 <= 8.0:
                    t1 = get_end_tangent(b1, at_start=True)
                    t1_rev = (-t1[0], -t1[1])
                    t2 = get_end_tangent(b2, at_start=True)
                    if angle_deg(t1_rev, t2) <= 45.0:
                        merged_branches[i] = list(reversed(b2[1:])) + b1
                        to_delete.add(j)
                        changed = True
                        break

        merged_branches = [b for idx, b in enumerate(merged_branches) if idx not in to_delete]

    return merged_branches

def extract_mouse_paths():
    img = cv2.imread(str(MOUSE_REF_PATH), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Missing {MOUSE_REF_PATH}")

    hm, wm = img.shape
    _, mouse_thresh = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

    skel_m = zhang_suen_thinning(mouse_thresh)
    skel_pts = np.argwhere(skel_m > 0)
    skel_set = set((pt[1], pt[0]) for pt in skel_pts)

    def get_8nbrs(p):
        x, y = p
        return [(x+dx, y+dy) for dx in [-1,0,1] for dy in [-1,0,1] if (dx!=0 or dy!=0) and (x+dx, y+dy) in skel_set]

    deg = {p: len(get_8nbrs(p)) for p in skel_set}
    junc_pixels = set(p for p, d in deg.items() if d > 2)
    regular_pixels = set(p for p in skel_set if p not in junc_pixels)

    visited_reg = set()
    branches = []

    for p in regular_pixels:
        if p in visited_reg:
            continue
        path = [p]
        visited_reg.add(p)
        
        curr = p
        while True:
            reg_nbrs = [n for n in get_8nbrs(curr) if n in regular_pixels and n not in visited_reg]
            if len(reg_nbrs) == 1:
                next_p = reg_nbrs[0]
                visited_reg.add(next_p)
                path.append(next_p)
                curr = next_p
            else:
                break
                
        curr = p
        while True:
            reg_nbrs = [n for n in get_8nbrs(curr) if n in regular_pixels and n not in visited_reg]
            if len(reg_nbrs) == 1:
                next_p = reg_nbrs[0]
                visited_reg.add(next_p)
                path.insert(0, next_p)
                curr = next_p
            else:
                break

        if len(path) >= 2:
            branches.append(path)

    merged = [list(b) for b in branches]
    changed = True
    while changed:
        changed = False
        n = len(merged)
        to_delete = set()
        for i in range(n):
            if i in to_delete: continue
            b1 = merged[i]
            for j in range(i+1, n):
                if j in to_delete: continue
                b2 = merged[j]
                d1 = math.hypot(b1[-1][0]-b2[0][0], b1[-1][1]-b2[0][1])
                if d1 <= 6.0:
                    t1 = get_end_tangent(b1, False)
                    t2 = get_end_tangent(b2, True)
                    if angle_deg(t1, t2) <= 50.0:
                        b1.extend(b2[1:])
                        to_delete.add(j)
                        changed = True
                        break
        merged = [b for idx, b in enumerate(merged) if idx not in to_delete]

    return merged, wm, hm

def generate():
    print("Executing Zhang-Suen 1-pixel skeletonization for Bappa...")
    bappa_branches = extract_bappa_paths()
    print(f"Extracted {len(bappa_branches)} continuous Bappa centerline paths!")

    processed_paths = []
    path_id_counter = 0

    for b in bappa_branches:
        pts_ref = [(float(p[0]), float(p[1])) for p in b]
        step_num, category = classify_bappa_path(pts_ref)
        canvas_pts = [bappa_ref_to_canvas(p[0], p[1]) for p in pts_ref]
        smoothed = smooth_points(canvas_pts, passes=2)
        bezier_segs = fit_bezier_spline(smoothed)

        width_super = 7.0 if category in ['Head Outline', 'Crown', 'Left Ear', 'Right Ear', 'Dhoti', 'Feet', 'Legs'] else 5.5
        name_str = f"{category.lower().replace(' ', '_')}_{path_id_counter}"

        processed_paths.append({
            'id': path_id_counter,
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
        path_id_counter += 1

    print("Executing Zhang-Suen 1-pixel skeletonization for Mooshak (new mouse reference)...")
    mouse_branches, wm, hm = extract_mouse_paths()
    print(f"Extracted {len(mouse_branches)} Mooshak mouse vector paths!")

    for mb in mouse_branches:
        pts_m = [(float(p[0]), float(p[1])) for p in mb]
        canvas_pts = [mouse_ref_to_canvas(p[0], p[1], wm, hm) for p in pts_m]
        smoothed = smooth_points(canvas_pts, passes=2)
        bezier_segs = fit_bezier_spline(smoothed)

        name_str = f"mooshak_mouse_{path_id_counter}"
        processed_paths.append({
            'id': path_id_counter,
            'name': name_str,
            'category': 'Mooshak (Mouse)',
            'step_order': 25,
            'bezier_segments': bezier_segs,
            'points': smoothed,
            'width': 5.0,
            'is_closed': False,
            'is_filled': False,
            'fill_color': None
        })
        path_id_counter += 1

    processed_paths.sort(key=lambda p: (p['step_order'], p['id']))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(processed_paths, f, indent=2)

    print(f"Successfully generated {len(processed_paths)} perfect continuous vector paths -> {OUT_JSON}")

if __name__ == "__main__":
    generate()
