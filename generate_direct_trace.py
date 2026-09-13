"""
generate_direct_trace.py

THE CORRECT APPROACH:
1:1 pixel-accurate trace of the reference image using direct cv2.findContours.
NO skeleton. NO branch tracing. NO path merging. NO Douglas-Peucker.
Every contour from the reference is preserved exactly as found.

Ganapati contours are scaled from the reference image space to the 2700x3300 supersampled canvas.
8% padding is applied so the full drawing fits on the 900x1100 screen.
Mouse contours are independently scaled and placed beside Bappa's feet with a clear white gap.

Eye contours are correctly detected by bbox in reference image pixel space:
  Right eye outer: contour [6]  ctr=(278,191) bbox=[261,296,171,212]
  Right eye pupil: contour [8]  ctr=(281,190) bbox=[271,290,176,209]  <- filled black on render
  Left eye outer:  contour [10] ctr=(199,166) bbox=[190,211,151,182]
  Left eye inner:  contour [13] ctr=(203,165) bbox=[196,209,154,179]  <- filled black on render
"""

import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
BAPPA_REF_PATH = ROOT / "assets" / "reference_image.png"
MOUSE_REF_PATH = ROOT / "assets" / "new_mouse_reference.png"
OUT_JSON = ROOT / "ganapati_python" / "vector_data.json"

# Supersampled canvas size
CANVAS_W = 2700
CANVAS_H = 3300

def classify_bappa_path(min_x, max_x, min_y, max_y, ref_w, ref_h):
    """
    Classifies each Bappa contour into anatomical step order.
    Uses EXACT pixel-space coordinates from reference image (467x517).
    Key eye contours identified by inspection:
      Right eye outer [6]:  bbox=[261,296,171,212] area=1002
      Right eye pupil [8]:  bbox=[271,290,176,209] area=305 (inner, will be filled black)
      Left eye outer [10]:  bbox=[190,211,151,182] area=504
      Left eye inner [13]:  bbox=[196,209,154,179] area=136 (inner, will be filled black)
    """
    avg_x = (min_x + max_x) / 2.0
    avg_y = (min_y + max_y) / 2.0
    bw = max_x - min_x
    bh = max_y - min_y

    # ---- EYES — classified by exact bbox + parent/child hierarchy ----
    #
    # RIGHT EYE structure (from RETR_CCOMP hierarchy inspection):
    #   [6] outer black filled ring  bbox=[261,296,171,212] area=1002  parent=-1
    #   [8] inner white eye-white    bbox=[271,290,176,209] area=305   parent=6   <- fill WHITE
    #   [7] tiny pupil dot           bbox=[274,279,189,196] area=25.5  parent=6   <- fill BLACK
    #
    # LEFT EYE structure:
    #   [10] outer black filled ring  bbox=[190,211,151,182] area=504  parent=-1
    #   [13] inner white eye-white    bbox=[196,209,154,179] area=136  parent=10  <- fill WHITE
    #   [12] small pupil detail       bbox=[198,201,164,170] area=13   parent=10  <- fill BLACK
    #
    # Rule: outer ring (large area, parent=-1) -> fill BLACK
    #       inner child (smaller area) -> fill WHITE (eye-white)
    #       tiny child inside inner (very small) -> fill BLACK (pupil dot)

    # Right eye outer ring — contour [6]: bbox=[261,296,171,212] fill BLACK
    if 258 <= min_x <= 265 and 292 <= max_x <= 300 and 168 <= min_y <= 175 and 208 <= max_y <= 216:
        return (7, 'Right Eye Black')
    # Right eye white area — contour [8]: bbox=[271,290,176,209] fill WHITE
    if 268 <= min_x <= 275 and 287 <= max_x <= 294 and 173 <= min_y <= 180 and 206 <= max_y <= 213:
        return (7, 'Right Eye White')
    # Right eye white highlight dot inside black pupil — contour [7]: bbox=[274,279,189,196] fill WHITE
    if 271 <= min_x <= 278 and 276 <= max_x <= 283 and 186 <= min_y <= 193 and 193 <= max_y <= 200:
        return (7, 'Right Eye Highlight')
    # Left eye outer ring — contour [10]: bbox=[190,211,151,182] fill BLACK
    if 188 <= min_x <= 193 and 208 <= max_x <= 215 and 148 <= min_y <= 155 and 179 <= max_y <= 186:
        return (6, 'Left Eye Black')
    # Left eye white area — contour [13]: bbox=[196,209,154,179] fill WHITE
    if 193 <= min_x <= 199 and 206 <= max_x <= 213 and 151 <= min_y <= 158 and 176 <= max_y <= 183:
        return (6, 'Left Eye White')
    # Left eye white highlight dot inside black pupil — contour [12]: bbox=[198,201,164,170] fill WHITE
    if 195 <= min_x <= 201 and 198 <= max_x <= 205 and 161 <= min_y <= 167 and 167 <= max_y <= 173:
        return (6, 'Left Eye Highlight')

    # ---- EYEBROWS (solid black filled) ----
    # Left eyebrow contour [15]: bbox=[202,216,120,129] -> fill BLACK
    if 198 <= min_x <= 206 and 212 <= max_x <= 220 and 117 <= min_y <= 124 and 126 <= max_y <= 133:
        return (5, 'Left Eyebrow')
    # Right eyebrow contour [14]: bbox=[298,310,144,165] -> fill BLACK
    if 294 <= min_x <= 302 and 306 <= max_x <= 314 and 140 <= min_y <= 148 and 161 <= max_y <= 169:
        return (5, 'Right Eyebrow')

    # ---- FOREHEAD TILAK SHAPES ----
    # Tilak U-shape (contour [16]: bbox=[239,274,110,150]) -> fill YELLOW
    if 235 <= min_x <= 245 and 268 <= max_x <= 278 and 106 <= min_y <= 114 and 146 <= max_y <= 154:
        return (24, 'Tilak U-Shape')
    # Tilak Water Drop (contour [9]: bbox=[236,242,152,163]) -> fill RED
    if 232 <= min_x <= 240 and 239 <= max_x <= 246 and 149 <= min_y <= 155 and 160 <= max_y <= 167:
        return (24, 'Tilak Water Drop')

    # Crown (topmost area)
    if min_y < ref_h * 0.20 and avg_x > ref_w * 0.35 and avg_x < ref_w * 0.85:
        return (1, 'Crown')
    # Left Ear (left side)
    if max_x < ref_w * 0.46 and min_y > ref_h * 0.13 and max_y < ref_h * 0.50 and avg_x < ref_w * 0.39:
        return (3, 'Left Ear')
    # Right Ear (right side, large)
    if min_x > ref_w * 0.56 and min_y > ref_h * 0.24 and max_y < ref_h * 0.62:
        return (4, 'Right Ear')
    # Crown/Forehead decoration band
    if ref_h * 0.20 <= min_y and max_y < ref_h * 0.36 and ref_w * 0.38 <= avg_x <= ref_w * 0.70:
        return (5, 'Forehead')
    # Head outline (large central head mass)
    if ref_w * 0.27 <= min_x and max_x <= ref_w * 0.76 and ref_h * 0.21 <= min_y <= ref_h * 0.52 and avg_y < ref_h * 0.55:
        return (2, 'Head Outline')
    # Left Arm (left side extended arm)
    if max_x < ref_w * 0.40 and min_y > ref_h * 0.38 and max_y < ref_h * 0.62 and min_x < ref_w * 0.30:
        return (10, 'Left Arm')
    # Left Hand (far left)
    if max_x < ref_w * 0.38 and min_y > ref_h * 0.35 and max_y < ref_h * 0.62 and min_x < ref_w * 0.22:
        return (11, 'Left Hand')
    # Right Arm
    if min_x > ref_w * 0.57 and min_y > ref_h * 0.50 and max_y < ref_h * 0.73:
        return (12, 'Right Arm')
    # Right Hand
    if min_x > ref_w * 0.62 and min_y > ref_h * 0.48 and max_y < ref_h * 0.73:
        return (13, 'Right Hand')
    # Trunk
    if ref_w * 0.33 <= min_x and max_x <= ref_w * 0.60 and ref_h * 0.37 <= min_y and max_y <= ref_h * 0.68:
        return (9, 'Trunk')
    # Necklace
    if ref_w * 0.30 <= min_x and max_x <= ref_w * 0.60 and ref_h * 0.45 <= min_y <= ref_h * 0.68 and avg_y < ref_h * 0.65:
        return (14, 'Necklace')
    # Waist Knot and hanging cloth
    if ref_w * 0.27 <= min_x and max_x <= ref_w * 0.47 and ref_h * 0.62 <= min_y and max_y <= ref_h * 0.84:
        if avg_y < ref_h * 0.73:
            return (17, 'Waist Knot')
        return (18, 'Hanging Cloth')
    # Torso and upper clothing
    if ref_w * 0.26 <= min_x and max_x <= ref_w * 0.64 and ref_h * 0.46 <= min_y and max_y <= ref_h * 0.72:
        return (15, 'Torso/Upper Clothing')
    # Feet (bottom-most area)
    if max_y > ref_h * 0.90:
        return (23, 'Feet')
    # Legs
    if min_y > ref_h * 0.84 and max_y <= ref_h * 0.94:
        return (21, 'Legs')
    # Dhoti area
    if min_y > ref_h * 0.66 and max_y <= ref_h * 0.93:
        if bw < ref_w * 0.12 and avg_y > ref_h * 0.75:
            return (20, 'Dhoti Folds')
        return (19, 'Dhoti')

    return (24, 'Small Details')

def contour_to_smooth_points(contour, scale_x, scale_y, smooth_passes=1):
    """Scales contour points to canvas space with light smoothing."""
    pts = [(float(pt[0][0]) * scale_x, float(pt[0][1]) * scale_y) for pt in contour]
    if smooth_passes > 0 and len(pts) > 3:
        arr = np.array(pts, dtype=float)
        kernel = np.array([0.15, 0.7, 0.15])
        for _ in range(smooth_passes):
            smoothed = np.zeros_like(arr)
            smoothed[:, 0] = np.convolve(arr[:, 0], kernel, mode='same')
            smoothed[:, 1] = np.convolve(arr[:, 1], kernel, mode='same')
            smoothed[0] = arr[0]
            smoothed[-1] = arr[-1]
            arr = smoothed
        pts = [(round(p[0], 2), round(p[1], 2)) for p in arr]
    return pts

def generate():
    # ===========================================================
    # 1. GANAPATI (BAPPA) - Direct contour trace, no skeleton
    # ===========================================================
    bappa_img = cv2.imread(str(BAPPA_REF_PATH), cv2.IMREAD_GRAYSCALE)
    if bappa_img is None:
        raise FileNotFoundError(f"Missing {BAPPA_REF_PATH}")

    ref_h, ref_w = bappa_img.shape
    print(f"Bappa reference image: {ref_w}x{ref_h}")

    _, bappa_thresh = cv2.threshold(bappa_img, 200, 255, cv2.THRESH_BINARY_INV)
    contours_bappa, _ = cv2.findContours(bappa_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    print(f"Raw Bappa RETR_CCOMP contours: {len(contours_bappa)}")

    # Scale: reference image pixels -> 2700x3300 supersampled canvas
    # Apply 13% padding so the whole drawing has clear margins at all sides
    PADDING = 0.13
    bappa_scale_x = (CANVAS_W * (1.0 - 2 * PADDING)) / ref_w
    bappa_scale_y = (CANVAS_H * (1.0 - 2 * PADDING)) / ref_h
    bappa_offset_x = CANVAS_W * PADDING
    bappa_offset_y = CANVAS_H * PADDING

    processed_paths = []
    path_id_counter = 0

    for c in contours_bappa:
        arc = cv2.arcLength(c, True)
        if arc < 8:
            continue  # Remove only true noise (< 8px arc length)

        pts_ref = c[:, 0, :]
        min_x, max_x = int(pts_ref[:, 0].min()), int(pts_ref[:, 0].max())
        min_y, max_y = int(pts_ref[:, 1].min()), int(pts_ref[:, 1].max())
        bw = max_x - min_x
        bh = max_y - min_y

        # Discard entire outer image border frame
        if bw > ref_w * 0.87 and bh > ref_h * 0.67:
            continue
        # Discard thin border edge lines
        if max_y < 12 or max_x < 20 or min_x > ref_w - 20:
            continue

        step_num, category = classify_bappa_path(min_x, max_x, min_y, max_y, ref_w, ref_h)

        # Scale contour to supersampled canvas space with 1 pass of light smoothing
        canvas_pts = contour_to_smooth_points(c, bappa_scale_x, bappa_scale_y, smooth_passes=1)
        # Apply padding offset
        canvas_pts = [(round(p[0] + bappa_offset_x, 2), round(p[1] + bappa_offset_y, 2)) for p in canvas_pts]

        # Eye, Eyebrow & Tilak layer fill logic:
        #   Black filled: eye outer ring + eyebrows
        #   White filled: inner eye-white region + pupil highlight catchlight dots
        #   Yellow filled: Tilak U-shape
        #   Red filled: Tilak Water Drop
        BLACK_FILL_CATS = ('Right Eye Black', 'Left Eye Black', 'Left Eyebrow', 'Right Eyebrow')
        WHITE_FILL_CATS = ('Right Eye White', 'Left Eye White', 'Right Eye Highlight', 'Left Eye Highlight')
        YELLOW_FILL_CATS = ('Tilak U-Shape',)
        RED_FILL_CATS = ('Tilak Water Drop',)

        if category in BLACK_FILL_CATS:
            is_filled = True
            fill_color = [20, 20, 20]
        elif category in WHITE_FILL_CATS:
            is_filled = True
            fill_color = [250, 249, 245]   # match background — creates white catchlight dots & eye whites
        elif category in YELLOW_FILL_CATS:
            is_filled = True
            fill_color = [255, 204, 0]     # Saffron Yellow fill for U-shape
        elif category in RED_FILL_CATS:
            is_filled = True
            fill_color = [220, 20, 20]     # Kumkum Red fill for Water Drop
        else:
            is_filled = False
            fill_color = None

        # Line width: outer body lines get 7.5, internal details get 5.5
        width_super = 7.5 if category in [
            'Head Outline', 'Crown', 'Left Ear', 'Right Ear', 'Dhoti', 'Feet', 'Legs'
        ] else 5.5

        processed_paths.append({
            'id': path_id_counter,
            'name': f"{category.lower().replace(' ', '_')}_{path_id_counter}",
            'category': category,
            'step_order': step_num,
            'points': canvas_pts,
            'bezier_segments': [],
            'width': width_super,
            'is_closed': True,
            'is_filled': is_filled,
            'fill_color': fill_color
        })
        path_id_counter += 1

    print(f"Extracted {len(processed_paths)} exact Bappa reference contours!")

    # ===========================================================
    # 2. MOOSHAK MOUSE - Direct contour trace, independent group
    #    Placed beside Bappa's right foot with clear white space gap.
    # ===========================================================
    mouse_img = cv2.imread(str(MOUSE_REF_PATH), cv2.IMREAD_GRAYSCALE)
    if mouse_img is not None:
        mh, mw = mouse_img.shape
        print(f"Mooshak mouse reference image: {mw}x{mh}")

        _, mouse_thresh = cv2.threshold(mouse_img, 220, 255, cv2.THRESH_BINARY_INV)
        contours_mouse, _ = cv2.findContours(mouse_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        print(f"Raw Mooshak RETR_CCOMP contours: {len(contours_mouse)}")

        # Mouse target position on supersampled canvas:
        # CY=2640 gives ~880px on 900x1100 screen (max_y ~ 940px), ensuring a clear bottom margin below the mouse equal to Ganapati.
        MOUSE_TARGET_CX = 2080   # target center x on 3x canvas
        MOUSE_TARGET_CY = 2640   # moved up to 2640 -> bottom margin below mouse matches Bappa's bottom margin
        MOUSE_TARGET_W  = 330    # target width on 3x canvas (110px on screen)
        MOUSE_TARGET_H  = 360    # target height on 3x canvas

        mouse_scale_x = MOUSE_TARGET_W / mw
        mouse_scale_y = MOUSE_TARGET_H / mh
        mouse_off_x = MOUSE_TARGET_CX - (mw / 2.0) * mouse_scale_x
        mouse_off_y = MOUSE_TARGET_CY - (mh / 2.0) * mouse_scale_y

        for c in contours_mouse:
            arc = cv2.arcLength(c, True)
            if arc < 6:
                continue

            pts_ref = c[:, 0, :]
            min_x, max_x = int(pts_ref[:, 0].min()), int(pts_ref[:, 0].max())
            min_y, max_y = int(pts_ref[:, 1].min()), int(pts_ref[:, 1].max())
            bw = max_x - min_x
            bh = max_y - min_y

            # Discard outer border of mouse image
            if bw > mw * 0.95 and bh > mh * 0.95:
                continue

            canvas_pts = []
            for pt in c:
                cx = round(pt[0][0] * mouse_scale_x + mouse_off_x, 2)
                cy = round(pt[0][1] * mouse_scale_y + mouse_off_y, 2)
                canvas_pts.append((cx, cy))

            # Light smooth
            if len(canvas_pts) > 3:
                arr = np.array(canvas_pts, dtype=float)
                kernel = np.array([0.15, 0.7, 0.15])
                smoothed = np.zeros_like(arr)
                smoothed[:, 0] = np.convolve(arr[:, 0], kernel, mode='same')
                smoothed[:, 1] = np.convolve(arr[:, 1], kernel, mode='same')
                smoothed[0] = arr[0]
                smoothed[-1] = arr[-1]
                canvas_pts = [(round(p[0], 2), round(p[1], 2)) for p in smoothed]

            processed_paths.append({
                'id': path_id_counter,
                'name': f"mooshak_mouse_{path_id_counter}",
                'category': 'Mooshak (Mouse)',
                'step_order': 25,
                'points': canvas_pts,
                'bezier_segments': [],
                'width': 5.0,
                'is_closed': True,
                'is_filled': False,
                'fill_color': None
            })
            path_id_counter += 1

        print(f"Extracted {len([p for p in processed_paths if p['category'] == 'Mooshak (Mouse)'])} Mooshak contours!")

    # Sort by anatomical drawing order
    processed_paths.sort(key=lambda p: (p['step_order'], p['id']))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(processed_paths, f, indent=2)

    total_bappa = len([p for p in processed_paths if p['category'] != 'Mooshak (Mouse)'])
    total_mouse = len([p for p in processed_paths if p['category'] == 'Mooshak (Mouse)'])
    print(f"\nGenerated {len(processed_paths)} total vector paths:")
    print(f"  Bappa:   {total_bappa} exact-trace contours")
    print(f"  Mooshak: {total_mouse} exact-trace contours")
    print(f"  -> {OUT_JSON}")

if __name__ == "__main__":
    generate()
