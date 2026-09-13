"""
paths.py

Path models and vector path loader for Ganapati line art.
Contains classes DrawingPath, EyeDetail, BeadDetail, and get_ganapati_paths().
Supports 3x Supersampling canvas (2700x3300).
"""

import json
import math
from pathlib import Path

class DrawingPath:
    """
    Represents a vector path composed of cubic Bezier curve control points
    and high-density sampled resolution points.
    """
    def __init__(self, path_id, name, category, step_order, bezier_segments, points, width=6.0, is_closed=False, is_filled=False, fill_color=None):
        self.path_id = path_id
        self.name = name
        self.category = category
        self.step_order = step_order
        self.bezier_segments = bezier_segments  # List of cubic Bezier control quads [(P0, P1, P2, P3), ...]
        self.points = [(float(pt[0]), float(pt[1])) for pt in points]  # Evaluated canvas points
        self.width = float(width)
        self.is_closed = is_closed
        self.is_filled = is_filled
        self.fill_color = fill_color

    def get_length(self):
        """Calculates total arc length of evaluated points."""
        length = 0.0
        for i in range(len(self.points) - 1):
            dx = self.points[i+1][0] - self.points[i][0]
            dy = self.points[i+1][1] - self.points[i][1]
            length += math.hypot(dx, dy)
        return max(1.0, length)


class EyeDetail:
    """
    High-priority eye vector structures: dark pupils and white highlights.
    Precise 3x Supersampled Coordinates (2700x3300).
    """
    def __init__(self):
        # Left Eye Pupil & Highlight
        self.left_pupil_center = (1141.0, 1129.0)
        self.left_pupil_radius = 33.0
        self.left_highlight_center = (1129.0, 1113.0)
        self.left_highlight_radius = 9.0

        # Right Eye Pupil & Highlight
        self.right_pupil_center = (1575.0, 1227.0)
        self.right_pupil_radius = 42.0
        self.right_highlight_center = (1558.0, 1210.0)
        self.right_highlight_radius = 12.0


class BeadDetail:
    """
    Necklace bead circles following natural necklace curve.
    Precise 3x Supersampled Coordinates (2700x3300).
    """
    def __init__(self):
        # 14 Bead centers along necklace loop
        self.bead_centers = [
            (935.0, 1550.0), (982.0, 1610.0), (1035.0, 1665.0),
            (1095.0, 1710.0), (1162.0, 1748.0), (1237.0, 1770.0),
            (1312.0, 1782.0), (1387.0, 1782.0), (1462.0, 1762.0),
            (1530.0, 1725.0), (1590.0, 1672.0), (1642.0, 1612.0),
            (1687.0, 1552.0), (1725.0, 1492.0)
        ]
        self.bead_radius = 15.0

        # Central Pendant Beads
        self.pendant_center_top = (1350.0, 1808.0)
        self.pendant_radius_top = 21.0
        self.pendant_center_bottom = (1350.0, 1852.0)
        self.pendant_radius_bottom = 27.0


def get_ganapati_paths():
    """
    Loads all smooth vector paths extracted from reference image analysis,
    ordered by 24 anatomical drawing steps.
    """
    json_path = Path(__file__).resolve().parent / "vector_data.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing vector path data file: {json_path}")

    with open(json_path, "r") as f:
        raw_data = json.load(f)

    paths = []
    for item in raw_data:
        dp = DrawingPath(
            path_id=item['id'],
            name=item['name'],
            category=item['category'],
            step_order=item['step_order'],
            bezier_segments=item['bezier_segments'],
            points=item['points'],
            width=item['width'],
            is_closed=item.get('is_closed', False),
            is_filled=item.get('is_filled', False),
            fill_color=item.get('fill_color', None)
        )
        # Only add valid multi-point paths (strictly filter out noise micro-dots)
        if len(dp.points) >= 4:
            paths.append(dp)

    # Sort paths by step order, then by vertical min_y for natural top-to-bottom stroke flow
    paths.sort(key=lambda p: (p.step_order, min(pt[1] for pt in p.points) if p.points else 0))

    return paths, EyeDetail(), BeadDetail()
