"""
drawing.py

Clean Vector drawing engine for Ganapati Pygame Application.
Guarantees NO connecting lines back to path start while sketching.
Renders ONLY the active line up to current pointer position.
"""

import math
import pygame
from config import (
    LINE_COLOR, WHITE, POINTER_COLOR, POINTER_ACCENT,
    BACKGROUND_COLOR, SUPERSAMPLE_SCALE
)

def evaluate_cubic_bezier(p0, p1, p2, p3, t):
    """Evaluates a single point on a cubic Bezier curve for parameter t in [0, 1]."""
    u = 1.0 - t
    u2 = u * u
    u3 = u2 * u
    t2 = t * t
    t3 = t2 * t

    x = u3 * p0[0] + 3 * u2 * t * p1[0] + 3 * u * t2 * p2[0] + t3 * p3[0]
    y = u3 * p0[1] + 3 * u2 * t * p1[1] + 3 * u * t2 * p2[1] + t3 * p3[1]
    return (x, y)


def draw_bezier(surface, p0, p1, p2, p3, color=LINE_COLOR, width=6.0, n_steps=100):
    """
    Renders a smooth cubic Bezier curve defined by control points p0, p1, p2, p3.
    """
    pts = [evaluate_cubic_bezier(p0, p1, p2, p3, i / float(n_steps)) for i in range(n_steps + 1)]
    draw_curve(surface, pts, color=color, width=width, is_closed=False)


def draw_line(surface, start_pos, end_pos, color=LINE_COLOR, width=6.0):
    """
    Draws a smooth line segment between start_pos and end_pos.
    """
    w_int = max(1, int(round(width)))
    sp = (int(round(start_pos[0])), int(round(start_pos[1])))
    ep = (int(round(end_pos[0])), int(round(end_pos[1])))
    pygame.draw.line(surface, color, sp, ep, w_int)


def draw_curve(surface, points, color=LINE_COLOR, width=6.0, is_closed=False):
    """
    Draws an open vector line connecting polyline points.
    is_closed is strictly False to prevent any line drawing from pen tip back to start.
    """
    if len(points) < 2:
        return

    w_int = max(1, int(round(width)))
    int_pts = [(int(round(pt[0])), int(round(pt[1]))) for pt in points]
    
    if len(int_pts) >= 2:
        # Always pass closed=False to prevent Pygame from drawing a chord line back to the start point!
        pygame.draw.lines(surface, color, False, int_pts, w_int)


def draw_path(surface, path, progress=1.0, color=LINE_COLOR, width_scale=1.0):
    """
    Progressively draws a DrawingPath up to progress fraction (0.0 to 1.0).
    Only renders the drawn portion from start to current pointer position.
    """
    if not path.points or progress <= 0.0 or len(path.points) < 2:
        return

    n_pts = len(path.points)
    if progress >= 1.0:
        visible_pts = path.points
    else:
        count = max(2, int(round(n_pts * progress)))
        visible_pts = path.points[:count]

    # Always render open line segment (is_closed=False) so no connecting chord line exists while sketching!
    draw_curve(surface, visible_pts, color=color, width=path.width * width_scale, is_closed=False)


def draw_ganapati(surface, paths, active_path_idx, active_path_progress):
    """
    Renders all completed paths and the currently active path up to active_path_progress.
    """
    surface.fill(BACKGROUND_COLOR)

    # Render completed paths
    for idx in range(min(len(paths), active_path_idx)):
        p = paths[idx]
        draw_path(surface, p, progress=1.0)

    # Render active path progressively
    if active_path_idx < len(paths):
        p = paths[active_path_idx]
        draw_path(surface, p, progress=active_path_progress)


def draw_pointer(surface, pos):
    """
    Renders ONLY the pen tip point / cursor at pos (current sketching location).
    No extra lines or body bars attached to distant points.
    """
    px, py = int(round(pos[0])), int(round(pos[1]))
    
    # Soft glow aura around current pen tip
    glow_surf = pygame.Surface((48, 48), pygame.SRCALPHA)
    pygame.draw.circle(glow_surf, (220, 50, 50, 90), (24, 24), 21)
    surface.blit(glow_surf, (px - 24, py - 24))

    # Clean Pen tip point cursor
    pygame.draw.circle(surface, POINTER_ACCENT, (px, py), 8, 2)
    pygame.draw.circle(surface, POINTER_COLOR, (px, py), 5)
