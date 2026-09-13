"""
drawing.py

Exact-trace vector rendering engine for Ganapati Pygame Application.
Renders closed contour loops (is_closed=True) or open stroke paths (is_closed=False).
Eye pupils are rendered as filled polygons traced from the reference — no hardcoded circles.
"""

import pygame
from config import (
    LINE_COLOR, WHITE, POINTER_COLOR, POINTER_ACCENT, BACKGROUND_COLOR
)


def draw_path(surface, path, progress=1.0, color=LINE_COLOR, width_scale=1.0):
    """
    Progressively draws a DrawingPath up to progress fraction (0.0 to 1.0).
    Handles closed contour loops, open strokes, and filled pupil polygons.
    """
    if not path.points or progress <= 0.0 or len(path.points) < 2:
        return

    n_pts = len(path.points)
    if progress >= 1.0:
        visible_pts = path.points
    else:
        count = max(2, int(round(n_pts * progress)))
        visible_pts = path.points[:count]

    int_pts = [(int(round(pt[0])), int(round(pt[1]))) for pt in visible_pts]
    if len(int_pts) < 2:
        return

    # Filled polygon — eye pupils, catchlights, and tilak yellow/red fills
    if path.is_filled and path.fill_color and progress >= 1.0 and len(int_pts) >= 3:
        fc = tuple(path.fill_color) if isinstance(path.fill_color, list) else path.fill_color
        pygame.draw.polygon(surface, fc, int_pts)
        w_int = max(1, int(round(path.width * width_scale)))
        pygame.draw.lines(surface, color, True, int_pts, w_int)
        return

    w_int = max(1, int(round(path.width * width_scale)))
    # Close the loop only when fully drawn — prevents chord line during animation
    closed = path.is_closed and progress >= 1.0
    pygame.draw.lines(surface, color, closed, int_pts, w_int)


def draw_ganapati(surface, paths, active_path_idx, active_path_progress):
    """
    Renders all completed paths progressively.
    Eye pupils are embedded as filled contours in vector_data.json — no separate hardcoded positions.
    """
    surface.fill(BACKGROUND_COLOR)

    for idx in range(min(len(paths), active_path_idx)):
        p = paths[idx]
        draw_path(surface, p, progress=1.0)

    if active_path_idx < len(paths):
        p = paths[active_path_idx]
        draw_path(surface, p, progress=active_path_progress)


def draw_pointer(surface, pos):
    """Renders the pen tip cursor at pos."""
    px, py = int(round(pos[0])), int(round(pos[1]))

    glow_surf = pygame.Surface((48, 48), pygame.SRCALPHA)
    pygame.draw.circle(glow_surf, (220, 50, 50, 90), (24, 24), 21)
    surface.blit(glow_surf, (px - 24, py - 24))

    pygame.draw.circle(surface, POINTER_ACCENT, (px, py), 8, 2)
    pygame.draw.circle(surface, POINTER_COLOR, (px, py), 5)
