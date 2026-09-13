"""
animation.py

Animation controller for Ganapati Pygame Application.
Manages progressive stroke drawing, pen pointer motion, speed control, restart, and reset.
Supports 3x Supersampled Coordinates (2700x3300).
"""

import math
from config import DRAW_SPEED, BASE_POINTS_PER_FRAME, PEN_TRANSITION_SPEED

class DrawingAnimator:
    def __init__(self, paths):
        self.paths = paths
        self.draw_speed = DRAW_SPEED
        self.current_path_idx = 0
        self.current_path_progress = 0.0  # 0.0 to 1.0
        
        # Pen state (2700x3300 space)
        self.pointer_pos = (1350.0, 1650.0)
        self.is_transitioning = False
        self.transition_start = (1350.0, 1650.0)
        self.transition_target = (1350.0, 1650.0)
        self.transition_progress = 1.0
        self.is_completed = False

        if self.paths and self.paths[0].points:
            self.pointer_pos = self.paths[0].points[0]

    def restart(self):
        """Restarts the animation from the beginning."""
        self.current_path_idx = 0
        self.current_path_progress = 0.0
        self.is_transitioning = False
        self.is_completed = False
        if self.paths and self.paths[0].points:
            self.pointer_pos = self.paths[0].points[0]

    def reset(self):
        """Resets drawing state."""
        self.restart()

    def set_speed(self, speed):
        """Sets drawing speed multiplier."""
        self.draw_speed = max(0.1, min(10.0, float(speed)))

    def increase_speed(self):
        """Increases drawing speed."""
        self.set_speed(self.draw_speed + 0.25)

    def decrease_speed(self):
        """Decreases drawing speed."""
        self.set_speed(self.draw_speed - 0.25)

    def update(self, dt):
        """
        Advances animation state by time step dt.
        """
        if self.is_completed or not self.paths:
            return

        if self.is_transitioning:
            tx = self.transition_target[0] - self.pointer_pos[0]
            ty = self.transition_target[1] - self.pointer_pos[1]
            dist = math.hypot(tx, ty)

            step = PEN_TRANSITION_SPEED * self.draw_speed * (dt * 60.0)
            if dist <= step or dist < 1.0:
                self.pointer_pos = self.transition_target
                self.is_transitioning = False
                self.current_path_progress = 0.0
            else:
                self.pointer_pos = (
                    self.pointer_pos[0] + (tx / dist) * step,
                    self.pointer_pos[1] + (ty / dist) * step
                )
            return

        path = self.paths[self.current_path_idx]
        n_pts = len(path.points)
        if n_pts < 2:
            self._advance_to_next_path()
            return

        pts_per_frame = (BASE_POINTS_PER_FRAME * self.draw_speed) / float(n_pts)
        self.current_path_progress += pts_per_frame * (dt * 60.0)

        if self.current_path_progress >= 1.0:
            self.current_path_progress = 1.0
            self.pointer_pos = path.points[-1]
            self._advance_to_next_path()
        else:
            pt_idx = int(round((n_pts - 1) * self.current_path_progress))
            pt_idx = max(0, min(n_pts - 1, pt_idx))
            self.pointer_pos = path.points[pt_idx]

    def _advance_to_next_path(self):
        """Triggers transition to next path or marks completion."""
        if self.current_path_idx + 1 < len(self.paths):
            self.current_path_idx += 1
            next_start = self.paths[self.current_path_idx].points[0]
            self.transition_start = self.pointer_pos
            self.transition_target = next_start
            self.is_transitioning = True
        else:
            self.is_completed = True

    def get_active_step_info(self):
        """Returns current step category and progress percentage."""
        if self.is_completed:
            return "Complete", 100
        if self.current_path_idx < len(self.paths):
            cat = self.paths[self.current_path_idx].category
            pct = int((self.current_path_idx / float(len(self.paths))) * 100)
            return cat, pct
        return "Drawing", 0
