"""
config.py

Configuration parameters for Ganapati Vector Drawing Pygame Application.
High-quality 3x Supersampling (2700x3300 -> 900x1100).
Accelerated smooth animation speed.
"""

# Canvas Dimensions
LOGICAL_WIDTH = 900
LOGICAL_HEIGHT = 1100
SUPERSAMPLE_SCALE = 3
SUPERSAMPLE_WIDTH = LOGICAL_WIDTH * SUPERSAMPLE_SCALE   # 2700
SUPERSAMPLE_HEIGHT = LOGICAL_HEIGHT * SUPERSAMPLE_SCALE # 3300

# Color Palette (RGB)
BACKGROUND_COLOR = (250, 249, 245)  # Off-white warm background matching reference
LINE_COLOR = (20, 20, 20)           # Deep ink black line art
WHITE = (255, 255, 255)             # Eye highlight white
POINTER_COLOR = (190, 35, 35)       # Pen tip accent red
POINTER_ACCENT = (40, 40, 40)       # Pen body dark charcoal
HUD_TEXT_COLOR = (60, 60, 60)       # UI HUD text color
HUD_BG_COLOR = (240, 238, 230, 200) # Subtle HUD panel background

# Animation Controls (Faster & Smoother)
DRAW_SPEED = 1.8                    # Default draw speed multiplier (faster default pace)
BASE_POINTS_PER_FRAME = 7.0         # Accelerated point progression per frame
PEN_TRANSITION_SPEED = 90.0         # Fast pen movement between stroke paths (at 2700x3300)
FPS = 60                            # Target 60 FPS
TOTAL_DURATION_SEC = 8.0            # ~8.0 seconds total completion time

# Drawing Order Sequence (25 anatomical steps including Mooshak mouse)
DRAWING_SEQUENCE = [
    "Crown",
    "Head Outline",
    "Left Ear",
    "Right Ear",
    "Face",
    "Forehead",
    "Left Eye",
    "Right Eye",
    "Trunk",
    "Left Arm",
    "Left Hand",
    "Right Arm",
    "Right Hand",
    "Necklace",
    "Torso/Upper Clothing",
    "Waist",
    "Waist Knot",
    "Hanging Cloth",
    "Dhoti",
    "Dhoti Folds",
    "Legs",
    "Right Leg",
    "Feet",
    "Small Details",
    "Mooshak (Mouse)"
]
