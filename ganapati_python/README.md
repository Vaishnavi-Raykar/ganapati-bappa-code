# Ganapati Vector Sketch Application (Python + Pygame)

A high-precision Python + Pygame application that reconstructs the reference Ganapati line art as clean, animated vector paths using cubic Bézier curves and 2x supersampling.

---

## Features

- **Reference-Accurate Line Art**: Recreates the exact pose, proportions, crown, ears, facial expression, eyes, trunk, necklace beads, hands, dhoti folds, and feet from the reference line art.
- **Dynamic Drawing Animation**: Simulates a pen drawing Ganapati step-by-step across 24 anatomical stages.
- **Smooth Bézier Curve Rendering**: Uses cubic Bézier curves and adaptive path sampling for smooth vector curves.
- **Supersampled Canvas**: Renders internally on an **1800 x 2200** canvas and downsamples using `pygame.transform.smoothscale` to **900 x 1100** for anti-aliased line rendering.
- **Interactive Controls**:
  - `SPACE`: Restart animation from step 1
  - `R`: Reset drawing state
  - `+` / `=`: Increase drawing speed
  - `-`: Decrease drawing speed
  - `ESC`: Exit application

---

## Project Structure

```text
ganapati_python/
├── main.py              # Main Pygame application loop & HUD rendering
├── drawing.py           # Bezier & vector drawing helper functions
├── paths.py             # Vector path models & data loader
├── animation.py         # Animation state controller & speed management
├── config.py            # Canvas dimensions, colors, and speed settings
├── vector_data.json     # Extracted Bezier control points and coordinates
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

---

## Installation & Running

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Application**:
   ```bash
   python main.py
   ```

---

## Drawing Sequence

The application draws Ganapati in 24 anatomical stages:
1. Crown & Ornaments
2. Head Outline & Face
3. Left Ear
4. Right Ear
5. Facial contours
6. Forehead Tilak & Markings
7. Left Eye (Almond contour, pupil & highlight)
8. Right Eye (Almond contour, pupil & highlight)
9. Tapering Curved Trunk
10. Left Arm
11. Left Hand & Fingers
12. Right Arm
13. Right Hand & Fingers
14. Bead Necklace & Pendant
15. Torso & Upper Garment
16. Waist Area
17. Central Waist Knot
18. Hanging Pleated Fabric
19. Dhoti Silhouette
20. Dhoti Internal Folds
21. Left Leg
22. Right Leg
23. Feet & Toe Details
24. Final Line Art Details
