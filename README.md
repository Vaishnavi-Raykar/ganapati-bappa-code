# Ganapati Animated Vector Sketching Application 🎨🙏

A Python + Pygame application that reconstructs the reference Ganapati line art as a continuous vector sketch animation.

![Reference Image](assets/reference_image.png)

## Features

- **Vector Line Art**: Smooth, 3x supersampled (2700x3300) vector curves downsampled to 900x1100 for high-quality anti-aliased rendering.
- **Progressive Animation**: Single pen tip pointer animating smoothly through 24 anatomical drawing steps.
- **Zero Artifacts**: No dotted trails, no extra circles, no straight rays to origin, and no upcoming path highlights.
- **Interactive Controls**: Real-time speed adjustment, reset, and pause/restart.

## Requirements

- Python 3.8+
- Pygame 2.0+
- NumPy
- OpenCV Python

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ganapati-sketch.git
   cd ganapati-sketch
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r ganapati_python/requirements.txt
   ```

3. **Run the Application**:
   ```bash
   cd ganapati_python
   python main.py
   ```

## Controls

| Key | Action |
| --- | --- |
| **`SPACE`** | Restart animation |
| **`R`** | Reset canvas |
| **`+` / `=`** | Increase drawing speed |
| **`-`** | Decrease drawing speed |
| **`ESC`** | Exit application |

## Project Structure

```
c:\Projects\ganesha\
├── assets/
│   └── reference_image.png       # Original reference line art image
├── ganapati_python/
│   ├── animation.py               # Animation progress & pointer trajectory controller
│   ├── config.py                  # Canvas resolution, colors, and step sequence settings
│   ├── drawing.py                 # Clean vector polyline rendering engine
│   ├── main.py                    # Pygame application entry point and main loop
│   ├── paths.py                   # DrawingPath model & vector_data loader
│   ├── requirements.txt           # Dependency requirements
│   └── vector_data.json           # Extracted smooth vector stroke paths
├── generate_vector_json.py        # Vector path generator script
├── .gitignore                     # Git ignore rules
└── README.md                      # Repository documentation
```
