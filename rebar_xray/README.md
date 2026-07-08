# Rebar X-Ray Pipeline

Production-style pipeline that converts an RCC column site photo into a clean **X-ray / wireframe** visualization of the **front reinforcement cage**, then counts vertical and horizontal rods with numbered arrows.

## Run

```bash
.venv/bin/python rebar_xray/main.py --image input_images/PXL_20260320_102716788.jpg
```

Notebook: `notebooks/rebar_analysis_demo.ipynb`

## Pipeline

1. Load image
2. Detect column (YOLO + classical fallback)
3. Segment front cage (SAM / YOLO-seg / GrabCut)
4. Estimate depth (Depth Anything V2 / MiDaS / classical)
5. Keep only nearest front reinforcement layer
6. Denoise foreground and remove background clutter
7. Detect rods (projection peaks + LSD/Hough)
8. Render X-ray wireframe on black background
9. Add numbered arrows `1, 2, 3...` for vertical and horizontal rods

## Outputs

Each run creates a folder under `outputs/<image>_xray_<timestamp>/`:

| File | Description |
|------|-------------|
| `original.jpg` | Input photo |
| `mask.png` | Column cage mask |
| `front_mask.png` | Front-layer mask (cropped) |
| `depth.png` | Depth map (cropped) |
| `foreground.png` | Denoised front cage only |
| `vertical_lines.png` | Vertical rod wireframe |
| `horizontal_lines.png` | Horizontal stirrup wireframe |
| `rendered_xray.png` | Final X-ray |
| `rendered_xray_counted.png` | X-ray + numbered arrows |
| `analysis.json` | Counts, metrics, line coordinates |

## AI models

| Step | Primary | Fallback |
|------|---------|----------|
| Column detect | YOLOv8 | Edge/contour bbox |
| Segmentation | SAM | YOLO-seg / GrabCut |
| Depth | Depth Anything V2 | Classical focus map |
| Lines | LSD | Hough |

Disable heavy models for faster CPU runs:

```bash
.venv/bin/python rebar_xray/main.py --image your.jpg --no-sam --no-depth-anything --cpu-only
```

## Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads YOLO weights automatically. SAM and Depth Anything download on first use when enabled.
