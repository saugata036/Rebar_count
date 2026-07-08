# Rebar Counter

Counts vertical and horizontal rebar bars from a front-face cage photo using 1D projection profiles and peak detection.

## Run

```bash
.venv/bin/python rebar_xray/main.py --image input_images/testImage_2.png --output-dir outputs/test
```

Notebook: `notebooks/rebar_analysis_demo.ipynb`

## Outputs

Each run saves:

| File | Description |
|------|-------------|
| `original.jpg` | Input image copy |
| `counted.png` | Annotated overlay with bar lines and numbers |
| `analysis.json` | Counts and bar positions |

## Pipeline

1. Grayscale + foreground mask
2. Column/row dark-pixel profiles
3. `scipy.find_peaks` with adaptive spacing and edge recovery
4. Render red vertical / blue horizontal lines with numbered labels
