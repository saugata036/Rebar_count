#!/usr/bin/env python3
"""Run front-face rebar counting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from rebar_xray.config import PipelineConfig
from rebar_xray.pipeline import RebarXRayPipeline


def main() -> int:
    image_path = PROJECT_ROOT / "input_images" / "testImage_1.png"
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])

    result = RebarXRayPipeline(PipelineConfig()).run(image_path)
    payload = result.payload
    print(json.dumps(payload, indent=2))

    if payload.get("status") != "ok":
        return 2

    original = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    counted = cv2.cvtColor(cv2.imread(payload["output_image"]), cv2.COLOR_BGR2RGB)
    counts = payload["bar_counts"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(original)
    axes[0].set_title("Front-face input")
    axes[0].axis("off")
    axes[1].imshow(counted)
    axes[1].set_title(f"Counted: {counts['vertical_rods']}V / {counts['horizontal_rods']}H")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
