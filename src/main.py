#!/usr/bin/env python3
"""Backward-compatible entry point that runs the X-ray pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rebar_xray.config import PipelineConfig
from rebar_xray.pipeline import RebarXRayPipeline
from rebar_xray.utils import setup_logging

OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_IMAGE = PROJECT_ROOT / "input_images" / "PXL_20260320_102716788.jpg"


def analyze_image(
    image_path: str | Path,
    output_dir: str | Path = OUTPUT_DIR,
    use_yolo_if_noisy: bool = False,
    yolo_model_path: str = "yolov8n-seg.pt",
    min_line_length: int = 20,
) -> dict:
    del use_yolo_if_noisy, yolo_model_path, min_line_length
    setup_logging()
    config = PipelineConfig(output_dir=Path(output_dir))
    return RebarXRayPipeline(config).run(image_path, output_dir=output_dir).payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze RCC column reinforcement and generate X-ray visualization.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args(argv)
    try:
        result = analyze_image(args.image, args.output_dir)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "message": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
