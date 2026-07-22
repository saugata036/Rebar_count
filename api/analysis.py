from __future__ import annotations

from pathlib import Path

from rebar_xray.config import PipelineConfig
from rebar_xray.pipeline import RebarXRayPipeline


def run_analysis(clean_image_path: Path, output_dir: Path) -> dict:
    pipeline = RebarXRayPipeline(PipelineConfig())
    result = pipeline.run(clean_image_path, output_dir=output_dir)
    return result.payload
