"""Simple front-face rebar counting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rebar_xray.config import PipelineConfig
from rebar_xray.counter import count_bars
from rebar_xray.renderer import draw_counts
from rebar_xray.utils import LOGGER, load_bgr, save_image, save_json


@dataclass
class PipelineResult:
    payload: dict
    output_dir: Path


class RebarXRayPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.config.ensure_dirs()

    def run(self, image_path: str | Path, output_dir: str | Path | None = None) -> PipelineResult:
        image_path = Path(image_path)
        run_dir = Path(output_dir) if output_dir else self.config.output_dir / image_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Loading front-face image: %s", image_path)
        image_bgr = load_bgr(image_path)

        LOGGER.info("Counting vertical and horizontal bars")
        result = count_bars(image_bgr, self.config.count)
        counted = draw_counts(
            image_bgr,
            result.vertical_positions,
            result.horizontal_positions,
            result.region,
            self.config.render,
        )

        save_image(run_dir / "original.jpg", image_bgr)
        save_image(run_dir / "counted.png", counted)

        payload = {
            "input_image": str(image_path.resolve()),
            "output_dir": str(run_dir.resolve()),
            "output_image": str((run_dir / "counted.png").resolve()),
            "bar_counts": {
                "vertical_rods": result.vertical_count,
                "horizontal_rods": result.horizontal_count,
            },
            "vertical_positions_px": result.vertical_positions,
            "horizontal_positions_px": result.horizontal_positions,
            "debug": result.debug,
            "status": "ok",
            "message": f"Found {result.vertical_count} vertical and {result.horizontal_count} horizontal bars.",
        }
        save_json(run_dir / "analysis.json", payload)
        return PipelineResult(payload=payload, output_dir=run_dir)
