"""Configuration for front-face rebar counting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CountConfig:
    background_threshold: int = 240
    vertical_distance_divisor: int = 7
    horizontal_distance_divisor: int = 12
    prominence_fraction: float = 0.08
    min_peak_distance_px: int = 20
    adaptive_peak_distance: bool = True
    rough_peak_distance_divisor: int = 20
    adaptive_spacing_fraction: float = 0.55
    recover_edge_peaks: bool = True
    edge_prominence_fraction: float = 0.5
    edge_gap_fraction: float = 0.45
    edge_peak_height_fraction: float = 0.58
    edge_margin_px: int = 15


@dataclass
class RenderConfig:
    vertical_color: tuple[int, int, int] = (0, 0, 255)
    horizontal_color: tuple[int, int, int] = (255, 0, 0)
    line_thickness: int = 12
    label_vertical_color: tuple[int, int, int] = (0, 0, 255)
    label_horizontal_color: tuple[int, int, int] = (255, 0, 0)


@dataclass
class PipelineConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "outputs")
    count: CountConfig = field(default_factory=CountConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
