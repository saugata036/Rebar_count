"""Metric helpers for rebar counting."""

from __future__ import annotations

import statistics
from typing import Any


def _spacing_stats(positions: list[int]) -> dict[str, float | None]:
    if len(positions) < 2:
        return {
            "average_px": None,
            "minimum_px": None,
            "std_dev_px": None,
            "samples": 0,
        }

    sorted_positions = sorted(positions)
    spacings = [abs(sorted_positions[i] - sorted_positions[i - 1]) for i in range(1, len(sorted_positions))]
    return {
        "average_px": round(float(statistics.mean(spacings)), 2),
        "minimum_px": round(float(min(spacings)), 2),
        "std_dev_px": round(float(statistics.pstdev(spacings)), 2) if len(spacings) > 1 else 0.0,
        "samples": len(spacings),
    }


def compute_metrics(counts: dict[str, Any]) -> dict[str, Any]:
    return {
        "vertical_rods": counts["vertical_rods"],
        "horizontal_rods": counts["horizontal_rods"],
        "total_rods": counts["total_rods"],
        "spacing_px": {
            "vertical": _spacing_stats(counts.get("vertical_positions_px", [])),
            "horizontal": _spacing_stats(counts.get("horizontal_positions_px", [])),
            "note": "Distances are in pixels. Multiply by mm-per-pixel scale to convert to millimeters.",
        },
        "column_region": counts.get("column_region", {}),
    }
