#!/usr/bin/env python3
"""CLI: count vertical and horizontal rebar on a front-face image."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count vertical and horizontal rebar on a front-face image.")
    parser.add_argument("--image", required=True, help="Front-face rebar image path")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)

    config = PipelineConfig()
    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    try:
        result = RebarXRayPipeline(config).run(args.image, output_dir=args.output_dir)
        print(json.dumps(result.payload, indent=2))
        return 0 if result.payload["status"] == "ok" else 2
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "message": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
