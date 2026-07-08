#!/usr/bin/env bash
# Open the notebook in Jupyter Lab (browser). Use this if Cursor kernel picker is empty.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv — run: bash scripts/setup_kernel.sh"
  exit 1
fi

source .venv/bin/activate
echo "Starting Jupyter Lab for: $ROOT"
echo "Open the URL shown below in your browser, then open notebooks/rebar_analysis_demo.ipynb"
echo ""
exec jupyter lab --notebook-dir="$ROOT" --no-browser
