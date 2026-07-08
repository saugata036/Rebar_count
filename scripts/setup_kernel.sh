#!/usr/bin/env bash
# One-time setup: register the project venv as a Jupyter kernel for notebooks.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Creating .venv..."
  if command -v python3.13 >/dev/null 2>&1; then
    python3.13 -m venv .venv
  else
    python3 -m venv .venv
  fi
fi

source .venv/bin/activate
pip install -q -r requirements.txt
python -m ipykernel install --user --name constraction-rebar-analyzer --display-name "Constraction Rebar (.venv)"

echo ""
echo "Kernel registered: Constraction Rebar (.venv)"
echo "In the notebook picker choose: Jupyter Kernel... -> Constraction Rebar (.venv)"
