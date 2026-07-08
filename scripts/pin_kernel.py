#!/usr/bin/env python3
"""Pin the project .venv as the workspace Python interpreter for Cursor/VS Code."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
WORKSPACE_ID = "bc736441df6abdf5eba50f3069a6d305"
STATE_DB = Path.home() / "Library/Application Support/Cursor/User/workspaceStorage" / WORKSPACE_ID / "state.vscdb"
WORKSPACE_KEY = f"autoSelectedWorkspacePythonInterpreter-{PROJECT_ROOT}"


def _interpreter_metadata() -> dict:
    if not VENV_PYTHON.exists():
        raise SystemExit(f"Missing venv interpreter: {VENV_PYTHON}")

    import subprocess

    version_json = subprocess.check_output(
        [str(VENV_PYTHON), "-c", "import json, sys; print(json.dumps(list(sys.version_info[:3])))"],
        text=True,
    )
    major, minor, patch = json.loads(version_json)
    sys_version = subprocess.check_output(
        [str(VENV_PYTHON), "-c", "import sys; print(sys.version)"],
        text=True,
    ).strip()
    version_label = f"{major}.{minor}.{patch}"

    return {
        "id": str(VENV_PYTHON),
        "sysPrefix": str(PROJECT_ROOT / ".venv"),
        "envType": "Venv",
        "envName": ".venv",
        "envPath": str(PROJECT_ROOT / ".venv"),
        "path": str(VENV_PYTHON),
        "architecture": 3,
        "sysVersion": sys_version,
        "version": {
            "raw": version_label,
            "major": major,
            "minor": minor,
            "patch": patch,
            "build": [],
            "prerelease": ["final", "0"],
        },
        "displayName": f"Python {version_label} ('.venv': venv)",
        "detailedDisplayName": f"Python {version_label} ('.venv': venv)",
    }


def main() -> None:
    if not VENV_PYTHON.exists():
        raise SystemExit(f"Missing venv interpreter: {VENV_PYTHON}")
    if not STATE_DB.exists():
        raise SystemExit(f"Cursor workspace database not found: {STATE_DB}")

    conn = sqlite3.connect(STATE_DB)
    row = conn.execute("SELECT value FROM ItemTable WHERE key = ?", ("ms-python.python",)).fetchone()
    if not row:
        raise SystemExit("ms-python.python workspace state not found. Open this project in Cursor first.")

    state = json.loads(row[0])
    state[WORKSPACE_KEY] = _interpreter_metadata()
    state["autoSelectionInterpretersQueried-" + str(PROJECT_ROOT)] = True

    conn.execute(
        "UPDATE ItemTable SET value = ? WHERE key = ?",
        (json.dumps(state), "ms-python.python"),
    )
    conn.commit()
    conn.close()

    print("Pinned workspace interpreter to:")
    print(f"  {VENV_PYTHON}")
    print("Reload Cursor, then run the notebook.")


if __name__ == "__main__":
    main()
