from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/step8_full_universe/runner.py"
PYTHON = sys.executable


def test_runner_help_exposes_r1_lifecycle_commands() -> None:
    result = subprocess.run([PYTHON, str(RUNNER), "--help"], cwd=ROOT, capture_output=True, text=True, check=True)
    assert "score-dev" in result.stdout
    assert "training-smoke" in result.stdout


def test_runner_test_command_fails_closed() -> None:
    result = subprocess.run([PYTHON, str(RUNNER), "test"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "STEP8_TEST_ACCESS_FORBIDDEN" in result.stderr
