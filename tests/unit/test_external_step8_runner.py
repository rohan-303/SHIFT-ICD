from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "external_step8" / "run_step8.py"


def invoke(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_external_runner_help_lists_contract_commands() -> None:
    result = invoke("--help")
    assert result.returncode == 0
    for command in ("preflight", "verify-data", "smoke", "zero-shot-dev", "training-probe", "test", "status"):
        assert command in result.stdout


def test_external_runner_cpu_preflight_passes() -> None:
    result = invoke("preflight", "--cpu-check")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["cpu_check"] is True


def test_external_runner_test_is_locked() -> None:
    result = invoke("test")
    assert result.returncode != 0
    assert "TEST_LOCK_REQUIRED" in result.stderr
