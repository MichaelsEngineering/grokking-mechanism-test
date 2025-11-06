"""Tests for the fixture-backed evaluation workflow.

The fixture provides a deterministic slice of training metrics captured from a
short, fixed-seed run. Exercising the evaluation entrypoint against that
snapshot protects research reproducibility by ensuring metric summarisation is
stable even when the full training loop is too costly to execute in CI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("run_subdir", ["mini_run"])
def test_evaluate_fixture_snapshot(run_subdir: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "tests" / "fixtures" / run_subdir
    scripts_dir = repo_root / "src" / "scripts"

    cmd = [
        sys.executable,
        str(scripts_dir / "evaluate.py"),
        "--run-dir",
        str(run_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    metrics = json.loads(result.stdout)

    assert metrics["num_points"] == 2

    expected_last = {
        "step": 8,
        "train_loss": 4.5537109375,
        "train_acc": 0.09375,
        "val_acc": 0.0729166667,
        "test_acc": 0.0625,
    }

    for field in ("last", "best_val"):
        got = metrics[field]
        assert got["step"] == expected_last["step"]
        for key in ("train_loss", "train_acc", "val_acc", "test_acc"):
            assert got[key] == pytest.approx(expected_last[key])
