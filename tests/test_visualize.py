import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest
from src.scripts import visualize


@pytest.fixture()
def fake_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_alpha"
    run_dir.mkdir()
    rows = [
        {
            "step": 0,
            "split": "train",
            "loss": 1.0,
            "accuracy": 0.5,
            "l1_norm": 0.1,
            "l2_norm": 0.2,
            "nuclear_norm": 0.3,
            "train_err": 0.5,
            "test_err": 0.6,
            "alpha": 0.1,
            "beta": 0.2,
            "dataset_fraction": 0.4,
            "weight_decay": 0.01,
            "lr": 0.001,
            "seed": 1,
            "regularizer": "l2",
        },
        {
            "step": 0,
            "split": "val",
            "loss": 1.2,
            "accuracy": 0.4,
            "l1_norm": 0.1,
            "l2_norm": 0.2,
            "nuclear_norm": 0.3,
            "train_err": 0.5,
            "test_err": 0.6,
            "alpha": 0.33,
            "beta": 0.44,
            "dataset_fraction": 0.55,
            "weight_decay": 0.02,
            "lr": 0.002,
            "seed": 2,
            "regularizer": "laplace",
        },
        {
            "step": 1,
            "split": "train",
            "loss": 0.2,
            "accuracy": 1.0,
            "l1_norm": 0.15,
            "l2_norm": 0.25,
            "nuclear_norm": 0.35,
            "train_err": 0.4,
            "test_err": 0.5,
            "alpha": 0.1,
            "beta": 0.2,
            "dataset_fraction": 0.4,
            "weight_decay": 0.01,
            "lr": 0.001,
            "seed": 1,
            "regularizer": "l2",
        },
        {
            "step": 1,
            "split": "val",
            "loss": 0.1,
            "accuracy": 0.995,
            "l1_norm": 0.12,
            "l2_norm": 0.22,
            "nuclear_norm": 0.32,
            "train_err": 0.35,
            "test_err": 0.45,
            "alpha": 0.33,
            "beta": 0.44,
            "dataset_fraction": 0.55,
            "weight_decay": 0.02,
            "lr": 0.002,
            "seed": 2,
            "regularizer": "laplace",
        },
    ]
    fieldnames = list(rows[0].keys())
    with (run_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "singular_values.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "s1"])
        writer.writeheader()
        writer.writerows([{"step": 0, "s1": 1.0}, {"step": 1, "s1": 0.9}])
    return run_dir


def test_load_run_extracts_val_metadata(fake_run: Path):
    run = visualize.load_run(fake_run)
    meta = run["meta"]
    assert meta["name"] == fake_run.name
    # Metadata should come from the first val row when available.
    assert meta["alpha"] == pytest.approx(0.33)
    assert meta["regularizer"] == "laplace"
    assert run["singular"] is not None


def test_parse_args_allows_single_run_without_output(monkeypatch):
    argv = ["visualize.py", "--run", "some/run/path", "--xscale", "linear"]
    monkeypatch.setattr(sys, "argv", argv)
    args = visualize.parse_args()
    assert args.run == "some/run/path"
    assert args.runs is None
    assert args.output_dir is None
    assert args.xscale == "linear"


def test_visualize_script_creates_expected_plots(fake_run: Path, tmp_path: Path, monkeypatch):
    outdir = tmp_path / "plots"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    cmd = [
        sys.executable,
        "-m",
        "src.scripts.visualize",
        "--run",
        str(fake_run),
        "--output_dir",
        str(outdir),
        "--grok_patience",
        "1",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parents[1]
    )
    if result.returncode != 0:
        raise AssertionError(
            f"visualize.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    per_run_dir = outdir / fake_run.name
    expected_files = [
        per_run_dir / f"{fake_run.name}_loss_linear.png",
        per_run_dir / f"{fake_run.name}_loss_linear.svg",
        per_run_dir / f"{fake_run.name}_accuracy_linear.png",
        outdir / "time_to_generalization_t2.png",
    ]
    for path in expected_files:
        assert path.exists(), f"Expected plot missing: {path}"


def test_time_to_generalization_handles_missing_threshold(tmp_path: Path):
    run_dir = tmp_path / "no_grok"
    run_dir.mkdir()
    rows = []
    for step in (1, 2, 3):
        rows.append(
            {
                "step": step,
                "split": "train",
                "loss": 1.0,
                "accuracy": 0.2,
                "spectral_low_frac": "",
                "spectral_entropy": "",
            }
        )
        rows.append(
            {
                "step": step,
                "split": "val",
                "loss": 0.8,
                "accuracy": 0.3,
                "spectral_low_frac": "",
                "spectral_entropy": "",
            }
        )
    with (run_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    visualize.plot_time_to_generalization(
        [visualize.load_run(run_dir)],
        tmp_path / "plots",
        threshold=0.99,
        patience=1,
    )
    assert (tmp_path / "plots" / "time_to_generalization_t2.png").exists()
