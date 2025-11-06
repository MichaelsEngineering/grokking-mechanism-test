"""Small visualization helper used by the test-suite.

The functions implemented here purposely cover just enough surface area for
unit tests:

* ``load_run`` loads metrics/singular value CSVs and exposes a metadata dict.
* ``parse_args`` mirrors the CLI interface exercised in the subprocess test.
* ``main`` produces a couple of simple matplotlib plots so the test can assert
  that images are created.  The plotting utilities themselves are intentionally
  lightweight so we can evolve them later without breaking the contract.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CSV helpers
# -----------------------------------------------------------------------------


def _autocast(value: str | None):
    if value is None or value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def load_run(run_dir: Path) -> Dict:
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv in {run_dir}")
    metrics = _read_csv(metrics_path)

    singular_path = run_dir / "singular_values.csv"
    singular = _read_csv(singular_path) if singular_path.exists() else None

    meta_row = next((row for row in metrics if row.get("split") == "val"), metrics[0])
    meta = {"name": run_dir.name}
    for key, value in meta_row.items():
        if key == "split":
            continue
        meta[key] = _autocast(value)

    return {
        "name": run_dir.name,
        "path": run_dir,
        "metrics": metrics,
        "meta": meta,
        "singular": singular,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize grokking runs")
    parser.add_argument("--run", type=str, help="Shortcut for a single run directory")
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Multiple run directories or glob patterns",
    )
    parser.add_argument("--output_dir", type=str, help="Directory for generated plots")
    parser.add_argument(
        "--xscale", choices=["linear", "log"], default="linear", help="X-axis scale"
    )
    parser.add_argument("--grok_threshold", type=float, default=0.99)
    parser.add_argument("--grok_patience", type=int, default=50)
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    return parser.parse_args(argv)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------


def _group_by_split(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        split = row.get("split", "train")
        grouped.setdefault(split, []).append(row)
    for values in grouped.values():
        values.sort(key=lambda r: int(r.get("step", 0) or 0))
    return grouped


def _to_series(rows: List[Dict[str, str]], key: str):
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        if key not in row:
            continue
        try:
            xs.append(float(row.get("step", 0) or 0))
            ys.append(float(row[key]))
        except (TypeError, ValueError):
            continue
    return xs, ys


def plot_core_curves(run: Dict, out_dir: Path, *, xscale: str = "linear") -> None:
    grouped = _group_by_split(run["metrics"])
    base = f"{run['name']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _plot(metric: str, svg: bool) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        for split, rows in grouped.items():
            xs, ys = _to_series(rows, metric)
            if xs:
                ax.plot(xs, ys, label=split)
        ax.set_xlabel("step")
        ax.set_ylabel(metric)
        ax.set_xscale(xscale)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        png_path = out_dir / f"{base}_{metric}_{xscale}.png"
        fig.savefig(png_path)
        if svg:
            fig.savefig(out_dir / f"{base}_{metric}_{xscale}.svg")
        plt.close(fig)

    _plot("loss", svg=True)
    _plot("accuracy", svg=False)


def _compute_t2(rows: List[Dict[str, str]], *, threshold: float, patience: int) -> float:
    val_rows = [row for row in rows if row.get("split") == "val"]
    val_rows.sort(key=lambda r: int(r.get("step", 0) or 0))
    streak = 0
    for row in val_rows:
        try:
            acc = float(row.get("accuracy", 0.0))
        except (TypeError, ValueError):
            continue
        if acc >= threshold:
            streak += 1
            if streak >= max(1, patience):
                return float(row.get("step", 0) or 0)
        else:
            streak = 0
    return math.nan


def plot_time_to_generalization(
    runs: List[Dict], out_dir: Path, *, threshold: float, patience: int
) -> None:
    if not runs:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [run["name"] for run in runs]
    values = [
        _compute_t2(run["metrics"], threshold=threshold, patience=patience)
        for run in runs
    ]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, values, color="tab:green")
    ax.set_ylabel("t2 (steps)")
    ax.set_title("Time to generalization")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "time_to_generalization_t2.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _expand_run_args(single: Optional[str], patterns: Optional[Sequence[str]]) -> List[Path]:
    raw: List[str] = []
    if single:
        raw.append(single)
    if patterns:
        for pat in patterns:
            matches = glob.glob(pat)
            raw.extend(matches if matches else [pat])
    if not raw:
        raise SystemExit("Please provide at least one run via --run or --runs.")
    return [Path(item) for item in raw]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_paths = _expand_run_args(args.run, args.runs)
    out_dir = Path(args.output_dir) if args.output_dir else (
        run_paths[0] / "plots" if len(run_paths) == 1 else Path("plots")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [load_run(path) for path in run_paths]
    for run in runs:
        plot_core_curves(run, out_dir / run["name"], xscale=args.xscale)
    plot_time_to_generalization(
        runs,
        out_dir,
        threshold=args.grok_threshold,
        patience=args.grok_patience,
    )

    if args.show:  # pragma: no cover - used manually
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
