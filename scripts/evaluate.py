#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class MetricRow:
    step: int
    train_loss: float
    train_acc: float
    val_acc: float
    test_acc: float

    @classmethod
    def from_raw(cls, row: dict[str, str]) -> "MetricRow":
        try:
            return cls(
                step=int(row["step"]),
                train_loss=float(row["train_loss"]),
                train_acc=float(row["train_acc"]),
                val_acc=float(row["val_acc"]),
                test_acc=float(row["test_acc"]),
            )
        except KeyError as exc:
            raise KeyError(f"Missing expected column {exc!s} in metrics CSV") from exc
        except ValueError as exc:
            raise ValueError(f"Could not parse metrics row: {row!r}") from exc

    def as_dict(self) -> dict[str, float | int]:
        return {
            "step": self.step,
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "test_acc": self.test_acc,
        }


def resolve_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def load_metrics(metrics_path: Path) -> List[MetricRow]:
    with metrics_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("Metrics CSV is empty or missing a header row")
        rows = [MetricRow.from_raw(row) for row in reader]
    if not rows:
        raise ValueError("Metrics CSV did not contain any metric rows")
    return rows


def summarise_metrics(rows: Sequence[MetricRow]) -> dict[str, object]:
    last = rows[-1]
    best_val = max(rows, key=lambda r: r.val_acc)
    summary = {
        "num_points": len(rows),
        "last": last.as_dict(),
        "best_val": best_val.as_dict(),
    }
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarise training metrics for a run directory.")
    ap.add_argument(
        "--run-dir",
        type=str,
        default="runs/modular_addition",
        help="Path to the run directory containing metrics.csv",
    )
    ap.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Explicit path to a metrics CSV file (overrides --run-dir)",
    )
    ap.add_argument(
        "--output",
        choices=["json"],
        default="json",
        help="Output format. Only JSON is currently supported.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    metrics_path = resolve_path(args.metrics) if args.metrics else resolve_path(args.run_dir) / "metrics.csv"
    if not metrics_path.exists():
        raise SystemExit(f"Metrics file not found: {metrics_path}")

    rows = load_metrics(metrics_path)
    summary = summarise_metrics(rows)

    if args.output == "json":
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
