"""Tiny evaluation helper that summarises metrics CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _read_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _coerce_value(value: str | None) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _aggregate_by_step(rows: List[Dict[str, str]]) -> List[Dict[str, float | int]]:
    by_step: Dict[int, Dict[str, float | int]] = {}
    for row in rows:
        step = int(row.get("step", 0) or 0)
        bucket = by_step.setdefault(step, {"step": step})
        split = row.get("split", "train")
        for key, value in row.items():
            if key in {"step", "split"}:
                continue
            coerced = _coerce_value(value)
            if coerced is None:
                continue
            bucket_key = f"{split}_{key}" if key in {"loss", "accuracy"} else key
            bucket[bucket_key] = coerced
    return [by_step[key] for key in sorted(by_step)]


def summarize(rows: List[Dict[str, str]]) -> Dict:
    entries = _aggregate_by_step(rows)
    summary: Dict[str, object] = {"num_points": len(entries)}
    if not entries:
        summary["last"] = None
        summary["best_val"] = None
        return summary
    last = entries[-1]
    best = max(entries, key=lambda item: (item.get("val_accuracy", float("-inf")), item["step"]))
    summary["last"] = last
    summary["best_val"] = best
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise metrics.csv artifacts")
    parser.add_argument("--run-dir", type=Path, help="Run directory containing metrics.csv")
    parser.add_argument("--metrics", type=Path, help="Explicit metrics.csv path")
    return parser.parse_args(argv)


def _resolve_metrics_path(args: argparse.Namespace) -> Path:
    if args.metrics:
        return Path(args.metrics)
    if not args.run_dir:
        raise SystemExit("Provide --run-dir or --metrics")
    path = Path(args.run_dir) / "metrics.csv"
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    metrics_path = _resolve_metrics_path(args)
    rows = _read_metrics(metrics_path)
    summary = summarize(rows)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
