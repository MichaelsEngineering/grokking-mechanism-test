#!/usr/bin/env python3
"""
visualize.py — Multi-run visualization for grokking-mechanism-test

Features (aligned with Power et al. 2022 and Tikeng Notsawo et al. 2025):
- Load 1..N run directories and aggregate metrics
- Core curves: train/val loss & accuracy vs steps (linear and log x-axis)
- Grokking markers: t1 (memorization), t2 (generalization at threshold), Δt
- Norm dynamics: L1/L2/Nuclear (ℓ*) norms vs steps when present
- Optional task metrics: sparse recovery and low-rank errors
- Singular-value trajectory plots when singular_values.csv exists
- Summary plots across runs: time-to-generalization (t2) and Δt comparisons

Expected files per run directory:
- metrics.csv with columns (any missing are optional and handled):
  step,split,loss,accuracy,l1_norm,l2_norm,nuclear_norm,
  train_err,test_err,alpha,beta,dataset_fraction,weight_decay,lr,seed,regularizer
- optional singular_values.csv: step,s1,s2,...

Outputs: .png and .svg files written to --output_dir.
"""
from __future__ import annotations
import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import csv

# Soft deps: prefer pandas if present; otherwise fall back to csv module
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

import matplotlib.pyplot as plt

# -------------------------------
# Utilities
# -------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Read CSV robustly without pandas."""
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k.strip(): v.strip() for k, v in r.items()})
    return rows


def _to_float(v: Optional[str]) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def _to_builtin_scalar(v):
    """Convert numpy/pandas scalars to vanilla Python scalars when possible."""
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return v
    return v


def _group_runs_by_tag(runs: List[Dict], tag_keys: List[str]) -> Dict[Tuple, List[Dict]]:
    groups: Dict[Tuple, List[Dict]] = {}
    for meta in runs:
        key = tuple(meta.get(k) for k in tag_keys)
        groups.setdefault(key, []).append(meta)
    return groups


# -------------------------------
# Loading and normalization
# -------------------------------

def load_run(run_dir: Path) -> Dict:
    """Load a single run directory into a normalized structure.

    Returns dict with keys:
      name, path, metrics (list of dict), df (pandas or None), meta (scalar hparams),
      singular (list) if singular_values.csv exists
    """
    run_dir = run_dir.resolve()
    meta: Dict[str, Optional[float] | Optional[str]] = {}

    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv in {run_dir}")

    if pd is not None:
        df = pd.read_csv(metrics_path)
        # Normalize column names
        df.columns = [c.strip() for c in df.columns]
        # Coerce common numeric fields
        for col in [
            "step","loss","accuracy","l1_norm","l2_norm","nuclear_norm",
            "train_err","test_err","alpha","beta","dataset_fraction","weight_decay","lr","seed"
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        rows = _read_csv(metrics_path)
        df = None

    # Rough metadata extraction from first row of `val` split if present else first row
    if pd is not None:
        row = None
        if df is not None and not df.empty:
            if "split" in df.columns:
                val_rows = df[df["split"] == "val"]
                if not val_rows.empty:
                    row = val_rows.iloc[0]
            if row is None:
                row = df.iloc[0]
        for k in ["alpha","beta","dataset_fraction","weight_decay","lr","regularizer"]:
            if row is not None and k in df.columns:
                meta[k] = _to_builtin_scalar(row[k])
            else:
                meta[k] = None
        # Add path/name
        meta["name"] = run_dir.name
        meta["path"] = str(run_dir)
    else:
        # csv fallback
        row_choice = None
        if rows:
            for cand in rows:
                if cand.get("split") == "val":
                    row_choice = cand
                    break
            if row_choice is None:
                row_choice = rows[0]
        for k in ["alpha","beta","dataset_fraction","weight_decay","lr","regularizer"]:
            if row_choice is not None:
                meta[k] = _to_float(row_choice.get(k)) if k != "regularizer" else row_choice.get(k)
            else:
                meta[k] = None
        meta["name"] = run_dir.name
        meta["path"] = str(run_dir)

    # Singular values (optional)
    sing_path = run_dir / "singular_values.csv"
    singular = None
    if sing_path.exists():
        if pd is not None:
            singular = pd.read_csv(sing_path)
            singular.columns = [c.strip() for c in singular.columns]
        else:
            singular = _read_csv(sing_path)

    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "df": df,
        "rows": None if pd is not None else rows,
        "meta": meta,
        "singular": singular,
    }


# -------------------------------
# Grokking metrics (t1, t2, Δt)
# -------------------------------

def compute_t1_t2(
    df_or_rows,
    grok_threshold: float,
    grok_patience: int,
) -> Tuple[Optional[int], Optional[int]]:
    """Compute t1 (train memorization) and t2 (val generalization) with patience.
    - t1: first step where TRAIN accuracy >= 0.999 (configurable in future), sustained for `grok_patience` steps.
    - t2: first step where VAL accuracy >= grok_threshold, sustained for `grok_patience` steps.
    Returns (t1, t2) as integer steps if found.
    """
    if pd is not None and hasattr(df_or_rows, "__class__") and df_or_rows.__class__.__name__ == "DataFrame":
        df = df_or_rows.copy()
        if "step" not in df.columns or "split" not in df.columns or "accuracy" not in df.columns:
            return (None, None)
        df = df.sort_values("step")

        t1 = _first_sustained_crossing(df[df["split"] == "train"], "accuracy", 0.999, grok_patience)
        t2 = _first_sustained_crossing(df[df["split"] == "val"], "accuracy", grok_threshold, grok_patience)
        return (t1, t2)
    else:
        # csv fallback path
        rows: List[Dict[str, str]] = df_or_rows
        # sort
        rows = sorted(rows, key=lambda r: float(r.get("step", 0)))
        t1 = _first_sustained_crossing_rows(rows, split="train", col="accuracy", thr=0.999, patience=grok_patience)
        t2 = _first_sustained_crossing_rows(rows, split="val", col="accuracy", thr=grok_threshold, patience=grok_patience)
        return (t1, t2)


def _first_sustained_crossing(df, col: str, thr: float, patience: int) -> Optional[int]:
    if df.empty or col not in df.columns:
        return None
    arr = df[["step", col]].dropna().values
    if arr.size == 0:
        return None
    # sliding window
    steps = df["step"].tolist()
    vals = df[col].tolist()
    cnt = 0
    start_step = None
    for s, v in zip(steps, vals):
        if v is not None and v >= thr:
            cnt += 1
            if start_step is None:
                start_step = s
            if cnt >= patience:
                return int(start_step)
        else:
            cnt = 0
            start_step = None
    return None


def _first_sustained_crossing_rows(rows: List[Dict[str, str]], split: str, col: str, thr: float, patience: int) -> Optional[int]:
    cnt = 0
    start_step = None
    for r in rows:
        if r.get("split") != split:
            continue
        try:
            s = int(float(r.get("step", 0)))
            v = float(r.get(col, "nan"))
        except Exception:
            continue
        if math.isnan(v):
            continue
        if v >= thr:
            cnt += 1
            if start_step is None:
                start_step = s
            if cnt >= patience:
                return start_step
        else:
            cnt = 0
            start_step = None
    return None


# -------------------------------
# Plotting helpers
# -------------------------------

def _line_plot(ax, x, y, label: str, xscale: str = "linear", alpha: float = 0.9):
    ax.plot(x, y, label=label, alpha=alpha)
    if xscale in {"log", "log10"}:
        ax.set_xscale("log")


def _finish(ax, title: str, xlabel: str = "steps", ylabel: str = "", legend: bool = True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()
    ax.grid(True, alpha=0.25)


def _save(fig, outdir: Path, fname: str, tight: bool = True):
    _ensure_dir(outdir)
    png = outdir / f"{fname}.png"
    svg = outdir / f"{fname}.svg"
    if tight:
        fig.tight_layout()
    fig.savefig(png, dpi=200)
    fig.savefig(svg)
    plt.close(fig)


# -------------------------------
# Plotting per-run curves
# -------------------------------

def plot_core_curves(run: Dict, outdir: Path, xscale: str):
    label = run["name"]
    if pd is not None and run["df"] is not None:
        df = run["df"].copy().sort_values("step")
        # Loss
        fig, ax = plt.subplots(figsize=(7,4))
        for split in ["train", "val"]:
            if "split" in df.columns and (df["split"] == split).any():
                sub = df[df["split"] == split]
                if "loss" in sub.columns:
                    _line_plot(ax, sub["step"], sub["loss"], f"{label}-{split}", xscale=xscale)
        _finish(ax, f"Loss vs Steps — {label}", ylabel="loss")
        _save(fig, outdir, f"{label}_loss_{xscale}")

        # Accuracy
        fig, ax = plt.subplots(figsize=(7,4))
        for split in ["train", "val"]:
            if "split" in df.columns and (df["split"] == split).any():
                sub = df[df["split"] == split]
                if "accuracy" in sub.columns:
                    _line_plot(ax, sub["step"], sub["accuracy"], f"{label}-{split}", xscale=xscale)
        _finish(ax, f"Accuracy vs Steps — {label}", ylabel="accuracy")
        _save(fig, outdir, f"{label}_accuracy_{xscale}")

        # Norms
        has_norm = False
        fig, ax = plt.subplots(figsize=(7,4))
        for c, nice in [("l1_norm","L1"),("l2_norm","L2"),("nuclear_norm","nuclear")]:
            if c in df.columns and df[c].notna().any():
                has_norm = True
                sub = df[df[c].notna()]
                _line_plot(ax, sub["step"], sub[c], f"{nice}", xscale=xscale)
        if has_norm:
            _finish(ax, f"Parameter Norms vs Steps — {label}", ylabel="norm")
            _save(fig, outdir, f"{label}_norms_{xscale}")

        # Task errors if present
        if "train_err" in df.columns or "test_err" in df.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            if "train_err" in df.columns and df["train_err"].notna().any():
                sub = df[df["train_err"].notna()]
                _line_plot(ax, sub["step"], sub["train_err"], f"train_err", xscale=xscale)
            if "test_err" in df.columns and df["test_err"].notna().any():
                sub = df[df["test_err"].notna()]
                _line_plot(ax, sub["step"], sub["test_err"], f"test_err", xscale=xscale)
            _finish(ax, f"Task Errors vs Steps — {label}", ylabel="error")
            _save(fig, outdir, f"{label}_task_errors_{xscale}")

        # Grok markers
        t1, t2 = compute_t1_t2(df, grok_threshold=args.grok_threshold, grok_patience=args.grok_patience)
        if t1 is not None or t2 is not None:
            # Overlay markers on accuracy plot (linear scale by default)
            fig, ax = plt.subplots(figsize=(7,4))
            for split in ["train", "val"]:
                if (df["split"] == split).any() and ("accuracy" in df.columns):
                    sub = df[df["split"] == split]
                    _line_plot(ax, sub["step"], sub["accuracy"], f"{label}-{split}", xscale="linear")
            if t1 is not None:
                ax.axvline(t1, color="tab:orange", linestyle="--", alpha=0.8, label=f"t1={t1}")
            if t2 is not None:
                ax.axvline(t2, color="tab:green", linestyle="--", alpha=0.8, label=f"t2={t2}")
            _finish(ax, f"Accuracy with Grok Markers — {label}", ylabel="accuracy")
            _save(fig, outdir, f"{label}_accuracy_markers_linear")
    else:
        # csv fallback minimal plots
        rows = run["rows"]
        # organize by split
        by_split: Dict[str, List[Tuple[int,float,float]]] = {}
        for r in rows:
            s = r.get("split", "")
            step = int(float(r.get("step", 0)))
            loss = _to_float(r.get("loss")) or float("nan")
            acc = _to_float(r.get("accuracy")) or float("nan")
            by_split.setdefault(s, []).append((step, loss, acc))
        for s in by_split.values():
            s.sort(key=lambda x: x[0])
        def _plot_xy(vals, idx, title, ylab, fname):
            fig, ax = plt.subplots(figsize=(7,4))
            for split, seq in by_split.items():
                x = [v[0] for v in seq]
                y = [v[idx] for v in seq]
                _line_plot(ax, x, y, f"{label}-{split}", xscale=xscale)
            _finish(ax, title, ylabel=ylab)
            _save(fig, outdir, fname)
        _plot_xy(by_split, 1, f"Loss vs Steps — {label}", "loss", f"{label}_loss_{xscale}")
        _plot_xy(by_split, 2, f"Accuracy vs Steps — {label}", "accuracy", f"{label}_accuracy_{xscale}")


# -------------------------------
# Cross-run summary plots
# -------------------------------

def plot_time_to_generalization(runs: List[Dict], outdir: Path):
    pts = []
    labels = []
    for run in runs:
        df = run["df"] if pd is not None else None
        rows = run["rows"] if pd is None else None
        t1, t2 = compute_t1_t2(df if df is not None else rows, args.grok_threshold, args.grok_patience)
        if t2 is not None:
            pts.append((t1, t2))
            name = run["name"]
            meta = run["meta"]
            label = name
            # enrich label with a few tags if present
            tag_bits = []
            for k in ["alpha","beta","dataset_fraction","regularizer"]:
                v = meta.get(k)
                if v is not None and v != "":
                    tag_bits.append(f"{k}={v}")
            if tag_bits:
                label += " (" + ", ".join(tag_bits) + ")"
            labels.append(label)
    if not pts:
        return
    t1s = [p[0] if p[0] is not None else float('nan') for p in pts]
    t2s = [p[1] for p in pts]
    deltas = [ (p[1] - p[0]) if (p[0] is not None) else float('nan') for p in pts ]

    # Plot t2
    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(range(len(t2s)), t2s, color="tab:green", alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    _finish(ax, "Time to Generalization (t2)", xlabel="steps", legend=False)
    _save(fig, outdir, "time_to_generalization_t2")

    # Plot Δt where available
    if not all(math.isnan(d) for d in deltas):
        fig, ax = plt.subplots(figsize=(8,4))
        ax.barh(range(len(deltas)), deltas, color="tab:blue", alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        _finish(ax, "Grokking Delay (Δt = t2 - t1)", xlabel="steps", legend=False)
        _save(fig, outdir, "grokking_delay_delta_t")


# -------------------------------
# Main
# -------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize grokking runs (multi-run supported)")
    p.add_argument("--run", type=str, help="Single run directory (shortcut for --runs)")
    p.add_argument("--runs", nargs="+", help="Run directories or glob patterns (space separated)")
    p.add_argument("--output_dir", type=str, help="Directory for output plots (defaults to run/plots or ./plots)")
    p.add_argument("--xscale", type=str, default="linear", choices=["linear","log"], help="X-axis scale")
    p.add_argument("--show", action="store_true", help="Show interactive windows (in addition to saving)")
    p.add_argument("--grok_threshold", type=float, default=0.99, help="Val accuracy threshold for t2")
    p.add_argument("--grok_patience", type=int, default=100, help="Steps to confirm sustained crossing")
    p.add_argument("--tag_keys", nargs="*", default=["alpha","beta","dataset_fraction","regularizer"], help="Keys to group/label runs")
    return p.parse_args()


def expand_runs(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pat in patterns:
        for m in glob.glob(pat):
            p = Path(m)
            if p.is_dir():
                paths.append(p)
    # de-dup and sort
    uniq = sorted(set(str(p.resolve()) for p in paths))
    return [Path(s) for s in uniq]


if __name__ == "__main__":
    args = parse_args()

    run_patterns: List[str] = []
    if args.run:
        run_patterns.append(args.run)
    if args.runs:
        run_patterns.extend(args.runs)
    if not run_patterns:
        raise SystemExit("Please provide at least one run via --run or --runs.")

    run_paths = expand_runs(run_patterns)
    if not run_paths:
        raise SystemExit("No run directories found for provided patterns.")

    if args.output_dir:
        outdir = Path(args.output_dir)
    else:
        outdir = run_paths[0] / "plots" if len(run_paths) == 1 else Path("plots")
    _ensure_dir(outdir)

    runs: List[Dict] = []
    for rp in run_paths:
        try:
            runs.append(load_run(rp))
        except Exception as e:
            print(f"[WARN] Skipping {rp}: {e}")

    # Per-run plots (both linear and log scale if user wants both, call twice externally)
    for run in runs:
        per_out = outdir / run["name"]
        _ensure_dir(per_out)
        plot_core_curves(run, per_out, xscale=args.xscale)

    # Cross-run summaries
    plot_time_to_generalization(runs, outdir)

    if args.show:
        plt.show()
