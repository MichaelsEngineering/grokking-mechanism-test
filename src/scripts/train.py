"""Minimal training utilities for the smoke tests.

The goal of this module is *not* to provide a full training pipeline yet –
we just need enough structure for the current tests:

* ``ModularDataset`` implements the deterministic hash split that the
  dataset tests expect.
* ``main`` loads a YAML config (with a very small fallback parser),
  applies dotted overrides, and then produces lightweight CSV artifacts
  so the smoke test has something to assert against.

The actual optimization loop will be filled in later.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Sequence

import torch
from torch.utils.data import Dataset

try:  # Optional – we do not require PyYAML in tests, but use it when present.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback parser handles the rest.
    yaml = None


class ModularDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Deterministic modular arithmetic dataset used throughout the tests."""

    def __init__(
        self,
        *,
        N: int,
        op: str,
        train_fraction: float,
        seed: int,
        split: str,
        role: str,
        val_fraction: float = 0.1,
    ) -> None:
        if split != "hash":
            raise ValueError("Only the deterministic hash split is supported")
        if op not in {"add", "mul"}:
            raise ValueError("op must be 'add' or 'mul'")

        self.N = int(N)
        self.op = op
        self.role = role
        pairs = [(a, b) for a in range(self.N) for b in range(self.N)]

        def _hash_pair(a: int, b: int) -> float:
            x = (a * 1315423911) ^ (b * 2654435761) ^ int(seed)
            x &= 0xFFFFFFFF
            return x / 0x100000000

        train_mask = [_hash_pair(a, b) < float(train_fraction) for (a, b) in pairs]
        pool = [p for p, keep in zip(pairs, train_mask) if keep]
        self.test_pairs = [p for p, keep in zip(pairs, train_mask) if not keep]
        k_val = max(1, int(len(pool) * float(val_fraction)))
        self.val_pairs = pool[:k_val]
        self.train_pairs = pool[k_val:]

        if role == "train":
            self.pairs = self.train_pairs
        elif role == "val":
            self.pairs = self.val_pairs
        else:
            self.pairs = self.test_pairs

    # Public helpers -----------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.pairs)

    def _label(self, a: int, b: int) -> int:
        if self.op == "add":
            return (a + b) % self.N
        return (a * b) % self.N

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = self.pairs[idx]
        label = self._label(a, b)
        vec = torch.zeros(self.N * 2, dtype=torch.float32)
        vec[a] = 1.0
        vec[self.N + b] = 1.0
        return vec, torch.tensor(label, dtype=torch.long)


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:  # pragma: no cover - depends on optional dep
        return yaml.safe_load(text)
    return _basic_yaml_parse(text)


def _basic_yaml_parse(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[tuple[int, MutableMapping[str, Any]]] = [(-1, root)]

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            new_dict: Dict[str, Any] = {}
            parent[key] = new_dict
            stack.append((indent, new_dict))
            continue
        parent[key] = _autocast(value)
    return root


def _autocast(value: str) -> Any:
    cleaned = value.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        inner = cleaned[1:-1].strip()
        if not inner:
            return []
        return [_autocast(part.strip()) for part in inner.split(",")]
    lowered = cleaned.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in cleaned for ch in [".", "e", "E"]):
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return cleaned


def apply_override(cfg: MutableMapping[str, Any], dotted: str) -> None:
    if "=" not in dotted:
        raise ValueError(f"Override '{dotted}' must look like key=value")
    key, raw = dotted.split("=", 1)
    parts = key.split(".")
    target = cfg
    for part in parts[:-1]:
        target = target.setdefault(part, {})  # type: ignore[assignment]
        if not isinstance(target, MutableMapping):
            raise ValueError(f"Cannot override inside non-mapping for '{key}'")
    target[parts[-1]] = _autocast(raw)


# -----------------------------------------------------------------------------
# Stub training run
# -----------------------------------------------------------------------------


def run_stub_training(cfg: Dict[str, Any], config_path: Path) -> None:
    train_cfg = cfg.setdefault("train", {})
    total_steps = int(train_cfg.get("total_steps", 1))
    logging_cfg = cfg.setdefault("logging", {})
    default_out = Path("runs") / config_path.stem
    out_dir = Path(logging_cfg.get("out_dir", default_out))
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.csv"
    fieldnames = ["step", "train_loss", "train_acc", "val_acc", "test_acc"]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(1, total_steps + 1):
            progress = step / max(total_steps, 1)
            row = {
                "step": step,
                "train_loss": round(1.0 - 0.5 * progress, 6),
                "train_acc": round(0.1 + 0.8 * progress, 6),
                "val_acc": round(0.05 + 0.85 * progress, 6),
                "test_acc": round(0.04 + 0.9 * progress, 6),
            }
            writer.writerow(row)

    config_copy = out_dir / "config_used.yaml"
    if yaml is not None:
        dumped = yaml.safe_dump(cfg, sort_keys=True)
        config_copy.write_text(dumped, encoding="utf-8")
    else:  # pragma: no cover - optional path
        config_copy.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stub trainer for smoke tests")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional dotted overrides like train.total_steps=50",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    for override in args.overrides:
        apply_override(cfg, override)
    run_stub_training(cfg, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
