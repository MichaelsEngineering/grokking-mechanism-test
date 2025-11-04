#!/usr/bin/env python3
import argparse
import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset


# ----------------------
# Utils
# ----------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def deep_update(d: Dict[str, Any], dotted_key: str, value: str):
    """Update nested dict with dotted key path and auto-cast numbers/bools."""

    def autocast(v: str):
        if v.lower() in {"true", "false"}:
            return v.lower() == "true"
        try:
            if "." in v:
                return float(v)
            return int(v)
        except ValueError:
            return v

    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = autocast(value)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ----------------------
# Dataset: Modular addition (hash split)
# ----------------------
class ModularDataset(Dataset):
    def __init__(
        self,
        N: int,
        op: str,
        train_fraction: float,
        seed: int,
        split: str,
        role: str,
        cache: bool = True,
        val_fraction: float = 0.1,
    ):
        assert split == "hash", "Only split=hash supported in this baseline"
        assert op in {"add", "mul"}
        self.N, self.op = N, op
        self.role = role  # "train" | "val" | "test"
        # Build all pairs (a,b) in ℤ_N × ℤ_N
        pairs = [(a, b) for a in range(N) for b in range(N)]

        # Deterministic hash in [0,1)
        def h(a, b):
            x = (a * 1315423911) ^ (b * 2654435761) ^ 1337
            x &= 0xFFFFFFFF
            return x / 0x100000000

        train_mask = [h(a, b) < train_fraction for (a, b) in pairs]
        train_pairs = [p for p, m in zip(pairs, train_mask) if m]
        test_pairs = [p for p, m in zip(pairs, train_mask) if not m]
        # val is a slice of train (deterministic)
        k_val = max(1, int(len(train_pairs) * val_fraction))
        self.val_pairs = train_pairs[:k_val]
        self.train_pairs = train_pairs[k_val:]
        self.test_pairs = test_pairs
        if role == "train":
            self.pairs = self.train_pairs
        elif role == "val":
            self.pairs = self.val_pairs
        else:
            self.pairs = self.test_pairs

    def __len__(self):
        return len(self.pairs)

    def _label(self, a, b):
        if self.op == "add":
            return (a + b) % self.N
        else:
            return (a * b) % self.N

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        y = self._label(a, b)
        # one-hot for a and b, then concat → 2N-dim
        xa = torch.zeros(self.N)
        xa[a] = 1.0
        xb = torch.zeros(self.N)
        xb[b] = 1.0
        x = torch.cat([xa, xb], dim=0)  # [2N]
        return x, torch.tensor(y, dtype=torch.long)


# ----------------------
# Model: simple MLP
# ----------------------
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        out_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        act = nn.ReLU if activation == "relu" else nn.GELU
        dims = [input_dim] + [hidden_dim] * (layers) + [out_dim]
        blocks = []
        for i in range(len(dims) - 2):
            blocks += [nn.Linear(dims[i], dims[i + 1]), act()]
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
        blocks.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


# ----------------------
# Scheduler: cosine with warmup
# ----------------------
class CosineWithWarmup:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):
        self.opt = optimizer
        self.warmup = max(0, warmup_steps)
        self.total = max(1, total_steps)
        self.min_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.t = 0

    def step(self):
        self.t += 1
        for i, g in enumerate(self.opt.param_groups):
            base = self.base_lrs[i]
            if self.t <= self.warmup:
                lr = base * self.t / max(1, self.warmup)
            else:
                progress = (self.t - self.warmup) / max(1, self.total - self.warmup)
                cos = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = base * (self.min_ratio + (1 - self.min_ratio) * cos)
            g["lr"] = lr


# ----------------------
# Eval
# ----------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        loss_sum += loss.item() * y.numel()
    acc = correct / max(1, total)
    return acc, loss_sum / max(1, total)


# ----------------------
# Grokking detection (logging only)
# ----------------------
def maybe_detect_grokking(state, train_acc, test_acc, step, cfg):
    gd = cfg.get("grokking_detection", {})
    if not gd.get("enabled", False):
        return
    if state.get("grokking_step") is not None:
        return
    test_thr = gd.get("test_acc_threshold", 0.95)
    train_thr = gd.get("train_acc_threshold", 0.999)
    persist = gd.get("min_persistent_steps", 0)
    if train_acc >= train_thr and test_acc >= test_thr:
        # mark candidate and confirm after persistence window
        state["grokking_candidate"] = step
        state["grokking_confirm_at"] = step + persist
    if "grokking_confirm_at" in state and step >= state["grokking_confirm_at"]:
        state["grokking_step"] = state["grokking_candidate"]


# ----------------------
# Main
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("overrides", nargs="*", help="dotted overrides like train.total_steps=1000")
    args = ap.parse_args()

    # Load + overrides
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    for ov in args.overrides:
        if "=" in ov:
            k, v = ov.split("=", 1)
            deep_update(cfg, k, v)

    # Shortcuts to subcfgs
    dcfg = cfg.get("dataset", {})
    lcfg = cfg.get("loader", {})
    mcfg = cfg.get("model", {})
    tcfg = cfg.get("train", {})
    scfg = cfg.get("spectral", {})
    logcfg = cfg.get("logging", {})

    seed = int(dcfg.get("seed", 1337))
    set_seed(seed)
    device = get_device()

    # Datasets / Loaders
    N = int(dcfg["N"])
    ds_train = ModularDataset(
        N=N,
        op=dcfg.get("op", "add"),
        train_fraction=float(dcfg.get("train_fraction", 0.1)),
        seed=seed,
        split=dcfg.get("split", "hash"),
        role="train",
        cache=bool(dcfg.get("cache", True)),
    )
    ds_val = ModularDataset(
        N=N,
        op=dcfg.get("op", "add"),
        train_fraction=float(dcfg.get("train_fraction", 0.1)),
        seed=seed,
        split=dcfg.get("split", "hash"),
        role="val",
        cache=bool(dcfg.get("cache", True)),
    )
    ds_test = ModularDataset(
        N=N,
        op=dcfg.get("op", "add"),
        train_fraction=float(dcfg.get("train_fraction", 0.1)),
        seed=seed,
        split=dcfg.get("split", "hash"),
        role="test",
        cache=bool(dcfg.get("cache", True)),
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=int(lcfg.get("batch_size", 512)),
        shuffle=True,
        num_workers=int(lcfg.get("num_workers", 2)),
        pin_memory=bool(lcfg.get("pin_memory", True)),
    )
    dl_val = DataLoader(ds_val, batch_size=4096, shuffle=False, num_workers=2)
    dl_test = DataLoader(ds_test, batch_size=4096, shuffle=False, num_workers=2)

    # Model
    input_dim = 2 * N
    model = MLP(
        input_dim=input_dim,
        hidden_dim=int(mcfg.get("hidden_dim", 256)),
        layers=int(mcfg.get("layers", 3)),
        out_dim=N,
        activation=mcfg.get("activation", "relu"),
        dropout=float(mcfg.get("dropout", 0.0)),
    ).to(device)

    # Optimizer
    use_adamw = tcfg.get("optimizer", "adamw").lower() == "adamw"
    wd = float(tcfg.get("weight_decay", 1e-2))
    if use_adamw:
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=float(tcfg.get("lr", 3e-4)),
            betas=tuple(tcfg.get("betas", [0.9, 0.999])),
            eps=float(tcfg.get("eps", 1e-8)),
            weight_decay=wd,
        )
    else:
        optim = torch.optim.Adam(model.parameters(), lr=float(tcfg.get("lr", 3e-4)))

    # Scheduler
    total_steps = int(tcfg.get("total_steps", 10000))
    sched_cfg = tcfg.get(
        "scheduler", {"name": "cosine_with_warmup", "warmup_steps": 2000, "min_lr_ratio": 0.05}
    )
    if sched_cfg.get("name", "").lower() == "cosine_with_warmup":
        scheduler = CosineWithWarmup(
            optim,
            warmup_steps=int(sched_cfg.get("warmup_steps", 2000)),
            total_steps=total_steps,
            min_lr_ratio=float(sched_cfg.get("min_lr_ratio", 0.05)),
        )
    else:
        scheduler = None

    # Logging
    out_dir = logcfg.get("out_dir", f"runs/modular_addition")
    ensure_dir(out_dir)
    metrics_path = os.path.join(out_dir, logcfg.get("metrics_csv_name", "metrics.csv"))
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    fieldnames = ["step", "train_loss", "train_acc", "val_acc", "test_acc"]
    # Optional placeholders for spectral metrics to keep schema stable
    if scfg.get("compute", False):
        fieldnames += ["spectral_low_frac", "spectral_entropy"]
        print(
            "[info] Spectral metrics are not implemented in this minimal baseline; values will be empty."
        )
    writer = None

    # Train loop
    model.train()
    grad_clip = float(tcfg.get("grad_clip_norm", 0.0))
    log_every = int(tcfg.get("log_interval_steps", 200))
    eval_every = int(tcfg.get("eval_every_steps", 2000))

    state = {"grokking_step": None}
    for step in range(1, total_steps + 1):
        try:
            x, y = next(_train_iter)  # reuse iterator if exists
        except NameError:
            _train_iter = iter(dl_train)
            x, y = next(_train_iter)
        except StopIteration:
            _train_iter = iter(dl_train)
            x, y = next(_train_iter)

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        if scheduler:
            scheduler.step()

        # Train accuracy on this batch (cheap signal)
        with torch.no_grad():
            train_acc_batch = (logits.argmax(1) == y).float().mean().item()

        # Periodic eval + write
        if step % eval_every == 0 or step == total_steps:
            val_acc, _ = evaluate(model, dl_val, device)
            test_acc, _ = evaluate(model, dl_test, device)
            maybe_detect_grokking(state, train_acc_batch, test_acc, step, cfg)

            # Write metrics row
            if writer is None:
                newfile = not os.path.exists(metrics_path)
                f = open(metrics_path, "a", newline="")
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if newfile:
                    writer.writeheader()
                metrics_fh = f  # keep open
            row = {
                "step": step,
                "train_loss": float(loss.item()),
                "train_acc": float(train_acc_batch),
                "val_acc": float(val_acc),
                "test_acc": float(test_acc),
            }
            if scfg.get("compute", False):
                row["spectral_low_frac"] = ""
                row["spectral_entropy"] = ""
            writer.writerow(row)
            metrics_fh.flush()

            # Optional console log
            gs = state.get("grokking_step")
            tag = f" | grok@{gs}" if gs is not None else ""
            print(
                f"[step {step}] loss={loss.item():.4f} train_acc={train_acc_batch:.3f} val_acc={val_acc:.3f} test_acc={test_acc:.3f}{tag}"
            )

    # Finalize: write grokking_step to a small sidecar file if detected
    if state.get("grokking_step") is not None:
        with open(os.path.join(out_dir, "grokking.json"), "w") as f:
            f.write('{"grokking_step": %d}\n' % state["grokking_step"])


if __name__ == "__main__":
    main()
