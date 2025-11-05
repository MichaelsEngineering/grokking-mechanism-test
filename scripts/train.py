#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

criterion = nn.CrossEntropyLoss()


# ----------------------
# Utils
# ----------------------
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy, and torch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Ensure each dataloader worker has a different yet reproducible seed."""

    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def resolve_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def deep_update(d: Dict[str, Any], dotted_key: str, value: str) -> None:
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


def ensure_dir(p: str | os.PathLike[str]) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------
# Dataset: Modular addition (hash split)
# ----------------------
class ModularDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        N: int,
        op: str,
        train_fraction: float,
        seed: int,
        split: str,
        role: str,
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

    def __len__(self) -> int:
        return len(self.pairs)

    def _label(self, a: int, b: int) -> int:
        if self.op == "add":
            return (a + b) % self.N
        else:
            return (a * b) % self.N

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        act_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "identity": nn.Identity,
        }
        if activation not in act_map:
            raise ValueError(f"Unknown activation: {activation}")
        Act = act_map[activation]
        dims = [input_dim] + [hidden_dim] * (layers) + [out_dim]
        blocks = []
        for i in range(len(dims) - 2):
            blocks += [nn.Linear(dims[i], dims[i + 1]), Act()]
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
        blocks.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


# ----------------------
# Eval
# ----------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Tuple[float, float]:
    was_training = model.training
    model.eval()

    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / max(1, total)
    avg_loss = loss_sum / max(1, total)

    if was_training:
        model.train()

    return acc, avg_loss


# ----------------------
# Grokking detection (logging only)
# ----------------------
def maybe_detect_grokking(
    state: Dict[str, Any], train_acc: float, test_acc: float, step: int, cfg: Dict[str, Any]
) -> None:
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
def main() -> None:
    ap = argparse.ArgumentParser(description="Train an MLP on modular arithmetic tasks.")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument(
        "--device", type=str, default=None, help="Force device string (e.g. cpu, cuda:0)"
    )
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic algorithms (overrides train.deterministic).",
    )
    ap.add_argument("overrides", nargs="*", help="dotted overrides like train.total_steps=1000")
    args = ap.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_config(config_path)
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
    device = get_device(args.device)
    deterministic = bool(tcfg.get("deterministic", False) or args.deterministic)
    set_seed(seed, deterministic=deterministic)
    print(f"[setup] device={device} seed={seed} deterministic={deterministic}")

    # Datasets / Loaders
    N = int(dcfg["N"])
    dataset_kwargs: dict[str, Any] = {
        "N": N,
        "op": dcfg.get("op", "add"),
        "train_fraction": float(dcfg.get("train_fraction", 0.1)),
        "seed": seed,
        "split": dcfg.get("split", "hash"),
    }
    ds_train = ModularDataset(role="train", **dataset_kwargs)
    ds_val = ModularDataset(role="val", **dataset_kwargs)
    ds_test = ModularDataset(role="test", **dataset_kwargs)
    print(
        f"[data] train={len(ds_train)} val={len(ds_val)} test={len(ds_test)} "
        f"batch={lcfg.get('batch_size', 512)}"
    )

    batch_size = int(lcfg.get("batch_size", 512))
    eval_batch_size = int(lcfg.get("eval_batch_size", 4096))
    num_workers = int(lcfg.get("num_workers", 2))
    workers_disabled = False
    if num_workers > 0:
        try:
            test_lock = mp.Lock()
            test_lock.acquire()
            test_lock.release()
        except (OSError, PermissionError):
            print(
                "[data] Disabling DataLoader multiprocessing; semaphore creation was denied. "
                "Set loader.num_workers=0 to silence this message."
            )
            num_workers = 0
            workers_disabled = True
    default_eval_workers = max(1, num_workers // 2)
    eval_workers = int(lcfg.get("eval_num_workers", default_eval_workers))
    if workers_disabled and eval_workers > 0:
        print("[data] Forcing eval_num_workers=0 to match train loader fallback.")
        eval_workers = 0
    pin_memory = bool(lcfg.get("pin_memory", True)) and device.type == "cuda"
    persistent_workers = bool(lcfg.get("persistent_workers", num_workers > 0)) and num_workers > 0
    generator = get_dataloader_generator(seed)

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        persistent_workers=persistent_workers,
    )
    common_eval_kwargs: dict[str, Any] = {
        "shuffle": False,
        "num_workers": eval_workers,
        "pin_memory": pin_memory,
    }
    if eval_workers > 0:
        common_eval_kwargs["worker_init_fn"] = seed_worker
    dl_val = DataLoader(ds_val, batch_size=eval_batch_size, **common_eval_kwargs)
    dl_test = DataLoader(ds_test, batch_size=eval_batch_size, **common_eval_kwargs)
    steps_per_epoch = max(1, len(dl_train))
    grad_accum = int(tcfg.get("grad_accum", 1))
    epochs = tcfg.get("epochs")
    default_total_steps = (
        int(epochs) * max(1, steps_per_epoch // grad_accum) if epochs is not None else 10000
    )
    total_steps = int(tcfg.get("total_steps", default_total_steps))

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
    scheduler = None
    sched_cfg = tcfg.get("scheduler", {})
    if sched_cfg.get("name", "").lower() == "cosine_with_warmup":
        warmup_steps = int(sched_cfg.get("warmup_steps", 2000))
        min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.05))

        cosine_steps = max(1, total_steps - warmup_steps)
        base_lrs = [g["lr"] for g in optim.param_groups]
        eta_min = min_lr_ratio * min(base_lrs)

        schedulers = []
        milestones: list[int] = []
        if warmup_steps > 0:
            warmup = LinearLR(
                optim,
                start_factor=1e-9,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            schedulers.append(warmup)
            milestones.append(warmup_steps)
        cosine = CosineAnnealingLR(optim, T_max=cosine_steps, eta_min=eta_min)
        schedulers.append(cosine)

        scheduler = SequentialLR(
            optim,
            schedulers=schedulers,
            milestones=milestones if milestones else [],
        )

    # Logging
    out_dir = resolve_path(logcfg.get("out_dir", "runs/modular_addition"))
    ensure_dir(out_dir)
    metrics_path = out_dir / logcfg.get("metrics_csv_name", "metrics.csv")
    config_copy_path = out_dir / "config_used.yaml"
    with config_copy_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    fieldnames = ["step", "train_loss", "train_acc", "val_acc", "test_acc"]
    # Optional placeholders for spectral metrics to keep schema stable
    if scfg.get("compute", False):
        fieldnames += ["spectral_low_frac", "spectral_entropy"]
        print(
            "[info] Spectral metrics are not implemented in this minimal baseline; values will be empty."
        )

    # Train loop
    model.train()
    grad_clip = float(tcfg.get("grad_clip_norm", 0.0))
    eval_every = int(tcfg.get("eval_every_steps", 2000))

    state: Dict[str, Any] = {"grokking_step": None}
    train_iter = iter(dl_train)
    metrics_fh = None
    writer: csv.DictWriter[str] | None = None
    try:
        metrics_fh = metrics_path.open("a", newline="")
        writer = csv.DictWriter(metrics_fh, fieldnames=fieldnames)
        if metrics_fh.tell() == 0:
            writer.writeheader()

        for step in range(1, total_steps + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(dl_train)
                batch = next(train_iter)

            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
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
                    f"[step {step}] loss={loss.item():.4f} train_acc={train_acc_batch:.3f} "
                    f"val_acc={val_acc:.3f} test_acc={test_acc:.3f}{tag}"
                )
    finally:
        if metrics_fh is not None:
            metrics_fh.close()

    # Finalize: write grokking_step to a small sidecar file if detected
    if state.get("grokking_step") is not None:
        grokking_path = out_dir / "grokking.json"
        grokking_path.write_text(
            f'{{"grokking_step": {state["grokking_step"]}}}\n', encoding="utf-8"
        )


if __name__ == "__main__":
    main()
