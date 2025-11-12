"""Analytic norm-minimization dynamics test inspired by Musat (2025)."""

from __future__ import annotations

from collections import defaultdict
from typing import TypedDict

import torch
from torch.nn.functional import cosine_similarity

from src.scripts.train import ModularDataset


class DatasetCfg(TypedDict):
    N: int
    op: str
    train_fraction: float
    split: str
    seed: int
    val_fraction: float


def test_norm_min_dynamics() -> None:
    modulus = 29
    dataset_cfg: DatasetCfg = {
        "N": modulus,
        "op": "add",
        "train_fraction": 0.06,
        "split": "hash",
        "seed": 2025,
        "val_fraction": 0.1,
    }
    train_ds = ModularDataset(role="train", **dataset_cfg)
    test_ds = ModularDataset(role="test", **dataset_cfg)

    train_pairs = list(train_ds.pairs)
    test_pairs = list(test_ds.pairs)
    assert train_pairs, "expected non-empty training split"

    train_indices = torch.tensor([a * modulus + b for a, b in train_pairs], dtype=torch.long)
    train_sums = torch.tensor([(a + b) % modulus for a, b in train_pairs], dtype=torch.long)
    test_sums = torch.tensor([(a + b) % modulus for a, b in test_pairs], dtype=torch.long)

    sum_to_pair_indices: defaultdict[int, list[int]] = defaultdict(list)
    for pair_idx, sum_idx in zip(train_indices.tolist(), train_sums.tolist()):
        sum_to_pair_indices[sum_idx].append(pair_idx)

    # Ensure we leave at least one sum unconstrained so analytic dynamics has something to learn.
    assert len(sum_to_pair_indices) < modulus

    dtype = torch.float64
    target = torch.eye(modulus, dtype=dtype)
    total_pairs = modulus * modulus
    sum_weights = torch.zeros((modulus, modulus), dtype=dtype)
    pair_weights = torch.zeros((total_pairs, modulus), dtype=dtype)

    torch.manual_seed(0)
    noise_scale = 3.0
    for sum_idx, pair_list in sum_to_pair_indices.items():
        noise = torch.randn(modulus, dtype=dtype) * noise_scale
        noise[sum_idx] = torch.randn((), dtype=dtype) * noise_scale
        base_vec = target[sum_idx] + noise
        for pair_idx in pair_list:
            pair_weights[pair_idx] = base_vec
        sum_weights[sum_idx] = target[sum_idx] - base_vec

    test_labels = test_sums.clone()

    def eval_test_accuracy() -> float:
        logits = sum_weights[test_sums]
        preds = logits.argmax(dim=1)
        return (preds == test_labels).float().mean().item()

    analytic_steps = 5000
    analytic_lr = 1e-3
    decay = 1.0 - analytic_lr
    test_history: list[float] = []

    for _ in range(analytic_steps):
        test_history.append(eval_test_accuracy())
        pair_weights[train_indices] = pair_weights[train_indices] * decay
        sum_weights *= decay
        for sum_idx, pair_list in sum_to_pair_indices.items():
            pair_vec = pair_weights[pair_list[0]]
            sum_weights[sum_idx] = target[sum_idx] - pair_vec

    final_acc = eval_test_accuracy()
    test_history.append(final_acc)

    train_logits = sum_weights[train_sums] + pair_weights[train_indices]
    train_targets = target[train_sums]
    train_loss = torch.mean((train_logits - train_targets) ** 2).item()

    fft_coeffs = torch.fft.fft(sum_weights, dim=0)
    real_norms = torch.abs(fft_coeffs).mean(dim=1).real
    norm_ratio = float(real_norms.std() / real_norms.mean())
    phase_alignment = float(
        cosine_similarity(fft_coeffs.real.flatten(), fft_coeffs.imag.flatten(), dim=0).abs()
    )

    assert train_loss < 1e-6
    assert test_history[0] < 0.2  # delayed generalization signal
    assert final_acc > 0.95
    assert final_acc - test_history[len(test_history) // 2] > 0.02
    assert norm_ratio < 0.05
    assert phase_alignment < 0.1
