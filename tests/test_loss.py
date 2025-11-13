from __future__ import annotations

import pytest
import torch

from src.grokking_mechanism.loss import LaplacianEnergyPenalty


def test_laplacian_penalty_smoke() -> None:
    """Basic smoke test for the Laplacian penalty calculation."""
    penalty = LaplacianEnergyPenalty(modulus=5, alpha=0.1)
    logits = torch.randn(25, 10)
    loss = penalty(logits)
    assert loss.item() > 0
    assert loss.shape == ()


def test_laplacian_penalty_invalid_inputs() -> None:
    """Check that invalid initialization values raise errors."""
    with pytest.raises(ValueError, match="Modulus must be a positive integer"):
        LaplacianEnergyPenalty(modulus=0, alpha=0.1)
    with pytest.raises(ValueError, match="Penalty strength alpha must be positive"):
        LaplacianEnergyPenalty(modulus=5, alpha=0.0)
    with pytest.raises(ValueError, match="Penalty strength alpha must be positive"):
        LaplacianEnergyPenalty(modulus=5, alpha=-0.1)


def test_laplacian_penalty_reproducible() -> None:
    """Ensure the penalty calculation is deterministic."""
    penalty = LaplacianEnergyPenalty(modulus=7, alpha=0.05)
    logits = torch.ones(49, 3) * 0.5
    loss1 = penalty(logits)
    loss2 = penalty(logits)
    assert torch.allclose(loss1, loss2)


def test_laplacian_penalty_smooth_vs_rough() -> None:
    """Verify that smoother logits yield a lower penalty."""
    modulus = 11
    alpha = 1.0
    penalty = LaplacianEnergyPenalty(modulus=modulus, alpha=alpha)

    # Smooth logits (low-frequency)
    coords = torch.arange(modulus, dtype=torch.float32)
    grid_a, grid_b = torch.meshgrid(coords, coords, indexing="ij")
    smooth_logits = torch.cos(2 * torch.pi * (grid_a + grid_b) / modulus)
    smooth_logits = smooth_logits.reshape(-1, 1)

    # Rough logits (high-frequency)
    rough_logits = torch.cos(2 * torch.pi * (5 * grid_a) / modulus)
    rough_logits = rough_logits.reshape(-1, 1)

    loss_smooth = penalty(smooth_logits)
    loss_rough = penalty(rough_logits)

    assert loss_smooth.item() < loss_rough.item()
