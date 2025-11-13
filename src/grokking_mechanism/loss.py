from __future__ import annotations

import torch

from .spectral import _normalized_laplacian


class LaplacianEnergyPenalty:
    """Computes the Laplacian energy of model logits to penalize high-frequency representations."""

    def __init__(self, *, modulus: int, alpha: float) -> None:
        if modulus <= 0:
            raise ValueError("Modulus must be a positive integer.")
        if not (alpha > 0):
            raise ValueError("Penalty strength alpha must be positive.")

        self.modulus = modulus
        self.alpha = float(alpha)
        self.eigenvalues = _normalized_laplacian(self.modulus)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculate the mean Laplacian energy of the logits grid.

        Args:
            logits: A tensor of shape (N*N, C) representing the output logits
                    for each of the N*N input pairs.

        Returns:
            A scalar tensor representing the mean Laplacian energy penalty.
        """
        num_classes = logits.shape[1]
        grid = logits.view(self.modulus, self.modulus, num_classes).permute(2, 0, 1)
        freq = torch.fft.fftn(grid, dim=(-2, -1))
        power = freq.abs().pow(2)
        spectrum = power.sum(dim=0).real
        total_energy = spectrum.sum().clamp_min(1e-9)

        weighted_energy = (spectrum * self.eigenvalues).sum()
        mean_energy = weighted_energy / total_energy
        return self.alpha * mean_energy
