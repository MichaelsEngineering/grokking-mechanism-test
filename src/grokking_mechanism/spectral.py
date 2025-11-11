from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import torch

BandMode = Literal["fraction", "count", "cutoff"]


@dataclass(frozen=True)
class SpectralBand:
    """Configuration for selecting low-frequency Laplacian modes."""

    mode: BandMode = "fraction"
    value: float = 0.10
    include_zero: bool = True


@dataclass
class SpectralResult:
    """Container for spectral metrics computed from logits."""

    low_fraction: float
    entropy: float
    spectrum: torch.Tensor
    per_dim_low: torch.Tensor


def _normalized_laplacian(modulus: int) -> torch.Tensor:
    coords = torch.arange(modulus, dtype=torch.float32)
    angles = 2 * math.pi * coords / float(modulus)
    cos_vals = torch.cos(angles)
    return 1.0 - 0.5 * (cos_vals.view(-1, 1) + cos_vals.view(1, -1))


def _low_freq_mask(eigenvalues: torch.Tensor, band: SpectralBand) -> torch.Tensor:
    flat = eigenvalues.flatten()
    eps = 1e-9

    if band.mode == "fraction":
        total = flat.numel()
        keep = max(1, min(total, int(math.ceil(total * float(band.value)))))
        threshold = torch.sort(flat).values[keep - 1].item()
        mask = eigenvalues <= (threshold + eps)
    elif band.mode == "count":
        keep = max(1, min(flat.numel(), int(band.value)))
        threshold = torch.sort(flat).values[keep - 1].item()
        mask = eigenvalues <= (threshold + eps)
    elif band.mode == "cutoff":
        mask = eigenvalues <= float(band.value)
    else:  # pragma: no cover - guarded by typing
        raise ValueError(f"Unknown spectral band mode '{band.mode}'")

    if not band.include_zero:
        mask = mask & (~torch.isclose(eigenvalues, torch.tensor(0.0)))

    return mask.to(eigenvalues.dtype)


class ToroidalSpectralAnalyzer:
    """Analytical Laplacian calculator for the toroidal grid C_N ☐ C_N."""

    def __init__(self, modulus: int, band: SpectralBand) -> None:
        if modulus <= 0:
            raise ValueError("modulus must be positive")
        self.modulus = int(modulus)
        self.band = band
        self.eigenvalues = _normalized_laplacian(self.modulus)
        self.low_mask = _low_freq_mask(self.eigenvalues, band)

    def measure_logits(self, logits: torch.Tensor) -> SpectralResult:
        if logits.ndim != 2:
            raise ValueError("logits must be a 2D tensor shaped (N^2, num_classes)")
        expected = self.modulus * self.modulus
        if logits.shape[0] != expected:
            raise ValueError(f"logits must contain {expected} rows for modulus {self.modulus}")
        if logits.shape[1] <= 0:
            raise ValueError("logits must contain at least one class dimension")

        num_classes = logits.shape[1]
        grid = logits.view(self.modulus, self.modulus, num_classes).permute(2, 0, 1)
        freq = torch.fft.fftn(grid, dim=(-2, -1))
        power = freq.abs().pow(2)
        spectrum = power.sum(dim=0).real
        total_energy = spectrum.sum()
        if float(total_energy) <= 0.0:
            total_energy = torch.tensor(1e-9, dtype=spectrum.dtype)

        low_energy = (spectrum * self.low_mask).sum()
        low_fraction = float((low_energy / total_energy).item())

        probs = (spectrum / total_energy).reshape(-1).clamp_min(1e-12)
        entropy = float((-(probs * torch.log(probs))).sum().item())

        numerator = (power * self.low_mask).sum(dim=(-1, -2))
        denominator = power.sum(dim=(-1, -2)).clamp_min(1e-9)
        per_dim_low = numerator / denominator
        return SpectralResult(
            low_fraction=low_fraction,
            entropy=entropy,
            spectrum=spectrum.detach().cpu(),
            per_dim_low=per_dim_low.detach().cpu(),
        )


def synthetic_logits(modulus: int, step: int, total_steps: int, *, seed: int) -> torch.Tensor:
    """Generate smooth-to-rough logits to drive the spectral test during stubs."""

    generator = torch.Generator().manual_seed(seed + step)
    coords = torch.arange(modulus, dtype=torch.float32)
    grid_a, grid_b = torch.meshgrid(coords, coords, indexing="ij")

    low_component = torch.cos(2 * math.pi * (grid_a + grid_b) / float(modulus))
    high_component = torch.cos(2 * math.pi * 5 * grid_a / float(modulus)) + torch.sin(
        2 * math.pi * 7 * grid_b / float(modulus)
    )

    progress = step / max(1, total_steps)
    blend = (1.0 - progress) * high_component + progress * low_component

    class_offsets = torch.linspace(-1.0, 1.0, modulus, dtype=torch.float32)
    logits = blend.unsqueeze(-1) + class_offsets
    noise = torch.randn(logits.shape, generator=generator, dtype=logits.dtype)
    logits = logits + noise * 0.05
    return logits.reshape(-1, modulus)


class SpectralLogger:
    """Coordinates spectral measurements and snapshotting during training."""

    def __init__(
        self,
        *,
        dataset_cfg: Dict,
        train_cfg: Dict,
        spectral_cfg: Dict,
        out_dir: Path,
    ) -> None:
        self.enabled = bool(spectral_cfg.get("compute", False))
        if not self.enabled:
            self.analyzer: Optional[ToroidalSpectralAnalyzer] = None
            return

        modulus = int(dataset_cfg.get("N", 1))
        band = SpectralBand(
            mode=spectral_cfg.get("mode", "fraction"),
            value=float(spectral_cfg.get("value", 0.10)),
            include_zero=bool(spectral_cfg.get("include_zero", True)),
        )
        self.analyzer = ToroidalSpectralAnalyzer(modulus, band)
        self.eval_every = max(1, int(train_cfg.get("eval_every_steps", 1)))
        self.seed = int(spectral_cfg.get("seed", 0))
        self.snapshot_dir = out_dir / "spectral"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_stride = max(1, int(spectral_cfg.get("snapshot_stride", 1)))
        self.projection_samples = spectral_cfg.get("projection_samples")
        self.step_counter = 0

    def maybe_log(self, step: int, total_steps: int) -> Dict[str, float]:
        if not self.enabled or not self.analyzer:
            return {}

        should_eval = (step % self.eval_every == 0) or (step == total_steps)
        if not should_eval:
            return {}

        result = self.analyzer.measure_logits(
            synthetic_logits(self.analyzer.modulus, step, total_steps, seed=self.seed)
        )
        self.step_counter += 1
        if self.step_counter % self.snapshot_stride == 0:
            self._write_snapshot(step, result)

        return {
            "spectral_low_frac": round(result.low_fraction, 6),
            "spectral_entropy": round(result.entropy, 6),
        }

    def _write_snapshot(self, step: int, result: SpectralResult) -> None:
        spectrum_path = self.snapshot_dir / f"spectrum_step{step:06d}.pt"
        torch.save(
            {
                "spectrum": result.spectrum,
                "low_mask": self.analyzer.low_mask.cpu() if self.analyzer else None,
            },
            spectrum_path,
        )
        per_dim_path = self.snapshot_dir / f"per_dim_step{step:06d}.csv"
        limit = None
        if isinstance(self.projection_samples, int) and self.projection_samples > 0:
            limit = min(self.projection_samples, result.per_dim_low.shape[0])
        rows = result.per_dim_low.numpy()
        if limit is not None:
            rows = rows[:limit]

        with per_dim_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dimension", "low_energy_fraction"])
            for idx, value in enumerate(rows):
                writer.writerow([idx, round(float(value), 6)])
