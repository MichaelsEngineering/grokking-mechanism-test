import math

import pytest
import torch

from src.grokking_mechanism.spectral import SpectralBand, ToroidalSpectralAnalyzer, synthetic_logits


def test_low_frequency_mask_fraction_mode():
    analyzer = ToroidalSpectralAnalyzer(
        17,
        SpectralBand(mode="fraction", value=0.25, include_zero=True),
    )
    mask = analyzer.low_mask
    assert mask.shape == (17, 17)
    # 25% of the modes should be marked low frequency (within rounding).
    assert pytest.approx(mask.mean().item(), rel=0.1) == 0.25


def test_synthetic_logits_shift_energy():
    modulus = 19
    analyzer = ToroidalSpectralAnalyzer(
        modulus,
        SpectralBand(mode="fraction", value=0.1, include_zero=True),
    )
    early = analyzer.measure_logits(synthetic_logits(modulus, step=1, total_steps=50, seed=7))
    late = analyzer.measure_logits(synthetic_logits(modulus, step=50, total_steps=50, seed=7))

    assert math.isfinite(early.low_fraction)
    assert math.isfinite(late.low_fraction)
    assert late.low_fraction > early.low_fraction
    assert late.entropy <= early.entropy + 1e-6


def test_analyzer_validates_logits_shape():
    analyzer = ToroidalSpectralAnalyzer(3, SpectralBand())
    bad_logits = torch.zeros(3, 3)
    with pytest.raises(ValueError):
        analyzer.measure_logits(bad_logits)
