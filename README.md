# Grokking Mechanism Test

**Exploring the Geometric Grokking Hypothesis** — that delayed generalization in neural networks arises from a *geometric phase transition* minimizing low-frequency energy of an implicit graph Laplacian in learned representations, rather than from weight decay or circuit efficiency alone.
*(Inspired by “Geometric GROKKING Unlocked & Explained,” AI Explained, 2024.)*

---

## Overview

This repository provides the scaffolding for reproducible tests of the **geometric grokking hypothesis**, focusing on controlled toy-tasks drawn from deep learning and reinforcement learning literature.
The experiments are designed to measure *spectral energy redistribution* and *representation smoothness* as networks transition from memorization to generalization.

---

## Repository Structure

grokking-mechanism-test/
├── README.md
├── LICENSE
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── requirements-*.txt
├── pyproject.toml
├── scripts/
│ ├── train.py # (Planned) entry point for running experiments
│ ├── evaluate.py # (Planned) evaluation utilities
│ └── visualize.py # (Planned) plotting/analysis helpers
├── src/
│ └── grokking_mechanism_test/ # core model + training logic (to be implemented)
├── tests/
│ └── test_backend_smoke.py # basic backend compatibility check
├── dist/ # built wheels (if packaged)
├── grokking-mech-env/ # optional local venv (ignore in docs)
└── .github/workflows/ci.yml # CI configuration


---

## Environment Setup

### Prerequisites
- **Python:** 3.11
- **Hardware:** NVIDIA GPU recommended
- **Framework:** PyTorch (default backend)

### Installation

# Clone and enter

```bash
git clone https://github.com/MichaelsEngineering/grokking-mechanism-test.git
cd grokking-mechanism-test
```

# Create virtual environment (optional)

```bash
python3.11 -m venv grokking-mech-env
source grokking-mech-env/bin/activate
```

# Install dependencies

```bash
pip install -r requirements-pytorch.txt
```

### Usage (Placeholder Examples)

The scripts are scaffolds for now — implementation in progress.

Training

```bash
python scripts/train.py --config configs/modular_addition.yaml
```

Evaluation

```bash
python scripts/evaluate.py --checkpoint runs/modular_addition/best.pt
```

Visualization

```bash
python scripts/visualize.py --run runs/modular_addition/
```

Expected outputs (once implemented):

runs/<experiment>/metrics.csv — epoch-level accuracy, loss, and spectral energy

runs/<experiment>/plots/ — Laplacian energy spectra and generalization curves

runs/<experiment>/checkpoints/ — model weights

## Planned Experiments

| **Experiment** | **Description** | **Metrics** | **Expected Outcome** | **Status** |
|----------------|-----------------|--------------|----------------------|-------------|
| **Spectral Energy Shift Test** | Track how learned representation energy moves from high- to low-frequency Laplacian modes during training. | Low-frequency energy ratio, spectral entropy, validation accuracy. | Gradual transfer of representational energy to smoother modes correlates with grokking onset. | 🚧 Planned |
| **Laplacian Energy Penalty Ablation** | Add or remove an explicit Laplacian energy regularizer to test causal role of geometric smoothness. | Grokking time, accuracy gap, mean feature Laplacian energy. | Models with controlled low-energy bias should grok faster or more consistently. | 🚧 Planned |
| **Weight-Decay Baseline** | Compare identical models trained with classic L2 weight decay. | Validation accuracy vs. epoch, parameter norm trajectory. | Weight decay alone reproduces some but not all smoothness signatures. | 🚧 Planned |
| **Circuit-Efficiency Proxy** | Test hypothesis that grokking stems from efficient sub-circuit selection rather than geometric reorganization. | Parameter sparsity, FLOPs, accuracy. | Improvements appear without geometric reorganization, distinguishing competing theories. | 🚧 Planned |
| **Synthetic Modular Arithmetic** | Minimal synthetic task (e.g., mod-N addition) for measuring grokking transition. | Accuracy, loss, spectral energy distribution. | Clear delayed generalization and spectral phase transition. | 🚧 Planned |
| **Parity & Sequence Copy Tasks** | RL-style toy domains from small-scale deep-RL benchmarks. | Reward, accuracy, smoothness metrics. | Reinforces that geometric smoothness generalizes beyond simple arithmetic tasks. | 🚧 Planned |


# Research Context

This repository seeks to provide empirical footing for the geometric grokking hypothesis, connecting observed generalization delays to measurable changes in representation geometry.
It aims to complement other explanations (regularization, sparsity, or circuit efficiency) by introducing tools to visualize phase transitions in representation manifolds.

# Citation

If you use or reference this repository, please cite:

@software{mcbride2025_grokking_mechanism_test,
  author = {Michael McBride},
  title = {grokking-mechanism-test: Geometric Grokking Hypothesis Experiments},
  year = {2025},
  url = {https://github.com/MichaelsEngineering/grokking-mechanism-test},
  version = {1.0}
}
