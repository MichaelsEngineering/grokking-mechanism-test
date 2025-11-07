# Grokking Mechanism Test

**Exploring the Geometric Grokking Hypothesis** — that delayed generalization in neural networks arises from a *geometric phase transition* minimizing low-frequency energy of an implicit graph Laplacian in learned representations, rather than from weight decay or circuit efficiency alone.
*(Inspired by “Geometric GROKKING Unlocked & Explained,” Discover AI, 2025.)*

---

## Overview

This repository provides the scaffolding for reproducible tests of the **geometric grokking hypothesis**, focusing on controlled toy-tasks drawn from deep learning and reinforcement learning literature.
The experiments are designed to measure *spectral energy redistribution* and *representation smoothness* as networks transition from memorization to generalization.

---

## Repository Structure

grokking-mechanism-test/ <br>
├── .github/ <br>
│ └── workflows/ <br>
│ └── ci.yml <br>
├── configs/ <br>
│ └── modular_addition.yaml <br>
├── runs/ <br>
├── plots/ <br>
│ └── modular_addition/ <br>
│ ├── config_used.yaml <br>
│ └── metrics.csv <br>
├── src/ <br>
│ ├── init.py <br>
│ ├── scripts/ <br>
│ │ ├── init.py <br>
│ │ ├── evaluate.py <br>
│ │ ├── train.py <br>
│ │ └── visualize.py <br>
│ └── pycache/ <br>
├── tests/ <br>
│ ├── fixtures/mini_run <br>
│ ├── test_backend_smoke.py <br>
│ ├── test_evaluate_fixture.py <br>
│ ├── test_modular_dataset_split.py <br>
│ └── test_visualize.py <br>
├── .gitignore <br>
├── .pre-commit-config.yaml <br>
├── AGENT.md <br>
├── CITATION.cff <br>
├── CODE_OF_CONDUCT.md <br>
├── CONTRIBUTING.md <br>
├── keras.json <br>
├── LICENSE <br>
├── Makefile <br>
├── pyproject.toml <br>
├── README.md <br>
├── requirements.txt <br>
├── requirements-dev.txt <br>
├── requirements-torch.txt <br>
├── requirements-tensorflow.txt <br>
└── requirements-jax.txt <br>

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

# Install dependencies (PyTorch default)

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
# or run: make init
pip install -r requirements-torch.txt or < Alternate backends are still available through the backend-specific files `requirements-jax.txt`/ `requirements-tensorflow.txt`).
```
### Usage

The YAML configuration files are the central control mechanism for the project. They are used by `src.scripts.train` to define and parameterize every aspect of a specific experiment, from data generation to metric computation.

Training

```bash
python -m src.scripts.train --config configs/modular_addition.yaml
```

Evaluation

```bash
python -m src.scripts.evaluate --run-dir runs/modular_addition
# or, if you only have the CSV path:
python -m src.scripts.evaluate --metrics runs/modular_addition/metrics.csv
```

Visualization

```bash
python -m src.scripts.visualize --run runs/modular_addition --output_dir plots
```

Quick smoke test (CPU-only, tiny run) – useful both locally and in CI before touching training code:

```bash
make smoke
```

Fast local quality gate:

```bash
make check
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

@software{mcbride_2025_grokking_mechanism_test,
  author = {Michael McBride},
  title = {grokking-mechanism-test: Geometric Grokking Hypothesis Experiments},
  year = {2025},
  url = {https://github.com/MichaelsEngineering/grokking-mechanism-test},
  version = {1.0}
}
