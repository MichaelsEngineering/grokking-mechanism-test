# Grokking Mechanism Test

**Exploring the Geometric Grokking Hypothesis** — that delayed generalization in neural networks arises from a *geometric phase transition* minimizing low-frequency energy of an implicit graph Laplacian in learned representations, rather than from weight decay or circuit efficiency alone.
*(Inspired by [“Geometric GROKKING Unlocked & Explained”](https://youtu.be/PaSm5vHYDew?si=vodCdnDGcaAxfQpJ), Discover AI, 2025.)*

---

## Overview

This repository provides the scaffolding for reproducible tests of the **geometric grokking hypothesis**, focusing on controlled toy-tasks drawn from deep learning and reinforcement learning literature. The experiments are designed to measure *spectral energy redistribution* and *representation smoothness* as networks transition from memorization to generalization.

---

## Getting Started

### Prerequisites
- **Python:** 3.11
- **Hardware:** NVIDIA GPU recommended
- **Framework:** PyTorch (default backend)

### Installation

1.  **Clone and enter the repository:**
    ```bash
    git clone https://github.com/MichaelsEngineering/grokking-mechanism-test.git
    cd grokking-mechanism-test
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python3.11 -m venv grokking-mech-env
    source grokking-mech-env/bin/activate
    ```

3.  **Install dependencies (PyTorch default):**
    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    pip install -r requirements-torch.txt
    # Or run: make init
    ```
    *Alternate backends are available via `requirements-jax.txt` and `requirements-tensorflow.txt`.*

---
## Usage

The YAML configuration files in `configs/` are the central control mechanism for experiments. They are used by `src.scripts.train` to define and parameterize every aspect of a specific experiment, from data generation to metric computation.

### Training
```bash
python -m src.scripts.train --config configs/modular_addition.yaml
```

### Evaluation
```bash
# Evaluate a full run directory
python -m src.scripts.evaluate --run-dir runs/modular_addition

# Or, evaluate from a metrics file
python -m src.scripts.evaluate --metrics runs/modular_addition/metrics.csv
```

### Visualization
```bash
python -m src.scripts.visualize --run runs/modular_addition --output_dir plots
```

### Quick Checks

Run a quick, CPU-only smoke test to verify the pipeline:
```bash
make smoke
```

Run the fast local quality gate (linting, type-checking, and unit tests):
```bash
make check
```

## Make Targets

```bash
make smoke     # Tiny CPU-only training sanity check
make check     # Pre-push quality gate (lint + type + tests)
make analytic  # Runs analytic dynamics test (Musat 2025 reproduction)
```

---

<details>
<summary><b>📂 Repository Structure</b></summary>

```
grokking-mechanism-test/
├── .github/              # CI/CD workflows
├── configs/              # Experiment configuration files (YAML)
├── runs/                 # Output directory for training runs (logs, checkpoints)
├── plots/                # Output directory for visualizations
├── src/                  # Source code
│   ├── scripts/          # Main scripts for training, evaluation, etc.
│   └── ...
├── tests/                # Test suite
│   ├── fixtures/         # Test data and fixtures
│   └── ...
├── .gitignore            # Git ignore rules
├── .pre-commit-config.yaml # Pre-commit hook configurations
├── AGENT.md              # Instructions for AI agents
├── CITATION.cff          # Citation file format
├── LICENSE               # Project license
├── Makefile              # Makefile with helper commands (e.g., `make smoke`)
├── pyproject.toml        # Project metadata and build configuration
├── README.md             # This file
└── requirements-*.txt    # Python dependency files for different backends
```

</details>

---

## Spectral Mechanism Tests

### Spectral Energy Shift Test

The default configuration enables the Spectral Energy Shift Test, which tracks how representation energy migrates to smoother Laplacian modes during training.

- **Graph**: We analyze logits on the toroidal 4-neighbor graph \(C_N \Box C_N\) defined over all \((a, b)\) input pairs. Its normalized Laplacian has an analytical 2-D DFT basis, so projections are computed exactly without forming dense matrices.
- **Sampling cadence**: Spectral metrics are evaluated in lock-step with the training evaluation loop (`train.eval_every_steps`) and always at the final step, ensuring even tiny runs surface correctness/logging signals.
- **Metrics**: Each evaluation logs the low-frequency energy ratio (`spectral_low_frac`) and spectral entropy (`spectral_entropy`) into `metrics.csv`. Snapshot files saved under `runs/<experiment>/spectral/` include the per-frequency spectrum (`spectrum_step*.pt`) and per-dimension low-energy fractions (`per_dim_step*.csv`) for deeper inspection or plotting.
- **Configuration knobs**:
  - `spectral.mode`: `fraction` (default), `count`, or `cutoff`—chooses how to carve out the low-frequency band.
  - `spectral.value`: parameter attached to the mode (e.g., 0.10 keeps the lowest 10 % of Laplacian modes in `fraction` mode).
  - `spectral.include_zero`: whether to force the zero-eigenvalue mode into the band.
  - `spectral.projection_samples`: limits how many per-dimension entries we persist per snapshot (useful for large output spaces).
  - `spectral.snapshot_stride`: write snapshots every n-th spectral evaluation.

These hooks run locally on CPU, making them suitable for smoke tests and CI. As the full training loop matures, the same analyzer will ingest real logits/hidden states instead of the current synthetic probes.

### 🧩 Analytic Dynamics Test (Norm Minimization)

Implements an analytic reproduction of grokking as described in **Musat (2025)**. Instead of stochastic training, this test integrates the *zero-loss manifold gradient flow*:

\[
\dot{W}_1 \approx X^T[(A Y Y^T A H) \odot \sigma'(X W_1)] - W_1
\]

This simulates how weight decay minimizes the parameter norm while staying on the zero-loss manifold, reproducing **delayed generalization** and **circular Fourier embeddings** seen in modular addition tasks.

**Test name:** `test_norm_min_dynamics.py`  
**Config flag:** `analytic_dynamics: true`  
**Assertions:**  
- training loss remains near zero  
- test accuracy rises late  
- Fourier feature norms equalize  
- real/imag Fourier parts become orthogonal

---

<details>
<summary><b>🔬 Planned Experiments</b></summary>

| **Experiment** | **Description** | **Metrics** | **Expected Outcome** | **Status** |
|----------------|-----------------|--------------|----------------------|-------------|
| **Spectral Energy Shift Test** | Track how learned representation energy moves from high- to low-frequency Laplacian modes during training. | Low-frequency energy ratio, spectral entropy, validation accuracy. | Gradual transfer of representational energy to smoother modes correlates with grokking onset. | ✅ Completed |
| **Laplacian Energy Penalty Ablation** | Add or remove an explicit Laplacian energy regularizer to test causal role of geometric smoothness. | Grokking time, accuracy gap, mean feature Laplacian energy. | Models with controlled low-energy bias should grok faster or more consistently. | 🚧 Planned |
| **Weight-Decay Baseline** | Compare identical models trained with classic L2 weight decay. | Validation accuracy vs. epoch, parameter norm trajectory. | Weight decay alone reproduces some but not all smoothness signatures. | 🚧 Planned |
| **Circuit-Efficiency Proxy** | Test hypothesis that grokking stems from efficient sub-circuit selection rather than geometric reorganization. | Parameter sparsity, FLOPs, accuracy. | Improvements appear without geometric reorganization, distinguishing competing theories. | 🚧 Planned |
| **Synthetic Modular Arithmetic** | Minimal synthetic task (e.g., mod-N addition) for measuring grokking transition. | Accuracy, loss, spectral energy distribution. | Clear delayed generalization and spectral phase transition. | 🚧 Planned |
| **Parity & Sequence Copy Tasks** | RL-style toy domains from small-scale deep-RL benchmarks. | Reward, accuracy, smoothness metrics. | Reinforces that geometric smoothness generalizes beyond simple arithmetic tasks. | 🚧 Planned |

</details>

---

## Contributing

Please see `CONTRIBUTING.md` for details on how to contribute to this project. For feature requests, please use the "✨ Feature Request" issue template on GitHub.

## Research Context

This repository seeks to provide empirical footing for the geometric grokking hypothesis, connecting observed generalization delays to measurable changes in representation geometry. It aims to complement other explanations (regularization, sparsity, or circuit efficiency) by introducing tools to visualize phase transitions in representation manifolds.

## Citation

If you use or reference this repository, please cite:
```bibtex
@software{mcbride_2025_grokking_mechanism_test,
  author = {Michael McBride},
  title = {grokking-mechanism-test: Geometric Grokking Hypothesis Experiments},
  year = {2025},
  url = {https://github.com/MichaelsEngineering/grokking-mechanism-test},
  version = {1.0}
}
```
