import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_keras_backend_switch(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("KERAS_BACKEND", "torch")

    # Ensure we do not reuse a previously imported keras module
    for module in list(sys.modules):
        if module.startswith("keras"):
            sys.modules.pop(module)

    import keras
    from keras import layers

    # Just ensure backend is set and a model compiles and trains for 1 step
    x = np.random.randn(64, 10).astype("float32")
    y = np.random.randn(64, 10).astype("float32")

    inp = keras.Input(shape=(10,))
    out = layers.Dense(10)(layers.Dense(32, activation="gelu")(inp))
    m = keras.Model(inp, out)
    m.compile(optimizer="adam", loss="mse")
    history = m.fit(x, y, epochs=1, batch_size=32, verbose=0)
    assert "loss" in history.history


def test_train_smoke(tmp_path: Path):
    pytest.importorskip("torch")
    repo_root = Path(__file__).resolve().parents[1]
    config = repo_root / "configs" / "modular_addition.yaml"
    out_dir = tmp_path / "runs" / "modular_addition"

    # run 50 steps on CPU; should be ~1–2s
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "train.py"),
        "--config",
        str(config),
        "train.total_steps=50",
        f"logging.out_dir={out_dir}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    # artifacts
    metrics = out_dir / "metrics.csv"
    cfg_copy = out_dir / "config_used.yaml"
    assert metrics.exists()
    assert cfg_copy.exists()

    # header & last step
    with metrics.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) >= 1
    assert {"step", "train_loss", "train_acc", "val_acc", "test_acc"}.issubset(reader.fieldnames)
    assert int(rows[-1]["step"]) == 50
