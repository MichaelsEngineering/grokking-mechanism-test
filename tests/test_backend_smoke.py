import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    keras: Any
    layers: Any


def test_keras_backend_switch(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("KERAS_BACKEND", "torch")

    # Ensure we do not reuse a previously imported keras module
    for module in list(sys.modules):
        if module.startswith("keras"):
            sys.modules.pop(module)

    keras = cast(Any, import_module("keras"))
    layers = cast(Any, getattr(keras, "layers"))

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
    scripts_dir = repo_root / "src" / "scripts"
    out_dir = tmp_path / "runs" / "modular_addition"

    # run 50 steps on CPU; should be ~1–2s
    cmd = [
        sys.executable,
        str(scripts_dir / "train.py"),
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

    # header & per-split rows
    with metrics.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) >= 3
    assert reader.fieldnames is not None
    expected_fields = {"step", "split", "loss", "accuracy", "spectral_low_frac", "spectral_entropy"}
    assert expected_fields.issubset(reader.fieldnames)

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    assert len(train_rows) == 50
    assert len(val_rows) == 50
    assert len(test_rows) == 50

    last_val = val_rows[-1]
    assert int(last_val["step"]) == 50
    assert last_val["spectral_low_frac"] not in ("", None)
    assert last_val["spectral_entropy"] not in ("", None)
