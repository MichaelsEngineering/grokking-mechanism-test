import sys

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
