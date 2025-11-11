"""Training and visualization helpers for grokking_mechanism_test."""

from importlib import import_module
from typing import Any

__all__ = ["train", "visualize", "evaluate"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)
