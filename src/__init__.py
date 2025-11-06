"""Core package for grokking_mechanism_test."""

from importlib import metadata as _metadata

__all__ = ["__version__"]


def __getattr__(name: str):
    if name == "__version__":
        try:
            return _metadata.version("grokking-mechanism-test")
        except _metadata.PackageNotFoundError:
            return "0.0.0"
    raise AttributeError(name)
