"""Public package interface for uncertainty-aware tools."""

from __future__ import annotations
from importlib.metadata import PackageNotFoundError, version

from .default_functions import FunctionBase1D, Hyper, Poly
from .exceptions import DataError, ExpressionError, InitialGuessError, ModelTypeError
from .unc_regression import UncRegression


try:
    __version__ = version("uncertainty_tools")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "DataError",
    "ExpressionError",
    "FunctionBase1D",
    "Hyper",
    "InitialGuessError",
    "ModelTypeError",
    "Poly",
    "UncRegression",
    "__version__",
]
