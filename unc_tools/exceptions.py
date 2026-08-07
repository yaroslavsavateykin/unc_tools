"""Custom exceptions for unc_tools."""

from __future__ import annotations

__all__ = [
    "UncToolsError",
    "DataError",
    "ExpressionError",
    "FitError",
    "InitialGuessError",
    "ModelTypeError",
    "RootFindingError",
]


class UncToolsError(Exception):
    """Base exception for unc_tools."""


class DataError(ValueError, UncToolsError):
    """Raised when input data is missing or malformed."""


class ModelTypeError(TypeError, UncToolsError):
    """Raised when a model or callable has an unsupported type."""


class ExpressionError(TypeError, UncToolsError):
    """Raised when an expression or coefficient set is invalid."""


class FitError(RuntimeError, UncToolsError):
    """Raised when a regression model cannot be fitted to the supplied data."""


class InitialGuessError(TypeError, UncToolsError):
    """Raised when a numerical solver lacks a valid initial guess."""


class RootFindingError(RuntimeError, UncToolsError):
    """Raised when a numerical inverse solution does not converge."""
