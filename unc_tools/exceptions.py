"""Custom exceptions for unc_tools."""

from __future__ import annotations

__all__ = [
    "UncToolsError",
    "DataError",
    "ExpressionError",
    "InitialGuessError",
    "ModelTypeError",
]


class UncToolsError(Exception):
    """Base exception for unc_tools."""


class DataError(ValueError, UncToolsError):
    """Raised when input data is missing or malformed."""


class ModelTypeError(TypeError, UncToolsError):
    """Raised when a model or callable has an unsupported type."""


class ExpressionError(TypeError, UncToolsError):
    """Raised when an expression or coefficient set is invalid."""


class InitialGuessError(TypeError, UncToolsError):
    """Raised when a numerical solver lacks a valid initial guess."""
