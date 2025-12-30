# unc_tools

Utilities for uncertainty-aware regression and symbolic function helpers.

Python version: 3.10+

## Install (uv)

```bash
uv venv
uv pip install -e .
```

## Install (pip)

```bash
pip install unc_tools
```

## Usage

```python
from unc_tools import FunctionBase1D, Poly, UncRegression

# Symbolic expression with coefficients.
expr = FunctionBase1D("a*x + b")

# Polynomial helper.
poly = Poly(2)

# Regression with uncertainty propagation.
reg = UncRegression([0, 1, 2], [0, 1.1, 1.9], func=expr)
pred = reg.predict([0.5, 1.5])
```

## Optional matplotlib and sympy patches

Importing `unc_tools.patches` monkey-patches matplotlib `Axes.scatter`/`Axes.plot`
and adds uncertainty-aware helpers for sympy substitution/lambdify. This is a
global side effect and should be enabled explicitly:

```python
import unc_tools.patches  # noqa: F401
```

## Public API

- `UncRegression`: regression with uncertainty propagation.
- `FunctionBase1D`, `Poly`, `Hyper`: symbolic helpers for 1D expressions.
- `DataError`, `ExpressionError`, `InitialGuessError`, `ModelTypeError`: custom exceptions.


## License

This project is licensed under the MIT License.
