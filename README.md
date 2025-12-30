
# uncertainty-tools

Utilities for uncertainty-aware regression and symbolic function helpers.

**Python version:** 3.10+

> 📦 PyPI package name: `uncertainty-tools`  
> 🧩 Python import name: `unc_tools`

---

## Installation

### Using pip

```bash
pip install uncertainty-tools
````

### Using uv

```bash
uv pip install uncertainty-tools
```

---

## Usage

```python
from unc_tools import FunctionBase1D, Poly, UncRegression

# Symbolic expression with coefficients
expr = FunctionBase1D("a*x + b")

# Polynomial helper
poly = Poly(2)

# Regression with uncertainty propagation
reg = UncRegression(
    x=[0, 1, 2],
    y=[0, 1.1, 1.9],
    func=expr,
)

pred = reg.predict([0.5, 1.5])
```

---

## Optional matplotlib and sympy patches

Importing `unc_tools.patches` monkey-patches:

- `matplotlib.axes.Axes.scatter`
    
- `matplotlib.axes.Axes.plot`
    

and adds uncertainty-aware helpers for SymPy substitution and `lambdify`.

⚠️ This introduces **global side effects** and must be enabled explicitly:

```python
import unc_tools.patches  # noqa: F401
```

---

## Public API

- `UncRegression` — regression with uncertainty propagation
- `FunctionBase1D`, `Poly`, `Hyper` — symbolic helpers for 1D expressions
- `DataError`, `ExpressionError`, `InitialGuessError`, `ModelTypeError` — custom exceptions
    

---

## Project status

⚠️ This project is under active development.  
The public API may change between minor versions.

---

## License

This project is licensed under the MIT License.


