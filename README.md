
# uncertainty-tools

Utilities for uncertainty-aware regression and symbolic function helpers.

**Python version:** 3.12+

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
uv add uncertainty-tools
```

--- 

## Notebooks

[link](https://github.com/yaroslavsavateykin/unc_tools/tree/main/notebooks)

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

## Optional matplotlib patches

Importing `unc_tools.patches` monkey-patches only:

- `matplotlib.axes.Axes.scatter`
- `matplotlib.axes.Axes.plot`
    

It also exposes `new_subs` and `new_lambdify` helpers for explicit SymPy use.
Plotly and SymPy global APIs are not modified.

⚠️ This introduces **global side effects** and must be enabled explicitly:

```python
import unc_tools.patches  # noqa: F401
```

---

## Project status

⚠️ This project is under active development.  
The public API may change between minor versions.

---

## License

This project is licensed under the MIT License.
