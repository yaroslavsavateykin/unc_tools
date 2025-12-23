# unc_tools

Utilities for uncertainty-aware regression and symbolic function helpers.

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
from unc_tools import UncRegression, FunctionBase1D
```

## Tests

The `tests/` directory contains runnable scripts:

```bash
uv run python tests/test_find_x.py
uv run python tests/test_expr_find_x.py
```

## Build and publish

```bash
uv pip install build twine
uv run python -m build
uv run twine upload dist/*
```

## License

This project is licensed under the MIT License.
