import numpy as np
import pytest
import uncertainties as unc
from matplotlib import rcParams
import matplotlib.pyplot as plt

from unc_tools import DataError, ExpressionError, FunctionBase1D, UncRegression, serif


def test_serif_is_opt_in_and_restores_matplotlib_defaults():
    assert not hasattr(UncRegression, "latex_style")
    original = {
        "font.family": list(rcParams["font.family"]),
        "font.serif": list(rcParams["font.serif"]),
    }

    serif()
    assert rcParams["text.usetex"] is False
    assert rcParams["font.family"] == ["serif"]
    assert rcParams["font.serif"][:3] == [
        "CMU Serif",
        "Computer Modern Roman",
        "DejaVu Serif",
    ]

    regression = UncRegression([0.0, 1.0, 2.0], [1.0, 3.01, 4.99])
    axes = regression.plot(serif=False, add_legend=False, show_expr=False)
    plt.close(axes.figure)

    assert rcParams["text.usetex"] is False
    assert rcParams["font.family"] == original["font.family"]
    assert rcParams["font.serif"] == original["font.serif"]


def test_regression_plot_accepts_serif_option():
    regression = UncRegression([0.0, 1.0, 2.0], [1.0, 3.01, 4.99])

    axes = regression.plot(serif=True, add_legend=False, show_expr=False)

    assert rcParams["font.family"] == ["serif"]
    plt.close(axes.figure)


def test_expression_orders_overlapping_coefficient_names_correctly():
    expression = FunctionBase1D("p_10*x + p_1")

    assert [str(symbol) for symbol in expression.args] == ["p_10", "p_1"]
    assert expression.lambda_fun(2.0, 3.0, 4.0) == 10.0


def test_expression_rejects_python_syntax():
    with pytest.raises(ExpressionError):
        FunctionBase1D("__import__('os').system('false')")


def test_x_uncertainties_use_odr_and_input_lengths_are_validated():
    x = np.array([unc.ufloat(value, 0.05) for value in [0.0, 1.0, 2.0, 3.0]])
    y = np.array(
        [unc.ufloat(value, 0.1) for value in [1.02, 2.98, 5.01, 7.03]]
    )
    regression = UncRegression(x, y)

    assert regression.coefs_nom == pytest.approx([2.0, 1.0], abs=0.05)
    with pytest.raises(DataError):
        UncRegression([0.0], [0.0, 1.0])
