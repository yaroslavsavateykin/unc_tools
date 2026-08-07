import numpy as np
import pytest
import uncertainties as unc

from unc_tools import DataError, ExpressionError, FunctionBase1D, UncRegression


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
