import numpy as np
import pytest
import uncertainties as unc

from unc_tools import FitError, RootFindingError, UncRegression


def quadratic(x, a):
    return a * x**2


def test_numerical_find_x_evaluates_uncertainty_at_root():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    regression = UncRegression(x, quadratic(x, 2.0), quadratic)

    root = regression.find_x(unc.ufloat(18.0, 0.9), x0=1.0)

    assert root.nominal_value == pytest.approx(3.0)
    assert root.std_dev == pytest.approx(0.075, rel=0.05)


def test_failed_fit_raises_instead_of_returning_placeholder_coefficients():
    with pytest.raises(FitError):
        UncRegression([1.0], [2.0], lambda x, a, b: a * x + b)


def test_zero_derivative_at_root_is_reported():
    regression = UncRegression(
        np.array([-2.0, -1.0, 1.0, 2.0]),
        np.array([4.0, 1.0, 1.0, 4.0]),
        quadratic,
    )

    with pytest.raises(RootFindingError):
        regression.find_x(0.0, x0=0.1)
