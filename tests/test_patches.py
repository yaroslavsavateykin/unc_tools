import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
import pytest
import sympy as sym
import uncertainties as unc

from unc_tools.patches import new_lambdify


def test_matplotlib_plot_keeps_standard_single_argument_form():
    figure, axes = plt.subplots()

    lines = axes.plot([1.0, 2.0, 3.0])

    assert len(lines) == 1
    plt.close(figure)


def test_matplotlib_plot_forwards_uncertainty_plot_options():
    figure, axes = plt.subplots()

    result = axes.plot(
        [0.0, 1.0],
        [unc.ufloat(1.0, 0.1), unc.ufloat(2.0, 0.1)],
        "s--",
        color="red",
    )

    assert isinstance(result, ErrorbarContainer)
    assert result.lines[0].get_color() == "red"
    assert result.lines[0].get_marker() == "s"
    plt.close(figure)


def test_uncertainty_lambdify_supports_scalar_and_multiple_arguments():
    x, y = sym.symbols("x y")
    function = new_lambdify((x, y), x * y, "unc")

    result = function(unc.ufloat(2.0, 0.1), unc.ufloat(3.0, 0.2))

    assert result.nominal_value == 6.0
    assert result.std_dev == pytest.approx(0.5)
