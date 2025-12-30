"""Regression utilities with uncertainty propagation."""

from __future__ import annotations

import re
import uuid
import warnings
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import sympy as sym
import uncertainties as unc
import uncertainties.unumpy as unp
from matplotlib.axes import Axes

from .default_functions import FunctionBase1D, Poly
from .exceptions import DataError, InitialGuessError, ModelTypeError

__all__ = ["UncRegression"]

Uncertain = unc.core.AffineScalarFunc | unc.core.Variable
Numeric = float | int | np.number
DataInput = Sequence[Numeric | Uncertain] | np.ndarray | pd.Series
IndexSlice = slice | Sequence[int] | np.ndarray
Solution = sym.Expr | Uncertain | tuple[Uncertain, Uncertain]
PredictInput = Numeric | Uncertain | Sequence[Numeric | Uncertain] | np.ndarray


class UncRegression:
    """Perform regression analysis with uncertainty-aware data and parameters."""

    @staticmethod
    def latex_style(tex: bool) -> None:
        """Configure matplotlib to use LaTeX text rendering.

        Args:
            tex: Whether to enable LaTeX text rendering.
        """
        if tex:
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern"],
                    "text.latex.preamble": r"""
                    \usepackage[utf8]{inputenc}
                    \usepackage[russian]{babel}
                    \usepackage[T2A]{fontenc}
                """,
                    "pgf.texsystem": "xelatex",
                }
            )
        else:
            plt.rcParams.update(
                {
                    "text.usetex": False,
                    "font.family": "sans-serif",
                    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
                    "pgf.texsystem": None,  # Can be removed to restore defaults.
                }
            )
        return None

    def __init__(
        self,
        x: DataInput,
        y: DataInput,
        func: Callable[..., Numeric | np.ndarray | Uncertain]
        | FunctionBase1D
        | str
        | None = None,
        sl: IndexSlice | None = None,
    ) -> None:
        """Initialize regression with uncertainty-aware inputs.

        Args:
            x: Input data for the independent variable.
            y: Input data for the dependent variable.
            func: Callable model, symbolic `FunctionBase1D`, expression string, or
                None for a linear polynomial.
            sl: Optional slice or index sequence applied to the inputs.

        Raises:
            ModelTypeError: If the model type is unsupported.
            DataError: If the input arrays are empty.
        """
        if issubclass(type(func), FunctionBase1D):
            self.func = func.lambda_fun
            self.expression = func
        elif isinstance(func, Callable):
            self.func = func
            self.expression = None
        elif isinstance(func, str):
            self.expression = FunctionBase1D(func)
            self.func = self.expression.lambda_fun
        elif func is None:
            self.expression = Poly(1)
            self.func = self.expression.lambda_fun
        else:
            raise ModelTypeError(
                "func must be a callable, FunctionBase1D instance, expression string, or None."
            )

        if len(x) == 0 or len(y) == 0:
            raise DataError("Input arrays cannot be empty.")

        if sl is not None:
            self.x = np.asarray(x)[sl]
            self.y = np.asarray(y)[sl]
        else:
            self.x = np.asarray(x)
            self.y = np.asarray(y)

        self.x_has_unc = len(self.x) > 0 and hasattr(self.x[0], "nominal_value")
        self.y_has_unc = len(self.y) > 0 and hasattr(self.y[0], "nominal_value")

        self._extract_values()

        self._fit()
        return None

    def _extract_values(self) -> None:
        """Extract nominal values and uncertainties from the input data."""
        try:
            self.x_nom = (
                unp.nominal_values(self.x)
                if self.x_has_unc
                else np.asarray(self.x, dtype=float)
            )
            self.y_nom = (
                unp.nominal_values(self.y)
                if self.y_has_unc
                else np.asarray(self.y, dtype=float)
            )
            self.x_std = unp.std_devs(self.x) if self.x_has_unc else None
            self.y_std = unp.std_devs(self.y) if self.y_has_unc else None
        except TypeError:
            self.x_nom = np.asarray(self.x, dtype=float)
            self.y_nom = np.asarray(self.y, dtype=float)
            self.x_std = None
            self.y_std = None

        # Replace zero uncertainties with small positive values.
        self._handle_zero_uncertainties()
        return None

    def _handle_zero_uncertainties(self) -> None:
        """Replace zero uncertainties with small positive values."""
        for arr, name in zip([self.x_std, self.y_std], ["x", "y"]):
            if arr is not None and np.any(arr == 0):
                non_zero = arr[arr > 0]
                replacement = np.min(non_zero) * 1e-5 if len(non_zero) > 0 else 1e-15
                arr[arr == 0] = replacement
                warnings.warn(
                    f"Zero uncertainties in {name} replaced with {replacement:.1e}"
                )
        return None

    def _fit(self) -> None:
        """Fit the model and compute uncertainty-aware coefficients."""
        try:
            if self.y_std is not None:
                self.popt, self.pcov = opt.curve_fit(
                    self.func,
                    self.x_nom,
                    self.y_nom,
                    sigma=self.y_std,
                    absolute_sigma=True,
                )
            else:
                self.popt, self.pcov = opt.curve_fit(self.func, self.x_nom, self.y_nom)
        except (RuntimeError, TypeError) as e:
            warnings.warn(f"Curve fitting failed: {e}")
            n_params = self.func.__code__.co_argcount - 1
            self.popt = np.ones(n_params)
            self.pcov = np.eye(n_params)

        residuals = self.y_nom - self.func(self.x_nom, *self.popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.y_nom - np.mean(self.y_nom)) ** 2)
        self.R2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        try:
            self.coefs = unp.uarray(self.popt, np.sqrt(np.diag(self.pcov)))
            if self.expression:
                self.expression.add_coefs(self.coefs)
        except (TypeError, ValueError) as exc:
            warnings.warn(f"Coefficient uncertainties could not be applied: {exc}")
            if self.expression:
                self.expression.add_coefs(self.popt)
            self.coefs = unp.uarray(self.popt, np.zeros_like(self.popt))
        return None

    @property
    def coefs_nom(self) -> np.ndarray:
        """Return nominal coefficient values without uncertainties."""
        return unp.nominal_values(self.coefs)

    @property
    def coefs_std(self) -> np.ndarray:
        """Return coefficient standard deviations."""
        return unp.std_devs(self.coefs)

    def plot(
        self,
        figsize: tuple[float, float] = (10, 5),
        labels: Sequence[str] | None = None,
        ax: Axes | None = None,
        x_ax: Sequence[float] | np.ndarray | None = None,
        path: str = "",
        label: str = "",
        show_errors: bool = True,
        show_band: bool = False,
        show_scatter: bool = True,
        band_alpha: float = 0.2,
        band_color: str | None = None,
        add_legend: bool = True,
        show_expr: bool = True,
        show_coefficients: bool = False,
        show_r2: bool = True,
        **kwargs: object,
    ) -> Axes:
        """Plot the regression results.

        Args:
            figsize: Figure size in inches.
            labels: Axis labels as ``[xlabel, ylabel]``.
            ax: Existing axes to draw on; creates one when None.
            x_ax: Optional x-axis grid for the fitted curve.
            path: Output path for saving the figure; empty disables saving.
            label: Prefix for legend entries.
            show_errors: Whether to show error bars when uncertainties are present.
            show_band: Whether to show the confidence band.
            show_scatter: Whether to draw the observation points.
            band_alpha: Alpha for the confidence band fill.
            band_color: Color for the confidence band.
            add_legend: Whether to add a legend.
            show_expr: Whether to include the analytic expression in the legend.
            show_coefficients: Whether to include coefficients in the legend.
            show_r2: Whether to include the R-squared value in the legend.
            **kwargs: Additional keyword arguments forwarded to matplotlib.

        Returns:
            The axes containing the plot.
        """
        if labels is None:
            labels = ["", ""]

        if x_ax is None:
            x_min, x_max = np.min(self.x_nom), np.max(self.x_nom)
            delta = (x_max - x_min) * 0.0001
            x_ax = np.linspace(x_min - delta, x_max + delta, 500)
        y_ax = self.func(x_ax, *self.popt)

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=150)
        else:
            fig = ax.figure

        plot_kwargs = kwargs.copy()
        if "color" not in plot_kwargs:
            try:
                prop_cycle = ax._get_lines.prop_cycler
                color = next(prop_cycle)["color"]
                plot_kwargs["color"] = color
            except (AttributeError, StopIteration):
                color_index = len(ax.lines) % 10
                plot_kwargs["color"] = f"C{color_index}"

        y_range = np.nanmax(self.y_nom) - np.nanmin(self.y_nom)
        min_visual_std = 0.02 * y_range if y_range > 0 else 1e-10

        if show_scatter:
            if show_errors and (self.x_std is not None or self.y_std is not None):
                y_err = None
                x_err = None
                if self.y_std is not None:
                    y_err = np.maximum(self.y_std, min_visual_std)
                if self.x_std is not None:
                    x_err = np.maximum(self.x_std, min_visual_std)

                errorbar_kwargs = {
                    "capsize": 3,
                    "capthick": 1.5,
                    "elinewidth": 1.5,
                    "ms": 4,
                    "alpha": 0.8,
                }
                errorbar_kwargs.update(plot_kwargs)

                eb = ax.errorbar(
                    self.x_nom,
                    self.y_nom,
                    xerr=x_err,
                    yerr=y_err,
                    fmt=".",
                    **errorbar_kwargs,
                )

                if band_color is None and show_band:
                    band_color = eb[0].get_color()
            else:
                ax.scatter(self.x_nom, self.y_nom, **plot_kwargs)
                if band_color is None and show_band:
                    band_color = plot_kwargs.get("color", "blue")

        latex_expr = ""
        latex_coefs = ""
        r2_str = ""

        if label:
            label = label + "\n"

        if show_r2:
            r2_str = f"R²= {self.R2:.5f}"

        if show_expr:
            if self.expression:
                latex_expr = f"$y = {self.expression.to_latex_expr()}$\n"
            else:
                warnings.warn(
                    "Expression display requested but no expression is available."
                )
                show_expr = False

        if show_coefficients and not self.expression:
            warnings.warn(
                "Coefficient display requested but no expression is available."
            )
            show_coefficients = False

        if show_coefficients:
            coefs = self.expression.args

            latex_coefs = "\n".join(
                f"${sym.latex(coefs[i])} = {self.coefs[i]:.2u}$"
                for i in range(len(self.coefs))
            )
            latex_coefs = (
                re.sub(
                    r"e([+-])(\d+)",
                    lambda m: rf"\cdot 10^{{{m.group(1)}{str(int(m.group(2)))}}}",
                    latex_coefs,
                )
                + "\n"
            )
            latex_coefs = latex_coefs.replace("+/-", "\\pm")

        label = f"{label}{latex_expr}{latex_coefs}{r2_str}"

        ax.plot(x_ax, y_ax, label=label, color=plot_kwargs.get("color"))

        if show_band:
            try:
                params = unc.correlated_values(self.popt, self.pcov)

                y_vals = [self.func(x_val, *params) for x_val in x_ax]

                y_nom_band = unp.nominal_values(y_vals)
                y_std_band = unp.std_devs(y_vals)

                ax.fill_between(
                    x_ax,
                    y_nom_band - 1.96 * y_std_band,
                    y_nom_band + 1.96 * y_std_band,
                    alpha=band_alpha,
                    color=band_color,
                    label=r"95$\%$ Confidence Interval",
                )
            except Exception as e:
                warnings.warn(f"Confidence band could not be plotted: {e}")

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.grid(True)

        if add_legend and (label or show_band):
            ax.legend()

        if path:
            if fig is None:
                fig = plt.gcf()
            fig.savefig(path, bbox_inches="tight")

        return ax

    def predict_n(self, x_new: PredictInput) -> np.ndarray | Numeric | Uncertain:
        """Predict values for new inputs using nominal parameters.

        Args:
            x_new: Input values for prediction.

        Returns:
            Predicted values from the fitted model.
        """
        return self.func(x_new, *self.popt)

    def predict(self, x_new: PredictInput) -> np.ndarray | Uncertain:
        """Predict values with uncertainty propagation from fit parameters.

        Args:
            x_new: Input values for prediction.

        Returns:
            Predicted values with parameter uncertainty propagated.
        """
        params = unc.correlated_values(self.popt, self.pcov)

        if hasattr(x_new, "__iter__"):
            y_pred = [self.func(x_val, *params) for x_val in x_new]
            return np.array(y_pred)
        else:
            y_pred = self.func(x_new, *params)
            return y_pred

    def find_x(
        self,
        y: Uncertain | Numeric,
        x0: Uncertain | Numeric | None = None,
        xtol_root: float = 1e-20,
        xtol_diff: float = 1e-20,
        maxiter: int = 500,
        # solve_numerically: bool = False,
        **kwargs: object,
    ) -> np.ndarray | Solution | list[Solution]:
        """Solve for x given a target y value.

        Uses analytical solutions when a symbolic expression is available; otherwise
        uses numerical root finding with uncertainty propagation.

        Args:
            y: Target y value, optionally with uncertainty.
            x0: Initial guess for numerical root finding.
            xtol_root: Tolerance for the root-finding step.
            xtol_diff: Tolerance used for numerical differentiation.
            maxiter: Maximum number of iterations.
            **kwargs: Additional arguments forwarded to `scipy.optimize.root_scalar`.

        Returns:
            Analytical solutions, a list of solutions, or an uncertainty-aware root.

        Raises:
            InitialGuessError: If numerical solving is required and no initial guess is provided.
        """
        if hasattr(y, "__iter__"):
            return np.array(
                [
                    self.find_x(
                        y0,
                        x0=x0,
                        xtol_diff=xtol_diff,
                        xtol_root=xtol_root,
                        maxiter=maxiter,
                        # solve_numerically=solve_numerically,
                        **kwargs,
                    )
                    for y0 in y
                ]
            )

        args_nominal = self.coefs_nom

        if isinstance(y, (unc.core.Variable, unc.core.AffineScalarFunc)):
            yval = y.nominal_value
            ytol = y.std_dev
        else:
            yval = y
            ytol = 0

        if isinstance(x0, (unc.core.Variable, unc.core.AffineScalarFunc)):
            xtol = x0.std_dev
            x0 = x0.nominal_value
        else:
            xtol = 0
            x0 = x0

        if self.expression and x0 is None:
            sols = self.expression.find_sols(y=y)

            if hasattr(sols, "__iter__"):
                if len(sols) == 1:
                    return sols[0]
                else:
                    return sols
            else:
                return sols

        else:
            if x0 is None:
                raise InitialGuessError("An initial guess x0 is required for numerical solving.")

            trys = 5
            i = 0
            while i < trys:
                result_root = opt.root_scalar(
                    f=lambda x, *args: self.func(x, *args) - yval,
                    x0=x0,
                    xtol=xtol_root,
                    maxiter=maxiter,
                    args=tuple(args_nominal),
                    **kwargs,
                )

                if not result_root.converged:
                    xtol_root *= 100
                    # print(f"Root not converged: {result_root.flag}, trying again")
                i += 1
            # if not result_root.converged:
            #     raise TypeError(f"Root not converged: {result_root.flag}")

            if self.expression is None:
                fun = self.func
            else:
                fun = FunctionBase1D(self.expression.expr_str).lambda_fun
            coefs_nom = unc.unumpy.nominal_values(self.coefs)
            coefs_std = unc.unumpy.std_devs(self.coefs)

            def f_vec(v):
                x = v[0]
                params = v[1:]
                return fun(x, *params)

            v0 = np.concatenate(([x0], coefs_nom))

            grad = opt.approx_fprime(v0, f_vec, epsilon=1e-8)

            dy = grad[0]
            dcoefs = grad[1:]

            dxtol = np.sqrt(ytol**2 + np.sum((dcoefs * coefs_std) ** 2)) / dy

            # print(xtol_root, xtol_diff, dxtol, xtol)
            xtol_summ = np.sqrt(xtol_root**2 + xtol_diff**2 + dxtol**2 + xtol**2)

            return unc.ufloat(result_root.root, xtol_summ)

    def to_df(self, export_plot: bool = False) -> pd.DataFrame:
        """Convert regression results to a DataFrame.

        Args:
            export_plot: If True, return the fitted curve grid instead of input data.

        Returns:
            DataFrame containing the input data or fitted curve samples.
        """
        if export_plot:
            x_min, x_max = np.min(self.x_nom), np.max(self.x_nom)
            delta = (x_max - x_min) * 0.1
            x_ax = np.linspace(x_min - delta, x_max + delta, 500)
            y_ax = self.func(x_ax, *self.popt)
            df = {"x_fit": x_ax, "y_fit": y_ax}

        else:
            y = unp.nominal_values(self.y)

            df = {
                "x": unp.nominal_values(self.x),
                "y": y,
                "x_std": self.x_std
                if self.x_std is not None and len(self.x_std) > 0
                else np.zeros(np.size(self.x)),
                "y_std": self.y_std
                if self.y_std is not None and len(self.y_std) > 0
                else np.zeros(np.size(self.y)),
            }

        df = pd.DataFrame(df)

        return df

    def to_csv(
        self,
        filename: str | None = None,
        export_plot: bool = False,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Export regression results to a CSV file.

        Args:
            filename: Output filename; generates a random name when None.
            export_plot: Export fitted curve data instead of the input data.
            *args: Positional arguments forwarded to `DataFrame.to_csv`.
            **kwargs: Keyword arguments forwarded to `DataFrame.to_csv`.

        Raises:
            OSError: If the file cannot be written.
        """
        df = self.to_df(export_plot=export_plot)

        if not filename:
            filename = f"{str(uuid.uuid4())[:7]}.csv"
            print(f"DataFrame was saved to {filename}")

        df.to_csv(filename, *args, **kwargs)
        return None
