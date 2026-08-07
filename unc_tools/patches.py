"""Matplotlib and sympy patches for uncertainty-aware workflows."""

from __future__ import annotations

from typing import Callable, Sequence

import matplotlib
import numpy as np
import uncertainties as unc
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.container import ErrorbarContainer
from matplotlib.collections import PolyCollection
import sympy as sym
from uncertainties.unumpy import nominal_values, std_devs

from .default_functions import FunctionBase1D

__all__ = [
    "get_unc_attr",
    "new_lambdify",
    "new_plot",
    "new_scatter",
    "new_subs",
    "set_unc_attr",
    # "plot_with_uncertainty",
]

Uncertain = unc.core.AffineScalarFunc | unc.core.Variable
Numeric = float | int | np.number
ArrayLike = Sequence[Numeric | Uncertain] | np.ndarray

_original_scatter = matplotlib.axes.Axes.scatter


def new_scatter(
    self: Axes,
    x: ArrayLike,
    y: ArrayLike,
    *args: object,
    **kwargs: object,
) -> Artist | ErrorbarContainer:
    """Create a scatter plot that visualizes uncertainties when present.

    Converts inputs to numpy arrays, extracts nominal values and standard deviations
    when uncertainty variables are provided, and falls back to the original
    matplotlib scatter if no meaningful uncertainty is available. Error bars are
    added automatically when uncertainty magnitudes exceed a minimal visual threshold.

    Args:
        self: Target axes for rendering.
        x: X-coordinates, optionally containing uncertainty values.
        y: Y-coordinates, optionally containing uncertainty values.
        *args: Additional positional arguments forwarded to matplotlib scatter.
        **kwargs: Additional keyword arguments forwarded to matplotlib scatter.

    Returns:
        Matplotlib artist or container returned by the scatter/errorbar call.

    Raises:
        TypeError: If inputs cannot be converted to numeric arrays.
        ValueError: If input shapes are incompatible for plotting.

    Side Effects:
        None.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> ax = plt.subplot()
        >>> _ = ax.scatter([1, 2], [3, 4])
    """
    x = [x] if not hasattr(x, "__iter__") else x
    y = [y] if not hasattr(y, "__iter__") else y
    x = np.asarray(x)
    y = np.asarray(y)

    try:
        x_nom = nominal_values(x)
        y_nom = nominal_values(y)
        x_std = std_devs(x)
        y_std = std_devs(y)
    except (TypeError, ValueError):
        x_nom = np.asarray(x, dtype=float)
        y_nom = np.asarray(y, dtype=float)
        x_std = None
        y_std = None

    min_visual_std = 2e-10

    scatter_kwargs = kwargs.copy()
    if "color" not in scatter_kwargs and "c" not in scatter_kwargs:
        try:
            prop_cycle = matplotlib.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key().get(
                "color", ["C0", "C1", "C2", "C3", "C4", "C5"]
            )
            color_index = len(self.collections) % len(colors)
            scatter_kwargs["color"] = colors[color_index]
        except (AttributeError, KeyError):
            color_index = len(self.collections) % 11
            scatter_kwargs["color"] = f"C{color_index}"

    if x_std is None and y_std is None:
        return _original_scatter(self, x_nom, y_nom, *args, **scatter_kwargs)

    x_err = None
    y_err = None

    if x_std is not None:
        x_err = np.where(x_std > min_visual_std, x_std, 1)
        if np.all(x_err == 1):
            x_err = None

    if y_std is not None:
        y_err = np.where(y_std > min_visual_std, y_std, 1)
        if np.all(y_err == 1):
            y_err = None

    if x_err is None and y_err is None:
        return _original_scatter(self, x_nom, y_nom, *args, **scatter_kwargs)

    errorbar_kwargs = {
        "capsize": 4,
        "capthick": 2.5,
        "elinewidth": 2.5,
        "markersize": scatter_kwargs.get("s", 20) ** 0.5
        if "s" in scatter_kwargs
        else 6,
        "alpha": scatter_kwargs.get("alpha", 1),
    }

    if "color" in scatter_kwargs:
        errorbar_kwargs["color"] = scatter_kwargs["color"]

    for arg in ["s", "marker", "linewidths", "edgecolors"]:
        if arg in scatter_kwargs:
            del scatter_kwargs[arg]

    errorbar_kwargs.update(scatter_kwargs)

    return self.errorbar(
        x_nom,
        y_nom,
        xerr=x_err,
        yerr=y_err,
        fmt="o",
        **errorbar_kwargs,
    )


matplotlib.axes.Axes.scatter = new_scatter


_original_plot = matplotlib.axes.Axes.plot


def new_plot(
    self: Axes,
    *args: object,
    **kwargs: object,
) -> Artist | ErrorbarContainer:
    """Draw a line plot that accounts for uncertainties via error bars.

    Converts iterable inputs to numpy arrays, extracts nominal values and standard
    deviations when uncertainty variables are detected, and reuses the original
    matplotlib plot when no significant uncertainty is present. Error bars are added
    with sensible defaults when standard deviations are available.

    Args:
        self: Target axes for rendering.
        x: X-coordinates that may include uncertainty values.
        y: Y-coordinates that may include uncertainty values.
        *args: Additional positional arguments forwarded to matplotlib plot.
        **kwargs: Additional keyword arguments forwarded to matplotlib plot.

    Returns:
        Matplotlib line or errorbar container produced by the plotting call.

    Raises:
        TypeError: If inputs cannot be coerced to numeric arrays.
        ValueError: If input shapes are incompatible for plotting.

    Side Effects:
        None.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> ax = plt.subplot()
        >>> _ = ax.plot([0, 1], [0, 1])
    """
    # Preserve Matplotlib's complete calling convention except for a single x/y
    # series, where uncertainty values can be rendered as error bars.
    if len(args) < 2 or len(args) > 3 or isinstance(args[1], str):
        return _original_plot(self, *args, **kwargs)

    x, y = args[:2]
    fmt = args[2] if len(args) == 3 else ""
    x = [x] if not hasattr(x, "__iter__") else x
    y = [y] if not hasattr(y, "__iter__") else y
    x = np.asarray(x)
    y = np.asarray(y)

    try:
        x_nom = nominal_values(x)
        y_nom = nominal_values(y)
        # x_nom = nominal_values(x) if x_has_unc else np.asarray(x, dtype=float)
        # y_nom = nominal_values(y) if y_has_unc else np.asarray(y, dtype=float)
        x_std = std_devs(x)
        y_std = std_devs(y)
        # x_std = std_devs(x) if x_has_unc else None
        # y_std = std_devs(y) if y_has_unc else None
    except (TypeError, ValueError):
        x_nom = np.asarray(x, dtype=float)
        y_nom = np.asarray(y, dtype=float)
        x_std = None
        y_std = None

    min_visual_std = 2e-10

    plot_kwargs = kwargs.copy()

    if x_std is not None or y_std is not None:
        y_err = None
        x_err = None

        if y_std is not None:
            y_err = np.where(y_std > min_visual_std, y_std, 1)
            if np.all(y_err == 1):
                y_err = None

        if x_std is not None:
            x_err = np.where(x_std > min_visual_std, x_std, 1)
            if np.all(x_err == 1):
                x_err = None

        if x_err is None and y_err is None:
            return _original_plot(self, x_nom, y_nom, fmt, **plot_kwargs)

        errorbar_kwargs = {
            "capsize": 4,
            "capthick": 2.5,
            "elinewidth": 2.5,
            "markersize": 5,
            "alpha": 1,
        }
        errorbar_kwargs.update(plot_kwargs)

        return self.errorbar(
            x_nom,
            y_nom,
            xerr=x_err,
            yerr=y_err,
            fmt=fmt,
            **errorbar_kwargs,
        )
    else:
        return _original_plot(self, x_nom, y_nom, fmt, **plot_kwargs)


matplotlib.axes.Axes.plot = new_plot

_original_lambdify = sym.lambdify


def new_lambdify(
    x: sym.Symbol | Sequence[sym.Symbol],
    expr: sym.Expr | tuple[sym.Expr, sym.Expr],
    backend: str = "numpy",
    *args: object,
    **kwargs: object,
) -> Callable[..., Uncertain | np.ndarray | float | complex]:
    """Create a callable from a sympy expression with optional uncertainty handling.

    Wraps sympy's `lambdify`, optionally generating an uncertainty-aware function
    when `backend` is set to ``"unc"`` by analytically propagating symbol deviations.
    Falls back to the original `lambdify` for other backends.

    Args:
        x: Symbol or iterable of symbols used as function arguments.
        expr: Expression to convert or tuple of nominal and uncertainty expressions.
        backend: Backend identifier; when ``"unc"`` constructs uncertainty-aware callable.
        *args: Additional positional arguments forwarded to sympy.lambdify.
        **kwargs: Additional keyword arguments forwarded to sympy.lambdify.

    Returns:
        Callable evaluating the symbolic expression, possibly returning uncertainties.

    Raises:
        TypeError: Propagated from sympy if expressions or arguments are invalid.

    Examples:
        >>> x = sym.symbols("x")
        >>> f = sym.lambdify(x, x**2, "unc")
        >>> f(unc.ufloat(2, 0.1))
    """
    if backend == "unc":
        if not (hasattr(x, "__iter__")):
            x = [x]
        else:
            x = list(x)

        if isinstance(expr, tuple) and len(expr) == 2:
            expr_nom = expr[0]
            expr_std = expr[1]

        else:
            expr_nom = expr

            expr_std = FunctionBase1D._calculate_uncertainty_analyticaly(expr_nom, x)

            # .subs({**nominal_coefs_dict, **std_coefs_dict})

        args_nom = x
        args_std = [*args_nom, *(sym.Symbol(f"Delta_{arg}") for arg in args_nom)]

        # print(expr_std.free_symbols)
        func_std = _original_lambdify(args_std, expr_std, "numpy")
        func_nom = _original_lambdify(args_nom, expr_nom, "numpy")

        def unc_func(*values: ArrayLike | Numeric | Uncertain) -> Uncertain | np.ndarray:
            """Evaluate the uncertainty-aware expression.

            Args:
                x: Input values that may carry uncertainties.

            Returns:
                Evaluated result containing propagated uncertainties.
            """
            if len(values) != len(args_nom):
                raise TypeError(f"Expected {len(args_nom)} arguments, got {len(values)}.")

            nominal_args = [nominal_values(value) for value in values]
            std_args = [std_devs(value) for value in values]
            nominal = func_nom(*nominal_args)
            standard_deviation = np.abs(func_std(*nominal_args, *std_args))

            if all(np.ndim(value) == 0 for value in values):
                return unc.ufloat(float(nominal), float(standard_deviation))
            return unc.unumpy.uarray(nominal, standard_deviation)

        return unc_func

    else:
        return _original_lambdify(x, expr, modules=backend, *args, **kwargs)


# sym.lambdify = new_lambdify


_unc_attrs = {}


def get_unc_attr(
    obj: object, attr: str, default: object | None = None
) -> object | None:
    """Retrieve a stored uncertainty attribute for a sympy object.

    Args:
        obj: Object whose cached attribute is requested.
        attr: Attribute name to read.
        default: Fallback value when attribute is missing.

    Returns:
        Stored attribute value or the provided default.
    """
    obj_id = id(obj)
    return _unc_attrs.get(obj_id, {}).get(attr, default)


def set_unc_attr(obj: object, attr: str, value: object) -> None:
    """Attach an uncertainty-related attribute to a sympy object.

    Args:
        obj: Object receiving the attribute.
        attr: Attribute name to set.
        value: Value to store under the attribute name.
    """
    obj_id = id(obj)
    if obj_id not in _unc_attrs:
        _unc_attrs[obj_id] = {"is_unc": False, "added_unc": sym.S.Zero}
    _unc_attrs[obj_id][attr] = value
    return None


_original_subs = sym.Basic.subs


def new_subs(
    self: sym.Basic,
    arg1: dict[object, object] | None = None,
    arg2: object | None = None,
    **kwargs: object,
) -> sym.Basic:
    """Substitute values into a sympy expression with uncertainty propagation.

    Extends the default `subs` to detect uncertainty variables, separate nominal
    values and standard deviations, and attach propagated uncertainty metadata to
    the resulting expression. Falls back to the original substitution when no
    uncertainty inputs are detected.

    Args:
        self: Expression subject to substitution.
        arg1: Primary substitution mapping that may contain uncertainty values.
        arg2: Secondary substitutions passed to the original method.
        **kwargs: Additional keyword arguments forwarded to the original `subs`.

    Returns:
        Expression with substitutions applied, possibly annotated with uncertainty metadata.
    """
    if arg1 is None:
        arg1 = {}
    is_unc = get_unc_attr(self, "is_unc", False)
    added_unc = get_unc_attr(self, "added_unc", sym.S.Zero)

    unc_args = [
        key
        for key in arg1
        if (hasattr(arg1[key], "nominal_value") and hasattr(arg1[key], "std_dev"))
    ]

    if unc_args or is_unc:
        coefs_dict = arg1

        # separating nominal_values and standart deviations
        nominal_coefs_dict = {}
        std_coefs_dict = {}
        for key in coefs_dict:
            delta = sym.Symbol(f"Delta_{str(key)}")
            if hasattr(coefs_dict[key], "nominal_value") and hasattr(
                coefs_dict[key], "std_dev"
            ):
                nominal_coefs_dict[key] = coefs_dict[key].nominal_value
                std_coefs_dict[delta] = coefs_dict[key].std_dev
            else:
                nominal_coefs_dict[key] = coefs_dict[key]
                std_coefs_dict[delta] = 1e-20

        expr_std = FunctionBase1D._calculate_uncertainty_analyticaly(self, unc_args)
        expr_std = _original_subs(expr_std, {**nominal_coefs_dict, **std_coefs_dict})
        expr_nom = _original_subs(self, nominal_coefs_dict, arg2, **kwargs)

        result_is_unc = get_unc_attr(expr_nom, "is_unc", False)
        result_added_unc = get_unc_attr(expr_nom, "added_unc", sym.S.Zero)

        if not result_is_unc:
            set_unc_attr(expr_nom, "is_unc", True)
            set_unc_attr(expr_nom, "added_unc", expr_std)
        else:
            new_unc = sym.sqrt(result_added_unc**2 + expr_std**2)
            set_unc_attr(expr_nom, "added_unc", new_unc)

        return expr_nom

    else:
        return _original_subs(self, arg1, arg2, **kwargs)


# sym.Basic.subs = new_subs

def plot_with_uncertainty(
    ax: Axes,
    x,
    y,
    *,
    sigma=None,
    k: float = 1.96,
    mode: str = "line",
    show_band: bool = True,
    show_errors: bool = True,
    color=None,
    alpha: float = 0.20,
    label: str | None = None,
    zorder=None,
    scatter_kwargs=None,
    line_kwargs=None,
    error_kwargs=None,
    band_kwargs=None,
):
    """
    Рисует line/scatter + errorbars + confidence band с единым цветом.
    """

    scatter_kwargs = scatter_kwargs or {}
    line_kwargs = line_kwargs or {}
    error_kwargs = error_kwargs or {}
    band_kwargs = band_kwargs or {}

    x = np.asarray(x)
    y = np.asarray(y)

    # --- nominal / sigma ---
    if nominal_values is not None:
        try:
            x_nom = np.asarray(nominal_values(x), dtype=float)
        except Exception:
            x_nom = np.asarray(x, dtype=float)
    else:
        x_nom = np.asarray(x, dtype=float)

    if nominal_values is not None and std_devs is not None:
        try:
            y_nom = np.asarray(nominal_values(y), dtype=float)
            y_std_auto = np.asarray(std_devs(y), dtype=float)
            if np.all(y_std_auto == 0):
                y_std_auto = None
        except Exception:
            y_nom = np.asarray(y, dtype=float)
            y_std_auto = None
    else:
        y_nom = np.asarray(y, dtype=float)
        y_std_auto = None

    if sigma is None:
        sigma = y_std_auto

    # ---------- ОСНОВНОЙ PLOT (задаёт цвет) ----------
    artist = None

    if mode in ("line", "line+scatter"):
        (artist,) = ax.plot(
            x_nom,
            y_nom,
            label=label,
            color=color,
            zorder=zorder,
            **line_kwargs,
        )

    if mode in ("scatter", "line+scatter"):
        artist = ax.scatter(
            x_nom,
            y_nom,
            label=label if mode == "scatter" else None,
            color=color,
            zorder=zorder,
            **scatter_kwargs,
        )

    if artist is None:
        raise ValueError("mode должен быть 'line', 'scatter' или 'line+scatter'")

    # --- финальный цвет (из artist!) ---
    if hasattr(artist, "get_color"):
        base_color = artist.get_color()
    else:
        base_color = artist.get_facecolor()[0]

    # ---------- ERROR BARS ----------
    if show_errors and sigma is not None:
        ax.errorbar(
            x_nom,
            y_nom,
            yerr=k * np.asarray(sigma),
            fmt="none",
            ecolor=base_color,
            capsize=3,
            zorder=zorder,
            **error_kwargs,
        )

    # ---------- CONFIDENCE BAND ----------
    band = None
    if show_band and sigma is not None:
        sig = np.asarray(sigma, dtype=float)
        lower = y_nom - k * sig
        upper = y_nom + k * sig

        band = ax.fill_between(
            x_nom,
            lower,
            upper,
            color=base_color,
            alpha=alpha,
            label=None,  # <<< важно: не засоряем легенду
            zorder=zorder,
            **band_kwargs,
        )

    return {
        "artist": artist,
        "band": band,
    }
