from copy import deepcopy
import matplotlib
import numpy as np
from typing import Tuple, Union, List, Optional, Any, Callable
import uncertainties as unc
from uncertainties.unumpy import nominal_values, std_devs, uarray


from unc_tools.default_functions import FunctionBase1D

_original_scatter = matplotlib.axes.Axes.scatter


def new_scatter(
    self,
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Create a scatter plot that visualizes uncertainties when present.

    Converts inputs to numpy arrays, extracts nominal values and standard deviations
    when `uncertainties` variables are provided, and falls back to the original
    matplotlib scatter if no meaningful uncertainty is available. Error bars are
    added automatically when uncertainty magnitudes exceed a minimal visual threshold.

    Args:
        self (matplotlib.axes.Axes): Target axes for rendering.
        x (list | np.ndarray): X-coordinates, optionally containing uncertainty values.
        y (list | np.ndarray): Y-coordinates, optionally containing uncertainty values.
        *args: Additional positional arguments forwarded to matplotlib scatter.
        **kwargs: Additional keyword arguments forwarded to matplotlib scatter.

    Returns:
        Any: Matplotlib artist or container returned by the scatter/errorbar call.

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

    x_has_unc = any(isinstance(xi, unc.core.Variable) for xi in x)
    y_has_unc = any(isinstance(yi, unc.core.Variable) for yi in y)

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
    self,
    x: Union[List, np.ndarray],
    y: Union[List, np.ndarray],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Draw a line plot that accounts for uncertainties via error bars.

    Converts iterable inputs to numpy arrays, extracts nominal values and standard
    deviations when uncertainty variables are detected, and reuses the original
    matplotlib plot when no significant uncertainty is present. Error bars are added
    with sensible defaults when standard deviations are available.

    Args:
        self (matplotlib.axes.Axes): Target axes for rendering.
        x (list | np.ndarray): X-coordinates that may include uncertainty values.
        y (list | np.ndarray): Y-coordinates that may include uncertainty values.
        *args: Additional positional arguments forwarded to matplotlib plot.
        **kwargs: Additional keyword arguments forwarded to matplotlib plot.

    Returns:
        Any: Matplotlib line or errorbar container produced by the plotting call.

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
    x = [x] if not hasattr(x, "__iter__") else x
    y = [y] if not hasattr(y, "__iter__") else y
    x = np.asarray(x)
    y = np.asarray(y)

    x_has_unc = any(isinstance(xi, unc.core.Variable) for xi in x)
    y_has_unc = any(isinstance(yi, unc.core.Variable) for yi in y)

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
    if "color" not in plot_kwargs:
        try:
            prop_cycle = matplotlib.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key().get(
                "color", ["C1", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
            )
            color_index = len(self.lines) % len(colors)
            plot_kwargs["color"] = colors[color_index]
        except (AttributeError, KeyError):
            color_index = len(self.lines) % 11
            plot_kwargs["color"] = f"C{color_index}"

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
            return _original_plot(self, x_nom, y_nom, *args, **plot_kwargs)

        errorbar_kwargs = {
            "capsize": 4,
            "capthick": 2.5,
            "elinewidth": 2.5,
            "markersize": 5,
            "alpha": 1,
        }
        errorbar_kwargs.update(plot_kwargs)

        for arg in ["marker", "linestyle", "linewidth"]:
            if arg in errorbar_kwargs:
                del errorbar_kwargs[arg]

        return self.errorbar(
            x_nom,
            y_nom,
            xerr=x_err,
            yerr=y_err,
            **errorbar_kwargs,
        )
    else:
        return _original_plot(self, x_nom, y_nom, *args, **plot_kwargs)


matplotlib.axes.Axes.plot = new_plot

import sympy as sym

_original_lambdify = sym.lambdify


def new_lambdify(
    x: Any,
    expr: Union[
        sym.core.add.Add,
        sym.core.expr.Expr,
        Tuple[sym.core.add.Add, sym.core.expr.Expr],
    ],
    backend: str = "numpy",
    *args: Any,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Create a callable from a sympy expression with optional uncertainty handling.

    Wraps sympy's `lambdify`, optionally generating an uncertainty-aware function
    when `backend` is set to ``"unc"`` by analytically propagating symbol deviations.
    Falls back to the original `lambdify` for other backends.

    Args:
        x (Any): Symbol or iterable of symbols used as function arguments.
        expr (sym.core.add.Add | sym.core.expr.Expr | tuple[sym.core.add.Add, sym.core.expr.Expr]): Expression to convert or tuple of nominal and uncertainty expressions.
        backend (str): Backend identifier; when ``"unc"`` constructs uncertainty-aware callable.
        *args: Additional positional arguments forwarded to sympy.lambdify.
        **kwargs: Additional keyword arguments forwarded to sympy.lambdify.

    Returns:
        Callable[..., Any]: Callable evaluating the symbolic expression, possibly returning uncertainties.

    Raises:
        TypeError: Propagated from sympy if expressions or arguments are invalid.

    Side Effects:
        Prints diagnostic information when using the uncertainty backend.

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
            args = []

            expr_nom = expr

            expr_std = FunctionBase1D._calculate_uncertainty_analyticaly(expr_nom, x)

            # .subs({**nominal_coefs_dict, **std_coefs_dict})

        args_nom = x
        args_std = deepcopy(args_nom)
        for arg in args_nom:
            args_std.append(sym.Symbol(f"Delta_{str(arg)}"))

        # print(expr_std.free_symbols)
        func_std = _original_lambdify(args_std, expr_std, "numpy")
        func_nom = _original_lambdify(args_nom, expr_nom, "numpy")

        def unc_func(x: Union[List, np.ndarray, Any]) -> Any:
            """Evaluate the uncertainty-aware expression.

            Args:
                x (list | np.ndarray | Any): Input values that may carry uncertainties.

            Returns:
                Any: Evaluated result containing propagated uncertainties.

            Raises:
                None.

            Side Effects:
                Prints nominal and uncertainty components for debugging purposes.
            """
            x = [x] if not hasattr(x, "__iter__") else x
            x = np.asarray(x)

            x_has_unc = any(isinstance(xi, unc.core.Variable) for xi in x)

            x_nom = nominal_values(x) if x_has_unc else np.asarray(x, dtype=float)
            x_std = std_devs(x) if x_has_unc else np.ones(np.shape(x_nom)) * 1e-30

            # x_std = np.stack((x_nom, x_std), axis=1)
            # x_std = np.concatenate((x_nom, x_std), axis=1)

            print(f"x = {x}")
            print(f"x_std = {x_std}")
            print(f"x_nom = {x_nom}")

            print(f"{expr_nom} : {func_nom(x)}")
            print(f"{expr_std} : {func_std(x_nom, x_std)}")

            if len(x) == 1:
                return unc.ufloat(func_nom(x_nom), func_std(x_std))
            else:
                y1 = func_nom(x)
                y2 = unc.ufloat(0, func_std(x_nom, x_std))

                return y1 + y2

        return unc_func

    else:
        return _original_lambdify(x, expr=expr, *args, **kwargs)


sym.lambdify = new_lambdify


sym.core.cache.use_cache = False

_unc_attrs = {}


def get_unc_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Retrieve a stored uncertainty attribute for a sympy object.

    Args:
        obj (Any): Object whose cached attribute is requested.
        attr (str): Attribute name to read.
        default (Any, optional): Fallback value when attribute is missing.

    Returns:
        Any: Stored attribute value or the provided default.

    Raises:
        None.

    Side Effects:
        None.
    """
    obj_id = id(obj)
    return _unc_attrs.get(obj_id, {}).get(attr, default)


def set_unc_attr(obj: Any, attr: str, value: Any) -> None:
    """Attach an uncertainty-related attribute to a sympy object.

    Ensures a tracking dictionary exists for the object before assignment.

    Args:
        obj (Any): Object receiving the attribute.
        attr (str): Attribute name to set.
        value (Any): Value to store under the attribute name.

    Returns:
        None

    Raises:
        None.

    Side Effects:
        Mutates the module-level `_unc_attrs` cache.
    """
    obj_id = id(obj)
    if obj_id not in _unc_attrs:
        _unc_attrs[obj_id] = {"is_unc": False, "added_unc": sym.S.Zero}
    _unc_attrs[obj_id][attr] = value


_original_subs = sym.Basic.subs


def new_subs(self, arg1: dict[Any, Any] = {}, arg2: Any = None, **kwargs: Any) -> Any:
    """Substitute values into a sympy expression with uncertainty propagation.

    Extends the default `subs` to detect `uncertainties` variables, separate nominal
    values and standard deviations, and attach propagated uncertainty metadata to the
    resulting expression. Falls back to the original substitution when no uncertainty
    inputs are detected.

    Args:
        self (sym.Basic): Expression subject to substitution.
        arg1 (dict): Primary substitution mapping that may contain uncertainty values.
        arg2 (Any, optional): Secondary substitutions passed to the original method.
        **kwargs: Additional keyword arguments forwarded to the original `subs`.

    Returns:
        Any: Expression with substitutions applied, possibly annotated with uncertainty metadata.

    Raises:
        TypeError: If the provided coefficients mismatch expected structure.

    Side Effects:
        Updates module-level uncertainty cache for returned expressions.
    """
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


sym.Basic.subs = new_subs

sym.core.cache.use_cache = True
