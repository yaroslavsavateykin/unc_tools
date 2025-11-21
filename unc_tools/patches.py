from copy import deepcopy
import matplotlib
import numpy as np
from typing import Tuple, Union, List, Optional
import uncertainties as unc
from uncertainties.unumpy import nominal_values, std_devs, uarray


from unc_tools.default_functions import FunctionBase1D

_original_plot = matplotlib.axes.Axes.plot


def new_plot(
    self, x: Union[List, np.ndarray], y: Union[List, np.ndarray], *args, **kwargs
):
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
    x,
    expr: Union[
        sym.core.add.Add,
        sym.core.expr.Expr,
        Tuple[sym.core.add.Add, sym.core.expr.Expr],
    ],
    backend="numpy",
    *args,
    **kwargs,
):
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

        def unc_func(x):
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


def get_unc_attr(obj, attr, default=None):
    obj_id = id(obj)
    return _unc_attrs.get(obj_id, {}).get(attr, default)


def set_unc_attr(obj, attr, value):
    obj_id = id(obj)
    if obj_id not in _unc_attrs:
        _unc_attrs[obj_id] = {"is_unc": False, "added_unc": sym.S.Zero}
    _unc_attrs[obj_id][attr] = value


_original_subs = sym.Basic.subs


def new_subs(self, arg1: dict = {}, arg2=None, **kwargs):
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

        # Получаем текущие значения для результата
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
