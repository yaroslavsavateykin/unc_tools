import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List, Optional
import uncertainties
from uncertainties.unumpy import nominal_values, std_devs

_original_plot = matplotlib.axes.Axes.plot


def new_plot(
    self, x: Union[List, np.ndarray], y: Union[List, np.ndarray], *args, **kwargs
):
    x = [x] if not hasattr(x, "__iter__") else x
    y = [y] if not hasattr(y, "__iter__") else y
    x = np.asarray(x)
    y = np.asarray(y)

    x_has_unc = len(x) > 0 and any(hasattr(xi, "nominal_value") for xi in x)
    y_has_unc = len(y) > 0 and any(hasattr(yi, "nominal_value") for yi in y)

    try:
        x_nom = nominal_values(x) if x_has_unc else np.asarray(x, dtype=float)
        y_nom = nominal_values(y) if y_has_unc else np.asarray(y, dtype=float)
        x_std = std_devs(x) if x_has_unc else None
        y_std = std_devs(y) if y_has_unc else None
    except (TypeError, ValueError):
        # Fallback для обычных массивов
        x_nom = np.asarray(x, dtype=float)
        y_nom = np.asarray(y, dtype=float)
        x_std = None
        y_std = None

    min_visual_std = 1e-10

    plot_kwargs = kwargs.copy()
    if "color" not in plot_kwargs:
        try:
            prop_cycle = matplotlib.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key().get(
                "color", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
            )
            color_index = len(self.lines) % len(colors)
            plot_kwargs["color"] = colors[color_index]
        except (AttributeError, KeyError):
            color_index = len(self.lines) % 10
            plot_kwargs["color"] = f"C{color_index}"

    if x_std is not None or y_std is not None:
        y_err = None
        x_err = None

        if y_std is not None:
            y_err = np.where(y_std > min_visual_std, y_std, 0)
            if np.all(y_err == 0):
                y_err = None

        if x_std is not None:
            x_err = np.where(x_std > min_visual_std, x_std, 0)
            if np.all(x_err == 0):
                x_err = None

        if x_err is None and y_err is None:
            return _original_plot(self, x_nom, y_nom, *args, **plot_kwargs)

        errorbar_kwargs = {
            "capsize": 3,
            "capthick": 1.5,
            "elinewidth": 1.5,
            "markersize": 4,
            "alpha": 0.8,
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
