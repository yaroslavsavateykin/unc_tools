import numpy as np
import matplotlib.pyplot as plt
import uncertainties as unc
import uncertainties.unumpy as unp
import scipy.differentiate as diff
import scipy.optimize as opt
import warnings

from typing import Union


class UncRegression:
    """
    Класс для выполнения регрессионного анализа с поддержкой неопределенностей

    Attributes:
        x, y : исходные данные
        func : функция для фитирования
        coefs : коэффициенты с неопределенностями
        R2 : коэффициент детерминации
    """

    @staticmethod
    def latex_style():
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",  # Используем засечковый шрифт
                "font.serif": ["Computer Modern"],
                "text.latex.preamble": r"""
                \usepackage[utf8]{inputenc}
                \usepackage[russian]{babel}
                \usepackage[T2A]{fontenc}
            """,
                "pgf.texsystem": "xelatex",
            }
        )

    def __init__(self, x, y, func=None):
        """
        Инициализация класса с данными и автоматическое выполнение фитирования

        Parameters:
        x, y : input data (arrays, pandas Series, or uarrays)
        func : custom function f(x, *params), по умолчанию линейная
        """
        # Проверка на пустые входные данные
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Input arrays cannot be empty")

        # Конвертация в numpy массивы
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        # Функция по умолчанию - линейная
        if func is None:
            self.func = lambda x, k, b: k * x + b
        else:
            self.func = func

        # Проверка наличия неопределенностей
        self.x_has_unc = len(self.x) > 0 and hasattr(self.x[0], "nominal_value")
        self.y_has_unc = len(self.y) > 0 and hasattr(self.y[0], "nominal_value")

        # Извлечение номинальных значений и стандартных отклонений
        self._extract_values()

        # Автоматическое выполнение фитирования при инициализации
        self._fit()

    def _extract_values(self):
        """Извлечение номинальных значений и неопределенностей"""
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

        # Обработка нулевых неопределенностей
        self._handle_zero_uncertainties()

    def _handle_zero_uncertainties(self):
        """Замена нулевых неопределенностей на малые значения"""
        for arr, name in zip([self.x_std, self.y_std], ["x", "y"]):
            if arr is not None and np.any(arr == 0):
                non_zero = arr[arr > 0]
                replacement = np.min(non_zero) * 1e-5 if len(non_zero) > 0 else 1e-10
                arr[arr == 0] = replacement
                warnings.warn(
                    f"Zero uncertainties in {name} replaced with {replacement:.1e}"
                )

    def _fit(self):
        """
        Выполнение регрессионного анализа
        """
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
            # Попытка с начальным приближением
            n_params = self.func.__code__.co_argcount - 1
            self.popt = np.ones(n_params)
            self.pcov = np.eye(n_params)

        # Расчет R²
        residuals = self.y_nom - self.func(self.x_nom, *self.popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.y_nom - np.mean(self.y_nom)) ** 2)
        self.R2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Создание коэффициентов с неопределенностями
        try:
            self.coefs = unp.uarray(self.popt, np.sqrt(np.diag(self.pcov)))
        except:
            self.coefs = unp.uarray(self.popt, np.zeros_like(self.popt))

    @property
    def coefs_values(self):
        """
        Возвращает только значения коэффициентов без погрешностей

        Returns:
        array : значения коэффициентов
        """
        return unp.nominal_values(self.coefs)

    @property
    def coefs_errors(self):
        """
        Возвращает только погрешности коэффициентов

        Returns:
        array : погрешности коэффициентов
        """
        return unp.std_devs(self.coefs)

    def plot(
        self,
        figsize=(10, 5),
        labels=None,
        ax=None,
        path="",
        label="",
        show_errors=True,
        show_band=False,
        band_alpha=0.2,
        band_color=None,
        add_legend=True,
        show_coefficients=True,
        **kwargs,
    ):
        """
        Построение графика с результатами регрессии

        Parameters:
        figsize : размер фигуры
        labels : [xlabel, ylabel]
        ax : matplotlib axis (опционально)
        path : путь для сохранения графика
        label : префикс для легенды
        show_errors : показывать ошибки если есть неопределенности
        show_band : показывать доверительную полосу
        band_alpha : прозрачность доверительной полосы
        band_color : цвет доверительной полосы
        add_legend : добавлять легенду
        show_coefficients : показывать коэффициенты в легенде
        kwargs : дополнительные параметры для plot/errorbar

        Returns:
        ax : matplotlib axis
        """
        if labels is None:
            labels = ["", ""]

        # Создание расширенного диапазона для построения линии
        x_min, x_max = np.min(self.x_nom), np.max(self.x_nom)
        delta = (x_max - x_min) * 0.1
        x_ax = np.linspace(x_min - delta, x_max + delta, 500)
        y_ax = self.func(x_ax, *self.popt)

        # Обработка осей
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=150)
        else:
            fig = ax.figure

        # Подготовка стиля графика
        plot_kwargs = kwargs.copy()
        if "color" not in plot_kwargs:
            try:
                prop_cycle = ax._get_lines.prop_cycler
                color = next(prop_cycle)["color"]
                plot_kwargs["color"] = color
            except (AttributeError, StopIteration):
                color_index = len(ax.lines) % 10
                plot_kwargs["color"] = f"C{color_index}"

        # Расчет минимального визуального размера для error bars
        y_range = np.nanmax(self.y_nom) - np.nanmin(self.y_nom)
        min_visual_std = 0.02 * y_range if y_range > 0 else 1e-10

        # Построение данных с или без error bars
        if show_errors and (self.x_std is not None or self.y_std is not None):
            # Применение минимального визуального размера для error bars
            y_err = None
            x_err = None
            if self.y_std is not None:
                y_err = np.maximum(self.y_std, min_visual_std)
            if self.x_std is not None:
                x_err = np.maximum(self.x_std, min_visual_std)

            # Настройки по умолчанию для errorbar
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

            # Использование цвета errorbar для полосы если не задан
            if band_color is None and show_band:
                band_color = eb[0].get_color()
        else:
            # Создание scatter plot
            ax.scatter(self.x_nom, self.y_nom, label=label, **plot_kwargs)
            if band_color is None and show_band:
                band_color = plot_kwargs.get("color", "blue")

        # Построение линии регрессии с R² в легенде
        line_label = f"{label} R² = {self.R2:.4f}" if label else f"R² = {self.R2:.4f}"
        ax.plot(x_ax, y_ax, label=line_label, color=plot_kwargs.get("color"))

        # Добавление коэффициентов в легенду
        if show_coefficients:
            for i, coef in enumerate(self.coefs):
                ax.plot(
                    [],
                    [],
                    " ",
                    label=f"$p_{{{i}}}$ = {coef:.2u}".replace("+/-", r"$\pm$"),
                )

        # Построение доверительной полосы
        if show_band:
            try:
                # Создание коррелированных неопределенностей для параметров
                params = unc.correlated_values(self.popt, self.pcov)

                # Расчет неопределенностей для предсказаний
                y_vals = [self.func(x_val, *params) for x_val in x_ax]

                # Извлечение номинальных значений и стандартных отклонений
                y_nom_band = unp.nominal_values(y_vals)
                y_std_band = unp.std_devs(y_vals)

                # Построение доверительной полосы (95% CI)
                ax.fill_between(
                    x_ax,
                    y_nom_band - 1.96 * y_std_band,
                    y_nom_band + 1.96 * y_std_band,
                    alpha=band_alpha,
                    color=band_color,
                    label="95% CI" if not label else f"95% CI ({label})",
                )
            except Exception as e:
                warnings.warn(f"Confidence band could not be plotted: {e}")

        # Установка подписей и сетки
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.grid(True)

        # Добавление легенды
        if add_legend and (
            label or line_label or (show_coefficients and len(self.coefs) > 0)
        ):
            ax.legend()

        # Сохранение графика
        if path:
            if fig is None:
                fig = plt.gcf()
            fig.savefig(path, bbox_inches="tight")

        return ax

    def predict(self, x_new):
        """
        Предсказание значений для новых x

        Parameters:
        x_new : новые значения x

        Returns:
        y_pred : предсказанные значения
        """
        return self.func(x_new, *self.popt)

    def predict_with_uncertainty(self, x_new):
        """
        Получение неопределенностей предсказаний для новых x

        Parameters:
        x_new : новые значения x

        Returns:
        y_pred : предсказанные значения с неопределенностями (uarray)
        """
        # Создание коррелированных неопределенностей для параметров
        params = unc.correlated_values(self.popt, self.pcov)

        # Расчет предсказаний с неопределенностями
        y_pred = [self.func(x_val, *params) for x_val in x_new]

        return np.array(y_pred)

    def find_x(
        self,
        y: Union[unc.core.Variable, float],
        x0,
        xtol_root=1e-20,
        xtol_diff=1e-20,
        **kwargs,
    ):
        args_nominal = self.coefs_values

        if isinstance(y, unc.core.Variable):
            y = y.nominal_value
            ytol = y.std_dev
        else:
            y = y
            ytol = 0

        def func(x):
            return self.func(x, *args_nominal) - y

        result_root = opt.root_scalar(f=func, x0=x0, xtol=xtol_root)

        if not result_root.converged:
            raise TypeError("Root not converged")

        result_diff = diff.derivative(
            f=func,
            x=result_root.root,
            tolerances={"atol": xtol_diff},
        )

        if not result_diff.success:
            error = {
                0: "The algorithm converged to the specified tolerances.",
                -1: "The error estimate increased, so iteration was terminated.",
                -2: "The maximum number of iterations was reached.",
                -3: "A non-finite value was encountered.",
                -4: "Iteration was terminated by callback.",
                1: "The algorithm is proceeding normally (in callback only).",
            }
            raise TypeError(error[result_diff.status])

        dy = result_diff.df
        dytol = (
            result_diff.error
        )  # пока не понял, как правильно учитывать погрешность дифференцирования
        dxtol = ytol / dy

        xtol_summ = np.sqrt(xtol_root**2 + xtol_diff**2 + dxtol**2)

        return unc.ufloat(result_root.root, xtol_summ)
