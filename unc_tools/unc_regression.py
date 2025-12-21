import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties as unc
import uncertainties.unumpy as unp
import scipy.differentiate as diff
import scipy.optimize as opt
import warnings
import sympy as sym

from typing import Callable, Union, Any, Sequence, Optional

from .default_functions import FunctionBase1D, Poly


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
    def latex_style(tex: bool) -> None:
        """Настроить стиль matplotlib для вывода LaTeX.

        Args:
            tex (bool): Флаг использования LaTeX при рендеринге текста.

        Returns:
            None

        Raises:
            None.

        Side Effects:
            Изменяет глобальные параметры `matplotlib.rcParams`.
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
                    "pgf.texsystem": None,  # или можно удалить этот параметр
                }
            )

    def __init__(
        self,
        x: Union[Sequence[Any], np.ndarray, pd.Series],
        y: Union[Sequence[Any], np.ndarray, pd.Series],
        func: Union[Callable[..., Any], FunctionBase1D, None] = None,
    ) -> None:
        """Инициализировать регрессию с поддержкой неопределенностей.

        Принимает входные данные, выбирает модель (пользовательскую или полиномиальную
        по умолчанию), извлекает номинальные значения/погрешности и запускает процесс
        фитирования.

        Args:
            x (Sequence[Any] | np.ndarray | pd.Series): Значения аргумента.
            y (Sequence[Any] | np.ndarray | pd.Series): Значения функции.
            func (Callable[..., Any] | FunctionBase1D | None): Пользовательская модель
                или объект `FunctionBase1D`; если None, используется линейная модель.

        Returns:
            None

        Raises:
            TypeError: Если тип функции неподдерживаемый.
            ValueError: Если входные массивы пусты.

        Side Effects:
            Выполняет фитирование и инициализирует внутренние метрики.
        """
        if issubclass(type(func), FunctionBase1D):
            self.func = func.lambda_fun
            self.expression = func
        elif isinstance(func, Callable):
            self.func = func
            self.expression = None
        elif func is None:
            self.expression = Poly(1)
            self.func = self.expression.lambda_fun
        else:
            raise TypeError(
                "func argument must be either Callable or child class of FunctionBase1D"
            )

        if len(x) == 0 or len(y) == 0:
            raise ValueError("Input arrays cannot be empty")

        self.x = np.asarray(x)
        self.y = np.asarray(y)

        self.x_has_unc = len(self.x) > 0 and hasattr(self.x[0], "nominal_value")
        self.y_has_unc = len(self.y) > 0 and hasattr(self.y[0], "nominal_value")

        self._extract_values()

        self._fit()

    def _extract_values(self) -> None:
        """Извлечь номинальные значения и неопределенности входных данных.

        Преобразует входные массивы к numpy, отделяет номинальные значения и
        стандартные отклонения, обрабатывая возможные ошибки приведения типов.

        Args:
            None.

        Returns:
            None

        Raises:
            TypeError: Если элементы нельзя преобразовать к числам.

        Side Effects:
            Устанавливает атрибуты `x_nom`, `y_nom`, `x_std`, `y_std`.
        """
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

    def _handle_zero_uncertainties(self) -> None:
        """Заменить нулевые неопределенности на малые значения.

        Проходит по рассчитанным стандартным отклонениям и подставляет минимальные
        ненулевые значения, предотвращая проблемы при последующих вычислениях.

        Args:
            None.

        Returns:
            None

        Raises:
            None.

        Side Effects:
            Модифицирует массивы неопределенностей и выводит предупреждения.
        """
        for arr, name in zip([self.x_std, self.y_std], ["x", "y"]):
            if arr is not None and np.any(arr == 0):
                non_zero = arr[arr > 0]
                replacement = np.min(non_zero) * 1e-5 if len(non_zero) > 0 else 1e-15
                arr[arr == 0] = replacement
                warnings.warn(
                    f"Zero uncertainties in {name} replaced with {replacement:.1e}"
                )

    def _fit(self) -> None:
        """Выполнить регрессионный анализ с учетом неопределенностей.

        Пытается выполнить взвешенный фит, оценивает остатки и коэффициент детерминации
        и сохраняет параметры как массивы неопределенных значений.

        Args:
            None.

        Returns:
            None

        Raises:
            RuntimeError: Если оптимизация не сходится (перехватывается и переводится в предупреждение).
            TypeError: Если модель не принимает переданные аргументы.

        Side Effects:
            Изменяет атрибуты `popt`, `pcov`, `coefs`, `R2` и может добавлять коэффициенты в выражение.
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
        except:
            if self.expression:
                self.expression.add_coefs(self.popt)
            self.coefs = unp.uarray(self.popt, np.zeros_like(self.popt))

    @property
    def coefs_values(self) -> np.ndarray:
        """Вернуть номинальные значения коэффициентов без погрешностей.

        Returns:
            np.ndarray: Массив номинальных значений коэффициентов.

        Raises:
            None.

        Side Effects:
            None.
        """
        return unp.nominal_values(self.coefs)

    @property
    def coefs_errors(self) -> np.ndarray:
        """Вернуть только погрешности коэффициентов.

        Returns:
            np.ndarray: Стандартные отклонения коэффициентов.

        Raises:
            None.

        Side Effects:
            None.
        """
        return unp.std_devs(self.coefs)

    def plot(
        self,
        figsize: tuple[float, float] = (10, 5),
        labels: Optional[Sequence[str]] = None,
        ax: Optional[plt.Axes] = None,
        path: str = "",
        label: str = "",
        show_errors: bool = True,
        show_band: bool = False,
        show_scatter: bool = True,
        band_alpha: float = 0.2,
        band_color: Optional[str] = None,
        add_legend: bool = True,
        show_expr: bool = True,
        show_coefficients: bool = False,
        show_r2: bool = True,
        **kwargs: Any,
    ) -> plt.Axes:
        """Построить график результатов регрессии.

        Строит линию регрессии, опционально отображает точки с погрешностями, полосу
        доверия и текстовые аннотации с выражением, коэффициентами и значением R².

        Args:
            figsize (tuple[float, float]): Размер фигуры в дюймах.
            labels (Sequence[str] | None): Подписи осей ``[xlabel, ylabel]``.
            ax (plt.Axes | None): Существующая ось для построения; если None, создается новая.
            path (str): Путь сохранения рисунка; пустая строка отключает сохранение.
            label (str): Префикс для подписей в легенде.
            show_errors (bool): Отображать error bars при наличии неопределенностей.
            show_band (bool): Показывать доверительную полосу.
            show_scatter (bool): Показывать точки наблюдений.
            band_alpha (float): Прозрачность доверительной полосы.
            band_color (str | None): Цвет доверительной полосы.
            add_legend (bool): Добавлять легенду.
            show_expr (bool): Отображать аналитическое выражение в легенде.
            show_coefficients (bool): Выводить коэффициенты в легенде.
            show_r2 (bool): Добавлять значение R² в подписи.
            **kwargs (Any): Дополнительные параметры для `plot` или `errorbar`.

        Returns:
            plt.Axes: Ось с построенным графиком.

        Raises:
            TypeError: Если выражение не задано при запросе отображения.
            RuntimeError: Если сохранение файла завершается ошибкой.

        Side Effects:
            Создает график, может сохранять файл и выводить предупреждения.

        Examples:
            >>> reg = UncRegression([0, 1], [0, 1])
            >>> _ = reg.plot(show_band=False)
        """
        if labels is None:
            labels = ["", ""]

        # Создание расширенного диапазона для построения линии
        x_min, x_max = np.min(self.x_nom), np.max(self.x_nom)
        delta = (x_max - x_min) * 0.0001
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
        if show_scatter:
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
                ax.scatter(self.x_nom, self.y_nom, **plot_kwargs)
                if band_color is None and show_band:
                    band_color = plot_kwargs.get("color", "blue")

        if show_r2:
            r2_str = f"\nR²= {self.R2:.5f}"
        else:
            r2_str = ""

        if show_expr:
            if self.expression:
                latex = self.expression.to_latex_expr()
                line_label = f"{label}\n{latex}{r2_str}"
            else:
                raise TypeError("Задайте функцию при помощи FunctionBase1D")
        elif show_coefficients and not show_expr:
            coefs = self.expression.args
            line_label = (
                "\n".join(
                    f"${sym.latex(coefs[i])} = {self.coefs[i]:.2u}$".replace(
                        "+/-", r"\pm"
                    )
                    for i in range(len(self.coefs))
                )
                + r2_str
            )
        else:
            line_label = r2_str.strip("\n")

        ax.plot(x_ax, y_ax, label=line_label, color=plot_kwargs.get("color"))

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

    def predict(self, x_new: Any) -> Any:
        """Предсказать значения для новых данных x.

        Args:
            x_new (Any): Значения аргумента для прогноза.

        Returns:
            Any: Предсказанные значения модели.

        Raises:
            TypeError: Если модель не принимает переданные аргументы.

        Side Effects:
            None.

        Examples:
            >>> reg = UncRegression([0, 1], [0, 1])
            >>> reg.predict([0.5])
        """
        return self.func(x_new, *self.popt)

    def predict_with_uncertainty(self, x_new: Any) -> np.ndarray:
        """Получить предсказания с учетом неопределенностей параметров.

        Args:
            x_new (Any): Значения аргумента для оценки.

        Returns:
            np.ndarray: Предсказанные значения с неопределенностями.

        Raises:
            TypeError: Если модель не принимает переданные аргументы.

        Side Effects:
            None.

        Examples:
            >>> reg = UncRegression([0, 1], [0, 1])
            >>> reg.predict_with_uncertainty([0.5])
        """
        # Создание коррелированных неопределенностей для параметров
        params = unc.correlated_values(self.popt, self.pcov)

        # Расчет предсказаний с неопределенностями
        y_pred = [self.func(x_val, *params) for x_val in x_new]

        return np.array(y_pred)

    def find_x(
        self,
        y: Union[unc.core.Variable, float],
        x0: Union[unc.core.Variable, float, None] = None,
        xtol_root: float = 1e-20,
        xtol_diff: float = 1e-20,
        maxiter: int = 500,
        solve_numerically: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Найти значение x для заданного y с учетом неопределенности.

        Использует аналитические решения при наличии выражения или численное решение
        с помощью `scipy.optimize.root_scalar`, оценивая погрешность результата.

        Args:
            y (unc.core.Variable | float): Значение функции, может содержать погрешность.
            x0 (unc.core.Variable | float | None): Начальное приближение для численного решения.
            xtol_root (float): Допуск для метода root-finding.
            xtol_diff (float): Допуск для численного дифференцирования.
            maxiter (int): Максимальное число итераций.
            solve_numerically (bool): Принудительно использовать численный метод.
            **kwargs (Any): Дополнительные аргументы для `root_scalar`.

        Returns:
            Any: Аналитическое решение, список решений или `ufloat` для численного случая.

        Raises:
            TypeError: Если не задано начальное значение x0 или решение не сходится.

        Side Effects:
            None.

        Examples:
            >>> reg = UncRegression([0, 1], [0, 1])
            >>> reg.find_x(unc.ufloat(0.5, 0.01), x0=0.5)
        """
        args_nominal = self.coefs_values

        if isinstance(y, unc.core.Variable):
            yval = y.nominal_value
            ytol = y.std_dev
        else:
            yval = y
            ytol = 0

        if isinstance(x0, unc.core.Variable):
            xtol = x0.std_dev
            x0 = x0.nominal_value
        else:
            xtol = 0
            x0 = x0

        if self.expression and not solve_numerically:
            sols = self.expression.find_sols(y=yval)

            if hasattr(sols, "__iter__"):
                if len(sols) == 1:
                    return sols[0]
                else:
                    return sols
            else:
                return sols

        else:
            if x0 is None:
                raise TypeError("Setup the x0")

            def func(x: float, *args: Any) -> float:
                """Вычислить разницу между моделью и целевым значением.

                Args:
                    x (float): Значение аргумента.
                    *args: Коэффициенты модели.

                Returns:
                    float: Разница между значением модели и целевым значением.

                Raises:
                    None.

                Side Effects:
                    None.
                """
                return self.func(x, *args) - yval

            result_root = opt.root_scalar(
                f=func,
                x0=x0,
                xtol=xtol_root,
                maxiter=maxiter,
                args=tuple(args_nominal),
                **kwargs,
            )

            if not result_root.converged:
                raise TypeError(f"Root not converged: {result_root.flag}")

            result_diff = diff.derivative(
                f=func,
                x=result_root.root,
                args=tuple(args_nominal),
                tolerances={"atol": xtol_diff},
                maxiter=maxiter,
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

            xtol_summ = np.sqrt(xtol_root**2 + xtol_diff**2 + dxtol**2 + xtol**2)

            return unc.ufloat(result_root.root, xtol_summ)

    def to_df(self, export_plot: bool = False) -> pd.DataFrame:
        """Преобразовать результаты регрессии в DataFrame.

        Формирует таблицу исходных данных и неопределенностей или, при запросе,
        возвращает сетку для построения кривой фита.

        Args:
            export_plot (bool): Если True, возвращает данные для построения линии фита.

        Returns:
            pd.DataFrame: Таблица данных или точки аппроксимации.

        Raises:
            None.

        Side Effects:
            None.
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
        filename: Optional[str] = None,
        export_plot: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Экспортировать результаты регрессии в CSV файл.

        Args:
            filename (str | None): Имя файла; при None генерируется случайное.
            export_plot (bool): Экспортировать данные аппроксимации вместо исходных.
            *args: Дополнительные позиционные аргументы для `DataFrame.to_csv`.
            **kwargs: Дополнительные ключевые аргументы для `DataFrame.to_csv`.

        Returns:
            None

        Raises:
            OSError: При ошибке записи файла.

        Side Effects:
            Создает CSV-файл на диске и выводит сообщение о сохранении при автогенерации имени.
        """
        df = self.to_df(export_plot=export_plot)

        if not filename:
            filename = f"{str(uuid.uuid4())[:7]}.csv"
            print(f"DataFrame was saved to {filename}")

        df.to_csv(filename, *args, **kwargs)
