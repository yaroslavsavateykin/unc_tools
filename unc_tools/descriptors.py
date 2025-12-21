import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from uncertainties import umath as um
from uncertainties import unumpy as unp
from uncertainties import ufloat
import uncertainties as unc
import warnings
from typing import Any, Sequence, Callable, Union


def calculate_r2(
    x: Sequence[Any] | np.ndarray, y: Sequence[Any] | np.ndarray
) -> float:
    """Calculate the R-squared statistic for a linear fit.

    Converts uncertainty-aware inputs to nominal values when needed, fits a linear
    model ``y = k*x + b`` using `scipy.optimize.curve_fit`, and computes the
    coefficient of determination.

    Args:
        x (Sequence[Any] | np.ndarray): Input x-values, possibly containing uncertainties.
        y (Sequence[Any] | np.ndarray): Input y-values, possibly containing uncertainties.

    Returns:
        float: Coefficient of determination for the fitted line.

    Raises:
        RuntimeError: Propagated if curve fitting fails to converge.
        TypeError: If inputs cannot be interpreted as numeric sequences.

    Side Effects:
        None.

    Examples:
        >>> calculate_r2([0, 1], [0, 1])
        1.0
    """
    # Extract nominal values if uncertainties are present
    try:
        y_nominal = [val.nominal_value for val in y]
        x_nominal = [val.nominal_value for val in x]
    except AttributeError:
        y_nominal = y
        x_nominal = x

    # Perform linear fit
    popt, pcov = curve_fit(lambda x, k, b: k * x + b, x_nominal, y_nominal)

    # Calculate predicted values
    y_pred = popt[0] * np.array(x_nominal) + popt[1]

    # Calculate R-squared
    y_true = np.array(y_nominal)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)


def plot_r2_heatmap(
    r2_df: pd.DataFrame,
    figsize: tuple[int, int] = (10, 10),
    dpi: int = 100,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    annotate: bool = True,
    annot_fmt: str = ".2f",
    title: str = "Тепловая карта $R^2$ между столбцами",
    path: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> plt.Axes:
    """Построить тепловую карту значений R² с дополнительными настройками.

    Создает матплотлиб-фигуру или использует переданную ось, отображает матрицу R²,
    при необходимости подписывает ячейки и сохраняет результат на диск.

    Args:
        r2_df (pd.DataFrame): Квадратная матрица значений R².
        figsize (tuple[int, int]): Размер фигуры в дюймах.
        dpi (int): Разрешение изображения.
        cmap (str): Цветовая карта для отображения.
        vmin (float): Минимальное значение цветовой шкалы.
        vmax (float): Максимальное значение цветовой шкалы.
        annotate (bool): Флаг отображения значений в ячейках.
        annot_fmt (str): Формат чисел для аннотаций.
        title (str): Заголовок графика.
        path (str | None): Путь сохранения изображения; если None, не сохраняется.
        ax (plt.Axes | None): Существующая ось; при None создается новая.
        **kwargs (Any): Дополнительные аргументы, передаваемые в `plt.imshow`.

    Returns:
        plt.Axes: Ось с построенной теплокартой.

    Raises:
        ValueError: Если входная матрица не квадратная или содержит несовместимые данные.

    Side Effects:
        Может создавать новую фигуру и сохранять файл на диск.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]])
        >>> ax = plot_r2_heatmap(df)
    """
    save_path = path
    # Создаем новую фигуру если ось не передана
    if ax is None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()

    # Проверка на симметричность матрицы
    if not np.allclose(r2_df.values, r2_df.values.T):
        print("Предупреждение: Матрица R² не симметрична")

    # Создаем тепловую карту
    im = ax.imshow(
        r2_df.values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
        **kwargs,
    )

    # Добавляем цветовую шкалу
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("$R^2$", rotation=270, labelpad=20)

    # Настраиваем подписи осей
    ax.set_xticks(np.arange(len(r2_df.columns)))
    ax.set_xticklabels(r2_df.columns, rotation=90)
    ax.set_yticks(np.arange(len(r2_df.index)))
    ax.set_yticklabels(r2_df.index)

    # Добавляем аннотации (значения в ячейках)
    if annotate:
        for i in range(len(r2_df.index)):
            for j in range(len(r2_df.columns)):
                ax.text(
                    j,
                    i,
                    format(r2_df.iloc[i, j], annot_fmt),
                    ha="center",
                    va="center",
                    color="w" if r2_df.iloc[i, j] < (vmax - vmin) / 2 else "k",
                )

    # Добавляем заголовок
    ax.set_title(title, pad=20)

    # Оптимизация расположения элементов
    plt.tight_layout()

    # Сохранение если указан путь
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)

    return ax


def plot_r2_heatmap_plotly(
    r2_df: pd.DataFrame, path: str, title: str = "Тепловая карта R²"
) -> None:
    """Строит интерактивную теплокарту R² в plotly.

    Использует `plotly.graph_objects.Heatmap` для визуализации матрицы R² и сохраняет
    результат в HTML или PNG в зависимости от расширения переданного пути.

    Args:
        r2_df (pd.DataFrame): Квадратная матрица R².
        path (str): Путь для сохранения HTML или PNG файла.
        title (str): Заголовок графика.

    Returns:
        None

    Raises:
        ValueError: Если формат пути не поддерживается библиотекой plotly.

    Side Effects:
        Создает файл на диске и открывает интерактивное окно отображения.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]])
        >>> plot_r2_heatmap_plotly(df, "heatmap.html")
    """
    # Убедимся, что импортирован go
    # import plotly.graph_objects as go  # <-- уже сделано вверху

    # Метки по осям
    columns = r2_df.columns.tolist()

    fig = go.Figure(
        data=go.Heatmap(
            z=r2_df.values,
            x=columns,
            y=columns,
            colorscale="Viridis",
            colorbar=dict(title="R²"),
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(ticks="", side="bottom"),
        yaxis=dict(ticks=""),
        width=800,
        height=700,
    )

    if path.endswith(".html"):
        fig.write_html(path)
    else:
        fig.write_png(path)

    fig.show()


def remove_unifrom_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Удалить столбцы с одинаковыми значениями по всей выборке.

    Args:
        df (pd.DataFrame): Входной DataFrame.

    Returns:
        pd.DataFrame: Новый DataFrame без постоянных столбцов.

    Raises:
        None.

    Side Effects:
        None.

    Examples:
        >>> import pandas as pd
        >>> remove_unifrom_columns(pd.DataFrame({'a':[1,1], 'b':[1,2]}))
    """
    # Определяем столбцы, у которых количество уникальных значений равно 1
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]

    # Удаляем эти столбцы
    return df.drop(columns=constant_cols)


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Удалить из DataFrame столбцы-дубликаты.

    Args:
        df (pd.DataFrame): Входной DataFrame.

    Returns:
        pd.DataFrame: DataFrame без полностью совпадающих столбцов.

    Raises:
        None.

    Side Effects:
        None.

    Examples:
        >>> import pandas as pd
        >>> remove_duplicate_columns(pd.DataFrame({'a':[1,2], 'b':[1,2]}))
    """
    # Используем .T.duplicated(), чтобы найти дублирующиеся строки в транспонированном DataFrame,
    # что соответствует дублирующим столбцам в исходном.
    mask = ~df.T.duplicated(keep="first")
    return df.loc[:, mask]


def batch_r2(
    df: pd.DataFrame,
    target: Union[np.ndarray, pd.Series, pd.DataFrame],
) -> pd.DataFrame:
    """Вычислить R² для каждого столбца DataFrame относительно цели.

    Нормализует тип цели к одномерному numpy-массиву и вычисляет коэффициент
    детерминации для каждого столбца входного DataFrame.

    Args:
        df (pd.DataFrame): Исходные данные признаков.
        target (np.ndarray | pd.Series | pd.DataFrame): Целевой вектор или столбец.

    Returns:
        pd.DataFrame: DataFrame со значениями R² для каждого столбца.

    Raises:
        ValueError: Если размерности цели и признаков не совпадают.
        TypeError: Если тип цели не поддерживается.

    Side Effects:
        None.

    Examples:
        >>> import pandas as pd, numpy as np
        >>> batch_r2(pd.DataFrame({'a':[1,2]}), np.array([1,2]))
    """
    # Приводим target к 1D numpy-массиву
    if isinstance(target, pd.DataFrame):
        if target.shape[1] != 1:
            raise ValueError(
                "Если target — DataFrame, он должен содержать ровно один столбец."
            )
        target_values = target.iloc[:, 0].values
    elif isinstance(target, pd.Series):
        target_values = target.values
    elif isinstance(target, np.ndarray):
        if target.ndim == 2 and target.shape[1] == 1:
            target_values = target.ravel()
        elif target.ndim == 1:
            target_values = target
        else:
            raise ValueError(
                "Если target — numpy-array, он должен быть 1D или 2D с одним столбцом."
            )
    else:
        raise TypeError(
            "target должен быть np.ndarray, pd.Series или pd.DataFrame с одним столбцом."
        )

    # Проверяем, что длины совпадают
    if len(target_values) != len(df):
        raise ValueError("Длина target и количество строк df должны совпадать.")

    selected_cols = []

    r2 = pd.DataFrame({})

    for col in df.columns:
        col_values = df[col]
        R2 = calculate_r2(col_values, target_values)
        pd.concat([r2, pd.Series({col: R2})], axis=1)

    return r2


def filter_by_r2_threshold(
    df: pd.DataFrame, target: Union[np.ndarray, pd.Series, pd.DataFrame], r2_threshold: float
) -> pd.DataFrame:
    """Отфильтровать столбцы по пороговому значению R² относительно цели.

    Приводит целевой вектор к одномерному виду, вычисляет R² для каждого столбца
    и возвращает новый DataFrame, содержащий только удовлетворяющие порогу столбцы.

    Args:
        df (pd.DataFrame): Исходный DataFrame с признаками.
        target (np.ndarray | pd.Series | pd.DataFrame): Целевой вектор или столбец.
        r2_threshold (float): Минимальное допустимое значение R² (0 ≤ r2_threshold ≤ 1).

    Returns:
        pd.DataFrame: DataFrame, содержащий столбцы с R² выше или равным порогу.

    Raises:
        ValueError: Если длина цели отличается от числа строк или порог некорректен.
        TypeError: Если тип цели не поддерживается.

    Side Effects:
        None.

    Examples:
        >>> import pandas as pd, numpy as np
        >>> filter_by_r2_threshold(pd.DataFrame({'a':[1,2]}), np.array([1,2]), 0.5)
    """
    # Приводим target к 1D numpy-массиву
    if isinstance(target, pd.DataFrame):
        if target.shape[1] != 1:
            raise ValueError(
                "Если target — DataFrame, он должен содержать ровно один столбец."
            )
        target_values = target.iloc[:, 0].values
    elif isinstance(target, pd.Series):
        target_values = target.values
    elif isinstance(target, np.ndarray):
        if target.ndim == 2 and target.shape[1] == 1:
            target_values = target.ravel()
        elif target.ndim == 1:
            target_values = target
        else:
            raise ValueError(
                "Если target — numpy-array, он должен быть 1D или 2D с одним столбцом."
            )
    else:
        raise TypeError(
            "target должен быть np.ndarray, pd.Series или pd.DataFrame с одним столбцом."
        )

    # Проверяем, что длины совпадают
    if len(target_values) != len(df):
        raise ValueError("Длина target и количество строк df должны совпадать.")

    selected_cols = []

    for col in df.columns:
        col_values = df[col].values
        r2 = calculate_r2(col_values, target_values)
        if r2 >= r2_threshold:
            selected_cols.append(col)

    return df[selected_cols]


def unc_curve_fit(
    x: Any, y: Any, f: Callable[..., Any] = lambda x, k, b: k * x + b
) -> np.ndarray:
    """Perform regression and return coefficients with uncertainties.

    Accepts regular numeric arrays, uarrays, or pandas Series containing uncertain
    values, extracts nominal values and standard deviations as needed, and performs
    weighted curve fitting.

    Args:
        x (Any): Набор значений признаков, допускаются ufloat элементы.
        y (Any): Набор целевых значений, допускаются ufloat элементы.
        f (Callable[..., Any]): Модельная функция для аппроксимации, по умолчанию линейная.

    Returns:
        np.ndarray: Массив коэффициентов как uarray с неопределенностями.

    Raises:
        RuntimeError: Если оптимизация не сходится.
        TypeError: Если входы не могут быть приведены к числовому виду.

    Side Effects:
        None.

    Examples:
        >>> unc_curve_fit([0, 1], [0, 1])
    """
    # Convert to numpy arrays while preserving ufloat objects
    x_arr = np.asarray(x, dtype=object)
    y_arr = np.asarray(y, dtype=object)

    # Helper function to check for ufloat elements
    def has_ufloats(arr: np.ndarray) -> bool:
        """Check whether an array contains uncertainty objects.

        Args:
            arr (np.ndarray): Array to inspect.

        Returns:
            bool: True if any element exposes a `nominal_value` attribute.

        Raises:
            None.

        Side Effects:
            None.
        """
        return any(hasattr(el, "nominal_value") for el in arr.flat)

    # Process x: extract nominal values if ufloats present
    if has_ufloats(x_arr):
        x_clean = unp.nominal_values(x_arr)
    else:
        x_clean = np.array(x_arr, dtype=float)

    # Process y: extract both nominal and std_dev if ufloats present
    if has_ufloats(y_arr):
        y_nom = unp.nominal_values(y_arr)
        y_std = unp.std_devs(y_arr)
        sigma = y_std
    else:
        y_nom = np.array(y_arr, dtype=float)
        sigma = None

    # Perform curve fitting with weights if uncertainties present
    popt, pcov = curve_fit(f, x_clean, y_nom, sigma=sigma, absolute_sigma=True)

    # Create uarray for coefficients with uncertainties
    coefs = unp.uarray(popt, np.sqrt(np.diag(pcov)))

    return coefs
