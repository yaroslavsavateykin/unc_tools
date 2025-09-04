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


def calculate_r2(x, y):
    """
    Calculate R-squared (coefficient of determination) for a linear fit y = k*x + b.

    Parameters:
    x, y : array-like or uncertainties.ufloat array
        Input data. Can be regular arrays or arrays with uncertainties.

    Returns:
    float
        R-squared value
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
    figsize: tuple = (10, 10),
    dpi: int = 100,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    annotate: bool = True,
    annot_fmt: str = ".2f",
    title: str = "Тепловая карта $R^2$ между столбцами",
    path: str = None,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    save_path = path
    """
    Построение тепловой карты значений R² с дополнительными настройками.
    
    Параметры:
    ----------
    r2_df : pd.DataFrame
        DataFrame с значениями R² (должен быть квадратной матрицей)
    figsize : tuple, optional
        Размер фигуры (по умолчанию (10, 10))
    dpi : int, optional
        Разрешение изображения (по умолчанию 100)
    cmap : str, optional
        Цветовая карта (по умолчанию 'viridis')
    vmin, vmax : float, optional
        Границы цветовой шкалы (по умолчанию 0.0 и 1.0)
    annotate : bool, optional
        Отображать ли значения в ячейках (по умолчанию True)
    annot_fmt : str, optional
        Формат аннотаций (по умолчанию ".2f")
    title : str, optional
        Заголовок графика
    save_path : str, optional
        Путь для сохранения изображения (если None - не сохранять)
    ax : plt.Axes, optional
        Ось для отрисовки (если None - создается новая фигура)
    **kwargs : 
        Дополнительные аргументы для plt.imshow()
    
    Возвращает:
    -----------
    plt.Axes
        Ось с построенной тепловой картой
    """
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
    r2_df: pd.DataFrame, path, title: str = "Тепловая карта R²"
) -> None:
    """
    Строит интерактивную теплокарту R² с помощью plotly.graph_objects.Heatmap.

    Параметры:
    -----------
    r2_df : pd.DataFrame
        Квадрат корреляционной матрицы (R²) — must be квадратный DataFrame.
    title : str
        Заголовок графика.
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
    """
    Возвращает новый DataFrame без столбцов, в которых все значения одинаковы.

    Параметры:
    df (pd.DataFrame): Входной DataFrame.

    Возвращает:
    pd.DataFrame: DataFrame без "постоянных" столбцов.
    """
    # Определяем столбцы, у которых количество уникальных значений равно 1
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]

    # Удаляем эти столбцы
    return df.drop(columns=constant_cols)


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет из DataFrame столбцы, которые полностью дублируют другие столбцы.

    Параметры:
    df (pd.DataFrame): Входной DataFrame.

    Возвращает:
    pd.DataFrame: Новый DataFrame без дублирующих столбцов.
    """
    # Используем .T.duplicated(), чтобы найти дублирующиеся строки в транспонированном DataFrame,
    # что соответствует дублирующим столбцам в исходном.
    mask = ~df.T.duplicated(keep="first")
    return df.loc[:, mask]


def batch_r2(
    df: pd.DataFrame,
    target,
) -> pd.DataFrame:
    """
    Параметры:
    -----------
    target : np.ndarray | pd.Series | pd.DataFrame
        Целевой вектор (или единственный столбец), по отношению к которому считается R².
        Может быть:
          • 1D numpy array (shape=(n,)) или 2D numpy array с одним столбцом (shape=(n,1))
          • pd.Series длины n
          • pd.DataFrame с одним столбцом и n строками
    df : pd.DataFrame
        DataFrame из n-строк и m-столбцов, столбцы которого будут отсортированы
        по величине R² относительно target.

    Возвращает:
    -----------
    pd.DataFrame
        Новый DataFrame, содержащий только те столбцы из df,
        у которых R²(target, столбец) >= r2_threshold.
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
    df: pd.DataFrame, target, r2_threshold: float
) -> pd.DataFrame:
    """
    Оставляет в df только те столбцы, у которых R² с target >= r2_threshold.

    Параметры:
    -----------
    target : np.ndarray | pd.Series | pd.DataFrame
        Целевой вектор (или единственный столбец), по отношению к которому считается R².
        Может быть:
          • 1D numpy array (shape=(n,)) или 2D numpy array с одним столбцом (shape=(n,1))
          • pd.Series длины n
          • pd.DataFrame с одним столбцом и n строками
    df : pd.DataFrame
        DataFrame из n-строк и m-столбцов, столбцы которого будут отсортированы
        по величине R² относительно target.
    r2_threshold : float
        Пороговое значение R² (0 ≤ r2_threshold ≤ 1). Столбцы с R² < r2_threshold будут отброшены.

    Возвращает:
    -----------
    pd.DataFrame
        Новый DataFrame, содержащий только те столбцы из df,
        у которых R²(target, столбец) >= r2_threshold.
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


def unc_curve_fit(x, y, f=lambda x, k, b: k * x + b):
    """
    Perform linear regression and return coefficients with uncertainties.

    Handles:
    - Regular arrays (floats/int)
    - uarrays from uncertainties package
    - Pandas Series containing floats or ufloat elements

    Parameters:
    x, y : array-like inputs (can be mix of regular/uarray)
    f : model function (default linear)

    Returns:
    uarray of coefficients with uncertainties
    """
    # Convert to numpy arrays while preserving ufloat objects
    x_arr = np.asarray(x, dtype=object)
    y_arr = np.asarray(y, dtype=object)

    # Helper function to check for ufloat elements
    def has_ufloats(arr):
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
