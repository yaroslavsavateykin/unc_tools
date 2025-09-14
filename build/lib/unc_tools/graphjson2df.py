import json
import pandas as pd
import numpy as np
from uncertainties import ufloat, UFloat

class JSONFlattener:
    def __init__(self, level_names):
        """
        Инициализирует flattener с заданными именами уровней вложенности.

        Параметры:
        -----------
        level_names : list of str
            Список имён для каждого уровня вложенности до измерений.
            Например: ["channel", "molecule", "flow_rate"] или
            ["channel", "molecule", "pH", "flow_rate"] и т. д.
        """
        self.level_names = level_names.copy()

    @staticmethod
    def load_json_from_file(filepath: str) -> dict:
        """
        Загружает JSON из файла и возвращает его как dict.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def flatten(
        self,
        json_data: dict,
        measurement_fields: list = None,
        select_fields: list = None
    ) -> pd.DataFrame:
        """
        Сплющивает иерархический JSON в DataFrame, создавая по одной строке на каждое
        измерение из списков measurement_fields.

        Параметры:
        -----------
        json_data : dict
            Словарь, загруженный из JSON, где уровни вложенности задаются self.level_names.
        measurement_fields : list of str, optional
            Список ключей на самом нижнем уровне, которые соответствуют спискам значений.
            Если None, будут взяты все ключи в любом первом найденном "листовом" словаре.
        select_fields : list of str, optional
            Список измерений (из measurement_fields), которые нужно оставить. Остальные будут игнорироваться.
            Если None, берутся все поля из measurement_fields.

        Возвращает:
        ---------
        pd.DataFrame
            Таблица, где каждая строка = одно измерение (один элемент из списков measurement_fields).
            Колонки: 
              • self.level_names[0], self.level_names[1], …, self.level_names[-1]
              • measurement_fields (или select_fields)
        """
        # Найдём measurement_fields автоматически, если не указано
        if measurement_fields is None:
            # Делаем быструю переборную функцию, чтобы найти первый листовой словарь
            def find_leaf_fields(dct, depth):
                if depth == len(self.level_names):
                    return list(dct.keys())
                for val in dct.values():
                    if isinstance(val, dict):
                        result = find_leaf_fields(val, depth + 1)
                        if result:
                            return result
                return []

            measurement_fields = find_leaf_fields(json_data, 0)
            if not measurement_fields:
                raise ValueError("Не удалось найти ни одного набора measurement_fields в структуре JSON.")

        # Если select_fields указаны, оставляем только их
        if select_fields is not None:
            measurement_fields = [fld for fld in measurement_fields if fld in select_fields]

        records = []

        def recurse(curr_dict: dict, path: dict, depth: int):
            if depth < len(self.level_names):
                key_name = self.level_names[depth]
                for key, subtree in curr_dict.items():
                    # Если это последний уровень перед measurement_fields, пробуем конвертировать ключ в float
                    if depth == len(self.level_names) - 1:
                        try:
                            path_value = float(key)
                        except Exception:
                            path_value = key
                    else:
                        path_value = key

                    path[key_name] = path_value
                    recurse(subtree, path, depth + 1)

                # После обработки ветвей этого уровня удаляем ключ из path
                if key_name in path:
                    del path[key_name]

            else:
                # Глубина равна числу уровней — curr_dict содержит measurement_fields
                # Берём списки по каждому полю:
                lists = [curr_dict.get(fld, []) for fld in measurement_fields]
                if not lists:
                    return
                n = min(len(lst) for lst in lists)
                for i in range(n):
                    record = {**path}
                    for fld in measurement_fields:
                        record[fld] = curr_dict.get(fld, [None] * n)[i]
                    records.append(record)

        recurse(json_data, {}, 0)
        df = pd.DataFrame.from_records(records)

        # Сортировка по всем уровням вложенности, если они есть
        if self.level_names:
            df = df.sort_values(by=self.level_names).reset_index(drop=True)
        return df

    def flatten_with_uncertainty(
        self,
        json_data: dict,
        measurement_fields: list = None,
        select_fields: list = None
    ) -> pd.DataFrame:
        """
        Сплющивает иерархический JSON в DataFrame, агрегируя списки измерений
        в одну пару (mean ± std_dev) через библиотеку uncertainties. 
        На выходе — по одной строке на каждый "лист" JSON, где measurement_fields
        заменяются на UFloat(mean, std).

        Параметры:
        -----------
        json_data : dict
            Исходный вложенный словарь.
        measurement_fields : list of str, optional
            Список ключей на уровне измерений. Если None, пытаемся найти автоматически.
        select_fields : list of str, optional
            Поднабор measurement_fields, которые нужно учесть.

        Возвращает:
        ---------
        pd.DataFrame
            Таблица, где каждая строка = один «лист» JSON (не каждый элемент списка, а агрегированно).
            Колонки:
              • self.level_names[0], self.level_names[1], …, self.level_names[-1]
              • measurement_fields, где вместо списков — один UFloat(mean, std).
        """
        # Найдём measurement_fields, если не указано
        if measurement_fields is None:
            def find_leaf_fields(dct, depth):
                if depth == len(self.level_names):
                    return list(dct.keys())
                for val in dct.values():
                    if isinstance(val, dict):
                        result = find_leaf_fields(val, depth + 1)
                        if result:
                            return result
                return []

            measurement_fields = find_leaf_fields(json_data, 0)
            if not measurement_fields:
                raise ValueError("Не удалось найти ни одного набора measurement_fields в структуре JSON.")

        # Оставляем только select_fields, если они указаны
        if select_fields is not None:
            measurement_fields = [fld for fld in measurement_fields if fld in select_fields]

        records = []

        def recurse_agg(curr_dict: dict, path: dict, depth: int):
            if depth < len(self.level_names):
                key_name = self.level_names[depth]
                for key, subtree in curr_dict.items():
                    if depth == len(self.level_names) - 1:
                        try:
                            path_value = float(key)
                        except Exception:
                            path_value = key
                    else:
                        path_value = key

                    path[key_name] = path_value
                    recurse_agg(subtree, path, depth + 1)

                if key_name in path:
                    del path[key_name]

            else:
                # Находим измерения и агрегируем
                record = {**path}
                for fld in measurement_fields:
                    values = curr_dict.get(fld, [])
                    arr = np.array(values, dtype=float)
                    if arr.size == 0:
                        # Если список пуст, запишем None
                        record[fld] = None
                    else:
                        mean_val = float(np.mean(arr))
                        std_val = float(np.std(arr, ddof=0))
                        record[fld] = ufloat(mean_val, std_val)
                records.append(record)

        recurse_agg(json_data, {}, 0)
        df = pd.DataFrame.from_records(records)

        if self.level_names:
            df = df.sort_values(by=self.level_names).reset_index(drop=True)
        return df
    
    
    def widen(
        self,
        json_data: dict,
        unc: bool = False,
        skip_err: bool = False,
        measurement_fields: list = None,
        select_fields: list = None
    ) -> pd.DataFrame:
        """
        Преобразует иерархический JSON в широкий формат DataFrame, разворачивая измерения в отдельные столбцы.

        Параметры:
        -----------
        json_data : dict
            Исходный вложенный словарь JSON.
        unc : bool, optional
            Если True, добавляет столбцы STD и RSD для числовых данных.
        skip_err : bool, optional
            Если True, пропускает ошибки при вычислении STD/RSD для строковых полей.
        measurement_fields : list of str, optional
            Список ключей измерений. Если None, определяется автоматически.
        select_fields : list of str, optional
            Поднабор measurement_fields для обработки.

        Возвращает:
        -----------
        pd.DataFrame
            Широкая таблица с развернутыми измерениями.
        """
        # 1. «Сырые» данные в длинном формате
        df = self.flatten(
            json_data,
            measurement_fields=measurement_fields,
            select_fields=select_fields
        )

        # Определяем поля измерений (исключая уровни вложенности)
        measure_cols = [col for col in df.columns if col not in self.level_names]

        # 2. Приведём все измерения к числовому, где возможно, чтобы избежать строковых значений
        #    Если в колонке есть нечисловая строка, она станет NaN
        for col in measure_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. Группируем по уровням вложенности
        grouped = df.groupby(self.level_names, sort=False)

        # Определяем максимальное количество повторов (измерений) по любой группе
        max_n = grouped.size().max()

        new_rows = []
        for group_name, group_df in grouped:
            # group_name может быть кортежем, если несколько уровней вложенности
            if len(self.level_names) == 1:
                new_row = {self.level_names[0]: group_name}
            else:
                new_row = dict(zip(self.level_names, group_name))

            # Для каждой измерительной колонки собираем серию чисел (отфильтрованную от NaN)
            for col in measure_cols:
                series = pd.to_numeric(group_df[col], errors='coerce')
                values = series.dropna().tolist()
                n_vals = len(values)

                # Разворачиваем измерения: создаём колонки col_0 ... col_{max_n-1}
                for i in range(max_n):
                    if i < n_vals:
                        new_row[f"{col}_{i}"] = values[i]
                    else:
                        new_row[f"{col}_{i}"] = np.nan

                # Если нужна Uncertainty (STD и RSD)
                if unc:
                    if n_vals == 0:
                        std_val = np.nan
                        rsd_val = np.nan
                    else:
                        arr = np.array(values, dtype=float)
                        # Если в списке только один элемент, std = 0
                        std_val = float(np.std(arr, ddof=0)) if n_vals > 1 else 0.0
                        mean_val = float(np.mean(arr))
                        rsd_val = (std_val / mean_val * 100.0) if mean_val != 0 else 0.0

                    new_row[f"{col}_STD"] = std_val
                    new_row[f"{col}_RSD"] = rsd_val

            new_rows.append(new_row)

        # Собираем результирующий DataFrame
        result_df = pd.DataFrame(new_rows)

        # 4. Убедимся, что уровневые колонки имеют правильные типы (например, float для flow_rate)
        for lvl in self.level_names:
            # Попробуем превратить в числовой, но если не выйдет — оставляем как есть
            result_df[lvl] = pd.to_numeric(result_df[lvl], errors='ignore')

        return result_df

    def to_hierarchical_dict(self, df: pd.DataFrame, measurement_fields: list = None) -> dict:
        """
        Преобразует DataFrame обратно в вложенный словарь, используя self.level_names
        как уровни вложенности и measurement_fields как поля измерений.

        Параметры:
        -----------
        df : pd.DataFrame
            DataFrame «длинного» формата, в котором каждая строка – одно измерение,
            и есть колонки, соответствующие level_names + measurement_fields.
        measurement_fields : list of str, optional
            Список колонок-измерений. Если None, будут взяты все колонки, 
            кроме self.level_names.

        Возвращает:
        -----------
        dict
            Вложенный словарь, где ключи на первом уровне – уникальные значения df[level_names[0]],
            на втором – df[level_names[1]] и т. д.; а на самом глубоком уровне –
            словарь вида {fld: [...список всех значений fld для данного пути...], ...}.
        """
        # 1. Определяем measurement_fields, если не указано
        if measurement_fields is None:
            measurement_fields = [col for col in df.columns if col not in self.level_names]

        # 2. Инициализируем пустой словарь для результата
        hierarchical = {}

        # 3. Проходим по каждой строке df
        for _, row in df.iterrows():
            # «sub» будет ссылаться на уровень вложенности внутри hierarchical
            sub = hierarchical

            # Строим или спускаемся по всем уровням из level_names
            for lvl in self.level_names:
                key = row[lvl]
                # Преобразуем числовые ключи обратно в строку (если нужно)
                # Иначе остаётся оригинальный тип (float, int или один из ваших UFloat и т.д.)
                if not isinstance(key, str):
                    key = str(key)

                if key not in sub:
                    sub[key] = {}
                sub = sub[key]

            # Теперь «sub» – это словарь на глубине = len(level_names).
            # Если он ещё пуст, создаём в нём для каждого measurement_fields по пустому списку
            if not sub:
                for fld in measurement_fields:
                    sub[fld] = []

            # Добавляем значения измерений в соответствующие списки
            for fld in measurement_fields:
                val = row[fld]
                # Если val – NaN, пусть будет None
                if isinstance(val, float) and np.isnan(val):
                    sub[fld].append(None)
                else:
                    sub[fld].append(val)

        return hierarchical