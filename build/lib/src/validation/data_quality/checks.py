"""Проверки качества данных: структура, пропуски, дубликаты, выбросы."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.result import MetricResult, OK, WARNING

MISSING_WARN = 0.30
OUTLIER_WARN = 0.05
MIN_ROWS = 1000


def check_schema(df: pd.DataFrame, target: str | None, date_col: str | None, id_col: str | None,
                 required: list[str] | None) -> list[MetricResult]:
    wanted = [c for c in [target, date_col, id_col] + list(required or []) if c]
    missing = [c for c in wanted if c not in df.columns]
    lim = "Проверяется только наличие столбцов, а не смысл их значений."
    out = [MetricResult(
        "Структура данных", float(len(missing)), WARNING if missing else OK,
        "Число ожидаемых столбцов, которых нет в данных.",
        f"Отсутствуют столбцы: {missing}." if missing else
        f"Строк: {len(df)}, столбцов: {df.shape[1]}. Все ожидаемые столбцы на месте.", lim,
        {"missing_columns": missing, "n_rows": int(len(df)), "n_cols": int(df.shape[1])})]
    n = len(df)
    out.append(MetricResult(
        "Достаточность наблюдений", float(n), WARNING if n < MIN_ROWS else OK,
        "Число строк в данных.",
        f"Строк {n}" + (f" (меньше {MIN_ROWS}): статистические оценки будут неустойчивы." if n < MIN_ROWS
                       else ": объёма достаточно для базовых проверок."),
        f"Порог {MIN_ROWS} - рабочий ориентир, зависит от доли дефолтов и числа признаков."))
    return out


def check_missing(df: pd.DataFrame, columns: list[str]) -> MetricResult:
    rates = df[columns].isna().mean().sort_values(ascending=False)
    constant = [c for c in columns if df[c].nunique(dropna=True) <= 1]
    bad = rates[rates > MISSING_WARN]
    table = [{"column": str(c), "missing_rate": float(v)} for c, v in rates[rates > 0].head(30).items()]
    return MetricResult(
        "Пропуски", float(rates.max()) if len(rates) else 0.0, WARNING if len(bad) or constant else OK,
        "Максимальная доля пропусков среди признаков.",
        (f"Столбцов с долей пропусков выше {MISSING_WARN:.0%}: {len(bad)}. " if len(bad) else
         "Сильных пропусков нет. ") + (f"Константных столбцов: {len(constant)}." if constant else ""),
        "Пропуск может быть информативным (например, значение не заполняют при отказе), "
        "поэтому высокая доля пропусков сама по себе не означает ошибку.",
        {"table": table, "constant_columns": constant})


def check_duplicates(df: pd.DataFrame, id_col: str | None) -> list[MetricResult]:
    n = len(df)
    full = int(df.duplicated().sum()) if n else 0
    out = [MetricResult(
        "Полные дубликаты строк", full / n if n else 0.0, WARNING if full else OK,
        "Доля строк, полностью совпадающих с другой строкой.",
        f"Дубликатов: {full}." if full else "Полных дубликатов нет.",
        "Совпадение всех полей не всегда ошибка: два разных заёмщика могут иметь одинаковые значения.",
        {"n_duplicates": full})]
    if id_col and id_col in df.columns:
        dup = int(df[id_col].duplicated().sum())
        out.append(MetricResult(
            "Повторы идентификатора", dup / n if n else 0.0, WARNING if dup else OK,
            f"Доля строк с повторяющимся значением {id_col}.",
            f"Повторов идентификатора: {dup}." if dup else "Идентификаторы уникальны.",
            "Повторы допустимы, если одна запись описывает один объект в разные даты.", {"n_duplicates": dup}))
    return out


def check_outliers(df: pd.DataFrame, columns: list[str], k: float = 3.0) -> MetricResult:
    rows = []
    for c in columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        x = df[c].dropna().to_numpy(float)
        if len(x) < 20 or np.unique(x).size < 5:
            continue
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        rows.append({"column": c, "outlier_share": float(np.mean((x < q1 - k * iqr) | (x > q3 + k * iqr)))})
    rows.sort(key=lambda r: -r["outlier_share"])
    worst = rows[0]["outlier_share"] if rows else 0.0
    flagged = [r["column"] for r in rows if r["outlier_share"] > OUTLIER_WARN]
    return MetricResult(
        "Аномальные значения", worst, WARNING if flagged else OK,
        f"Максимальная доля значений за пределами Q1 - {k}*IQR и Q3 + {k}*IQR среди числовых признаков.",
        f"Признаков с долей выбросов выше {OUTLIER_WARN:.0%}: {len(flagged)}." if flagged else
        "Заметной доли выбросов нет.",
        "Правило IQR плохо работает для сильно скошенных распределений (доходы, суммы): "
        "там длинный хвост нормален. Значения не удаляются.", {"table": rows[:20]})
