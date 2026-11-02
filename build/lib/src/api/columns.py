"""Проверка отдельных столбцов запроса: бинарный исход, вероятность, даты, периоды."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.api.errors import InvalidRequest
from src.api.schemas import Period


def need_column(df: pd.DataFrame, name: str, role: str) -> None:
    if name not in df.columns:
        raise InvalidRequest(f"Столбец {name!r} ({role}) не найден в наборе данных.")


def binary_target(s: pd.Series, name: str, default_value=None) -> pd.Series:
    """Приводит целевую переменную к 0/1.

    Если default_value задан, дефолтом считается ровно это значение (например, "Charged Off"),
    всё остальное — 0, пропуски остаются пропусками. Бинаризация только по явному указанию:
    без default_value столбец должен быть числовым 0/1.
    """
    if default_value is not None:
        out = (s == default_value).astype(float)
        out[s.isna()] = np.nan
        return out
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if not pd.api.types.is_numeric_dtype(s):
        values = sorted(map(str, s.dropna().unique()))[:5]
        raise InvalidRequest(f"Столбец {name!r} (y_true) должен быть числовым со значениями 0 и 1, "
                             f"а он имеет тип {s.dtype} и значения {values}. Если это столбец со статусом "
                             f"кредита (например, loan_status), укажите default_value — какое значение "
                             f"считать дефолтом (например, {next((v for v in values if 'default' in v.lower() or 'off' in v.lower()), values[0] if values else '')!r}).")
    bad = sorted(set(s.dropna().unique()) - {0, 1})
    if bad:
        raise InvalidRequest(f"Столбец {name!r} (y_true) должен содержать только 0 и 1, найдено: {bad[:5]}. "
                             f"Если это статусы кредита, укажите default_value — какое значение считать "
                             f"дефолтом (например, {bad[0]!r}).")
    return s.astype(float)


def probability(s: pd.Series, name: str, score_kind: str = "probability") -> pd.Series:
    """Проверяет столбец скора. Для score_kind='probability' требует диапазон [0, 1];
    для score_kind='score' допускает любой числовой диапазон (ранговый балл)."""
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        raise InvalidRequest(f"Столбец {name!r} (y_score) должен быть числовым, а он имеет тип {s.dtype}.")
    vals = s.dropna()
    if score_kind == "probability" and len(vals) and (vals.min() < 0 or vals.max() > 1):
        raise InvalidRequest(f"Столбец {name!r} (y_score) должен быть вероятностью в [0, 1], "
                             f"диапазон: [{vals.min():.4g}, {vals.max():.4g}]. Если это ранговый балл "
                             f"(например, ставка или grade), укажите score_kind='score'.")
    return s.astype(float)


def parse_dates(s: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        raise InvalidRequest(f"Столбец {name!r} (date) числовой. Укажите столбец с датами, например 2015-01-31.")
    parsed = pd.to_datetime(s, errors="coerce")
    share = float(parsed.notna().mean())
    if share < 0.9:
        raise InvalidRequest(f"В столбце {name!r} (date) удалось разобрать как дату только {share:.0%} значений.")
    return parsed


def period_mask(dates: pd.Series, p: Period) -> pd.Series:
    """Строки внутри периода, границы включительно. Строки без даты в период не попадают."""
    mask = dates.notna()
    if p.start is not None:
        mask &= dates >= pd.Timestamp(p.start)
    if p.end is not None:
        mask &= dates < pd.Timestamp(p.end) + pd.Timedelta(days=1)
    return mask
