"""Проверка отдельных столбцов запроса: бинарный исход, вероятность, даты, периоды."""
from __future__ import annotations

import pandas as pd

from src.api.errors import InvalidRequest
from src.api.schemas import Period


def need_column(df: pd.DataFrame, name: str, role: str) -> None:
    if name not in df.columns:
        raise InvalidRequest(f"Столбец {name!r} ({role}) не найден в наборе данных.")


def binary_target(s: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if not pd.api.types.is_numeric_dtype(s):
        raise InvalidRequest(f"Столбец {name!r} (y_true) должен быть числовым со значениями 0 и 1, "
                             f"а он имеет тип {s.dtype}.")
    bad = sorted(set(s.dropna().unique()) - {0, 1})
    if bad:
        raise InvalidRequest(f"Столбец {name!r} (y_true) должен содержать только 0 и 1, найдено: {bad[:5]}.")
    return s.astype(float)


def probability(s: pd.Series, name: str) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        raise InvalidRequest(f"Столбец {name!r} (y_score) должен быть числовым, а он имеет тип {s.dtype}.")
    vals = s.dropna()
    if len(vals) and (vals.min() < 0 or vals.max() > 1):
        raise InvalidRequest(f"Столбец {name!r} (y_score) должен быть вероятностью в [0, 1], "
                             f"диапазон: [{vals.min():.4g}, {vals.max():.4g}]. "
                             "Если это логит или балл, преобразуйте его в вероятность до загрузки.")
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
