"""Описание загруженной таблицы: типы столбцов, подсказка сопоставления, предупреждения о структуре."""
from __future__ import annotations

import re
import warnings

import pandas as pd

from src.api.schemas import ColumnInfo, SuggestedMapping

TARGET_NAMES = ("y_true", "target", "default", "label", "bad", "outcome", "actual", "flag_default")
SCORE_NAMES = ("y_score", "score", "prob", "proba", "probability", "pd", "y_pred", "predicted", "prediction")
DATE_NAMES = ("issue_d", "date", "dt", "period", "month", "origination", "observation")
GROUP_NAMES = ("group", "segment", "region", "model_version", "cohort", "grade", "product")


def looks_like_date(series: pd.Series) -> bool:
    """Текстовый столбец считается датой, если на образце разбирается не менее 90% значений."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = series.dropna().astype(str).head(500)
    if sample.empty or sample.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.5:
        return False  # числа не считаем датами: pandas разберёт 2015 как дату
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.9)


def describe_columns(df: pd.DataFrame) -> list[ColumnInfo]:
    out = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            kind = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            kind = "datetime"
        else:
            kind = "text"
        info = ColumnInfo(name=c, dtype=str(s.dtype), kind=kind, missing_share=float(s.isna().mean()),
                          n_unique=int(s.nunique(dropna=True)), looks_like_date=looks_like_date(s))
        if kind == "datetime" and s.notna().any():
            info.date_min, info.date_max = str(s.min()), str(s.max())
        out.append(info)
    return out


def suggest_mapping(columns: list[ColumnInfo]) -> SuggestedMapping:
    """Подсказка по именам и содержимому. Это предложение пользователю, а не решение."""
    by_name = {c.name.lower(): c for c in columns}

    def pick(names, ok=lambda c: True):
        for n in names:
            c = by_name.get(n)
            if c is not None and ok(c):
                return c.name
        return None

    y_true = pick(TARGET_NAMES, lambda c: c.kind in ("numeric", "boolean") and (c.n_unique or 0) <= 2)
    y_score = pick(SCORE_NAMES, lambda c: c.kind == "numeric")
    date = pick(DATE_NAMES, lambda c: c.looks_like_date or c.kind == "datetime")
    if date is None:
        date = next((c.name for c in columns if c.kind == "datetime"), None)
    group = pick(GROUP_NAMES, lambda c: c.kind == "text" and 2 <= (c.n_unique or 0) <= 1000)
    return SuggestedMapping(y_true=y_true, y_score=y_score, date=date, group=group)


def structure_warnings(df: pd.DataFrame, columns: list[ColumnInfo]) -> list[str]:
    out = []
    mangled = [c for c in df.columns if re.search(r"\.\d+$", c) and re.sub(r"\.\d+$", "", c) in df.columns]
    if mangled:
        out.append(f"В файле повторяются имена столбцов (pandas переименовал: {mangled[:5]}).")
    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed:
        out.append(f"Столбцы без имени: {len(unnamed)}. Возможно, в файле лишний индекс или пустой заголовок.")
    empty = [c.name for c in columns if c.missing_share >= 1.0]
    if empty:
        out.append(f"Полностью пустые столбцы: {empty[:5]}.")
    const = [c.name for c in columns if c.n_unique is not None and c.n_unique <= 1 and c.missing_share < 1.0]
    if const:
        out.append(f"Константные столбцы: {const[:5]}.")
    n_dup = int(df.duplicated().sum())
    if n_dup:
        out.append(f"Полных дубликатов строк: {n_dup}.")
    return out
