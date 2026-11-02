"""Очистка и определение целевой переменной для Lending Club."""
from __future__ import annotations

import pandas as pd

# Итоговые статусы. Current, Late, Grace Period, In Grace Period - исход ещё не известен.
GOOD_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}
BAD_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}


def build_target(df: pd.DataFrame, status_col: str = "loan_status") -> pd.DataFrame:
    """Добавляет target: 1 - дефолт, 0 - погашен. Незрелые наблюдения исключаются.

    Ограничение: используется итоговый статус, а не фиксированный 12-месячный горизонт.
    """
    if status_col not in df.columns:
        raise KeyError(f"Нет столбца {status_col}")
    status = df[status_col].astype("string").str.strip()
    target = pd.Series(pd.NA, index=df.index, dtype="Int8")
    target[status.isin(GOOD_STATUSES)] = 0
    target[status.isin(BAD_STATUSES)] = 1
    mask = target.notna()
    # Одна копия отфильтрованных строк вместо трёх (assign -> фильтр -> assign): важно по памяти.
    out = df.loc[mask].copy()
    out["target"] = target[mask].astype("int8")
    return out


def parse_issue_date(df: pd.DataFrame, col: str = "issue_d") -> pd.DataFrame:
    """Приводит дату выдачи (например, 'Dec-2015') к datetime. Некорректные значения -> NaT."""
    parsed = pd.to_datetime(df[col], format="%b-%Y", errors="coerce")
    return df.assign(**{col: parsed})


def drop_invalid_rows(df: pd.DataFrame, date_col: str = "issue_d") -> pd.DataFrame:
    """Удаляет полные дубликаты и строки без даты выдачи."""
    out = df.drop_duplicates()
    if date_col in out.columns:
        out = out[out[date_col].notna()]
    return out.reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Полный шаг очистки: target -> дата -> дубликаты."""
    return drop_invalid_rows(parse_issue_date(build_target(df)))


def target_summary(df: pd.DataFrame, date_col: str = "issue_d") -> pd.DataFrame:
    """Доля дефолтов по годам выдачи: нужна, чтобы увидеть дисбаланс и смещение зрелости."""
    year = df[date_col].dt.year.rename("year")
    return df.groupby(year)["target"].agg(n="size", default_rate="mean")
