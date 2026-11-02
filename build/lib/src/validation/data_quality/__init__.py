"""Проверка качества данных."""
from __future__ import annotations

import pandas as pd

from src.validation.data_quality.checks import (
    check_duplicates, check_missing, check_outliers, check_schema)
from src.validation.data_quality.target_checks import check_target
from src.validation.data_quality.temporal import check_dates, check_maturity
from src.validation.result import ValidationReport


def validate_dataset(df: pd.DataFrame, target: str | None = None, date_col: str | None = None,
                     id_col: str | None = None, features: list[str] | None = None,
                     required_columns: list[str] | None = None) -> ValidationReport:
    """Проверяет структуру, полноту, дубликаты, выбросы, target и временную согласованность.

    Выполняются только те проверки, для которых переданы нужные столбцы: без target не проверяется
    дисбаланс, без date_col - зрелость и даты.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df должен быть pandas.DataFrame")
    report = ValidationReport(meta={"n_rows": int(len(df)), "n_cols": int(df.shape[1])})
    section = "Data Quality"
    report.add(section, check_schema(df, target, date_col, id_col, required_columns))
    if df.empty:
        report.warnings.append("[Data Quality] Датасет пуст, остальные проверки пропущены.")
        return report
    special = {c for c in (target, date_col, id_col) if c}
    feats = [c for c in (features if features is not None else df.columns)
             if c in df.columns and c not in special]
    if feats:
        report.add(section, check_missing(df, feats))
        report.add(section, check_outliers(df, feats))
    report.add(section, check_duplicates(df, id_col))
    has_target = bool(target) and target in df.columns
    has_dates = bool(date_col) and date_col in df.columns
    if has_target:
        report.add(section, check_target(df[target]))
    if has_dates:
        report.add(section, check_dates(df[date_col]))
        if has_target:
            report.add(section, check_maturity(df[date_col], df[target]))
    skipped = [n for n, ok in (("target", has_target), ("date_col", has_dates)) if not ok]
    if skipped:
        report.meta["skipped_checks"] = f"Не переданы: {', '.join(skipped)}. Связанные проверки не выполнялись."
    return report

