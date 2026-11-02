"""Таблицы IV по признакам и по периодам."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.informativeness.information_value import information_value, iv_level
from src.validation.result import MetricNotComputable


def iv_table(X: pd.DataFrame, y: np.ndarray, n_bins: int) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        try:
            iv = information_value(X[col], y, n_bins)["iv"]
        except MetricNotComputable:
            continue
        rows.append({"feature": col, "iv": iv, "level": iv_level(iv)})
    if not rows:
        return pd.DataFrame(columns=["feature", "iv", "level"])
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def iv_by_period(X: pd.DataFrame, y: np.ndarray, dates: pd.Series, features: list[str],
                 n_bins: int, min_per_class: int = 50) -> pd.DataFrame:
    """IV по календарным годам. Годы, где дефолтов или недефолтов меньше min_per_class, пропускаются."""
    years = pd.to_datetime(dates, errors="coerce").dt.year.to_numpy(float)
    rows = []
    for year in np.unique(years[np.isfinite(years)]):
        pos = np.flatnonzero(years == year)
        yy = y[pos]
        if yy.sum() < min_per_class or (yy == 0).sum() < min_per_class:
            continue
        for col in features:
            try:
                rows.append({"period": int(year), "feature": col,
                             "iv": information_value(X[col].iloc[pos], yy, n_bins)["iv"]})
            except MetricNotComputable:
                continue
    return pd.DataFrame(rows, columns=["period", "feature", "iv"])
