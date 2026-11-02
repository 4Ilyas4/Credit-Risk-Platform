"""Качество модели по временным периодам: динамика AUC, Gini, KS, Brier."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.performance.classification_metrics import brier_score
from src.validation.performance.ranking_metrics import gini, ks_statistic, roc_auc
from src.validation.result import MetricNotComputable


def backtest_by_period(y: np.ndarray, s: np.ndarray, dates: pd.Series, freq: str = "Q",
                       min_per_class: int = 30) -> pd.DataFrame:
    """Метрики по периодам. Периоды, где дефолтов или недефолтов меньше min_per_class, помечаются
    как недостаточные: их AUC не считается, а не подставляется."""
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce")
    ok = d.notna().to_numpy()
    if not ok.any():
        raise MetricNotComputable("Ни одну дату не удалось разобрать: backtesting невозможен.")
    periods = d.dt.to_period(freq)
    rows = []
    for p in sorted(periods[ok].unique()):
        idx = np.flatnonzero(ok & (periods == p).to_numpy())
        yy, ss = y[idx], s[idx]
        n_pos, n_neg = int(yy.sum()), int((yy == 0).sum())
        row = {"period": str(p), "n": int(len(idx)), "n_positive": n_pos, "default_rate": float(yy.mean()),
               "mean_score": float(ss.mean()), "brier": brier_score(yy, ss),
               "auc": np.nan, "gini": np.nan, "ks": np.nan,
               "sufficient": bool(n_pos >= min_per_class and n_neg >= min_per_class)}
        if row["sufficient"]:
            row.update(auc=roc_auc(yy, ss), gini=gini(yy, ss), ks=ks_statistic(yy, ss))
        rows.append(row)
    return pd.DataFrame(rows)


def trend_summary(table: pd.DataFrame, column: str = "auc") -> dict:
    """Разброс метрики по достаточным периодам и изменение от первого к последнему."""
    vals = table.loc[table["sufficient"], column].dropna()
    if len(vals) < 2:
        raise MetricNotComputable(
            f"Нужно минимум 2 периода с достаточным числом наблюдений, есть {len(vals)}.")
    return {"first": float(vals.iloc[0]), "last": float(vals.iloc[-1]), "min": float(vals.min()),
            "max": float(vals.max()), "range": float(vals.max() - vals.min()),
            "change": float(vals.iloc[-1] - vals.iloc[0]), "n_periods": int(len(vals))}
