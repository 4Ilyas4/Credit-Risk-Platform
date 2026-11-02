"""Information Value (IV) и Weight of Evidence. Собственная реализация."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.result import MetricNotComputable

SMOOTH = 0.5  # добавка к числу дефолтов и недефолтов в корзине (сглаживание нулей)


def make_bins(x: pd.Series, n_bins: int = 10) -> pd.Series:
    """Номера корзин: числовой признак - квантили, пропуск - отдельная корзина 'missing'.

    Категориальный признак - корзина = значение. Границы определяются по переданным данным,
    поэтому для честной оценки их нужно считать на обучающей части.
    """
    if pd.api.types.is_numeric_dtype(x):
        edges = np.unique(np.nanquantile(x.to_numpy(float), np.linspace(0, 1, n_bins + 1)[1:-1]))
        ids = pd.Series(np.searchsorted(edges, x.to_numpy(float), side="left"), index=x.index).astype(str)
        return ids.where(x.notna(), "missing")
    return x.astype("string").fillna("missing").astype(str)


def information_value(x: pd.Series, y: np.ndarray, n_bins: int = 10, smooth: float = SMOOTH) -> dict:
    """IV = sum (share_good - share_bad) * WoE, WoE = ln(share_good / share_bad).

    Знак: дефолт = 1 ("bad"), поэтому положительный WoE означает меньший риск, чем в среднем.
    """
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("Длины x и y не совпадают")
    n_bad, n_good = int(y.sum()), int((y == 0).sum())
    if n_bad == 0 or n_good == 0:
        raise MetricNotComputable(f"В выборке нет одного из классов (дефолтов {n_bad}, недефолтов {n_good}).")
    bins = make_bins(pd.Series(x).reset_index(drop=True), n_bins)
    df = pd.DataFrame({"bin": bins, "bad": y, "good": 1 - y}).groupby("bin").sum()
    df["n"] = df["bad"] + df["good"]
    k = len(df)
    share_bad = (df["bad"] + smooth) / (n_bad + smooth * k)
    share_good = (df["good"] + smooth) / (n_good + smooth * k)
    df["woe"] = np.log(share_good / share_bad)
    df["iv_part"] = (share_good - share_bad) * df["woe"]
    df["default_rate"] = df["bad"] / df["n"]
    return {"iv": float(df["iv_part"].sum()), "table": df.reset_index(), "n_bins": k}


def iv_level(iv: float) -> str:
    """Общепринятые ориентиры силы признака (кредитный скоринг)."""
    if iv < 0.02:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious"
