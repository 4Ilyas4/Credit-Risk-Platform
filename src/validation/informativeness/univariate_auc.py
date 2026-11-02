"""Univariate ROC-AUC признака и проверки потенциальной утечки."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.performance.ranking_metrics import roc_auc
from src.validation.result import MetricNotComputable

LEAK_AUC = 0.85  # одномерная сила признака, после которой нужна ручная проверка происхождения


def univariate_auc(x: pd.Series, y: np.ndarray, min_obs: int = 50) -> dict:
    """AUC одного числового признака. strength = max(AUC, 1-AUC): направление связи не важно."""
    xv = pd.to_numeric(x, errors="coerce").to_numpy(float)
    mask = np.isfinite(xv)
    yy = np.asarray(y, dtype=float)[mask]
    if mask.sum() < min_obs:
        raise MetricNotComputable(f"Слишком мало непустых значений ({int(mask.sum())}).")
    auc = roc_auc(yy, xv[mask])  # MetricNotComputable, если в непустой части один класс
    return {"auc": auc, "strength": max(auc, 1 - auc), "direction": "+" if auc >= 0.5 else "-",
            "n": int(mask.sum()), "coverage": float(mask.mean())}


def leakage_table(X: pd.DataFrame, y: np.ndarray, threshold: float = LEAK_AUC) -> pd.DataFrame:
    """Признаки с подозрительно сильной одномерной связью. Это сигнал, а не доказательство утечки."""
    rows = []
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            continue
        try:
            r = univariate_auc(X[col], y)
        except MetricNotComputable:
            continue
        rows.append({"feature": col, **r, "suspicious": r["strength"] >= threshold})
    cols = ["feature", "auc", "strength", "direction", "n", "coverage", "suspicious"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)
