"""Корреляции и мультиколлинеарность числовых признаков."""
from __future__ import annotations

import numpy as np
import pandas as pd


def high_correlations(X: pd.DataFrame, threshold: float = 0.9, max_rows: int = 200_000,
                      seed: int = 42) -> pd.DataFrame:
    """Пары числовых признаков с |корреляция Спирмена| >= threshold. Большие данные берутся выборкой."""
    num = X.select_dtypes("number")
    num = num.loc[:, num.nunique(dropna=True) > 1]
    if num.shape[1] < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "spearman"])
    if len(num) > max_rows:
        num = num.sample(max_rows, random_state=seed)
    corr = num.corr(method="spearman")
    rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iat[i, j]
            if np.isfinite(v) and abs(v) >= threshold:
                rows.append({"feature_a": cols[i], "feature_b": cols[j], "spearman": float(v)})
    out = pd.DataFrame(rows, columns=["feature_a", "feature_b", "spearman"])
    return out.reindex(out["spearman"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def variance_inflation(X: pd.DataFrame, max_rows: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """VIF по диагонали обратной корреляционной матрицы (Пирсон, пропуски исключаются построчно).

    Если матрица вырождена (признаки линейно зависимы), VIF не определён и возвращается inf.
    """
    num = X.select_dtypes("number")
    num = num.loc[:, num.nunique(dropna=True) > 1].dropna()
    if num.shape[1] < 2 or len(num) < num.shape[1] + 10:
        return pd.DataFrame(columns=["feature", "vif"])
    if len(num) > max_rows:
        num = num.sample(max_rows, random_state=seed)
    corr = np.corrcoef(num.to_numpy(float), rowvar=False)
    if np.linalg.matrix_rank(corr) < corr.shape[0]:
        vif = np.full(corr.shape[0], np.inf)
    else:
        vif = np.diag(np.linalg.inv(corr))
    return pd.DataFrame({"feature": num.columns, "vif": vif}).sort_values("vif", ascending=False).reset_index(drop=True)
