"""Mutual Information признаков с target. Используется scikit-learn (оценка по k ближайшим соседям)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.validation.result import MetricNotComputable


def mutual_information(X: pd.DataFrame, y: np.ndarray, max_rows: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """MI для числовых признаков. Пропуски заполняются медианой, на больших данных берётся выборка.

    Оценка приближённая и зависит от выборки и seed. Использовать для ранжирования признаков,
    а не для сравнения абсолютных значений.
    """
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        raise MetricNotComputable("В выборке один класс: Mutual Information не определена.")
    num = X.select_dtypes("number")
    num = num.loc[:, num.nunique(dropna=True) > 1]
    if num.shape[1] == 0:
        return pd.DataFrame(columns=["feature", "mutual_information"])
    idx = np.arange(len(num))
    if len(num) > max_rows:
        idx = np.random.default_rng(seed).choice(len(num), max_rows, replace=False)
    filled = num.iloc[idx].fillna(num.median())
    mi = mutual_info_classif(filled.to_numpy(float), y[idx], random_state=seed)
    return (pd.DataFrame({"feature": num.columns, "mutual_information": mi})
            .sort_values("mutual_information", ascending=False).reset_index(drop=True))
