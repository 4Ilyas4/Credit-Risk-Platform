"""Проверка и подготовка входов: y_true (0/1) и y_score (вероятность в [0, 1])."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PreparedInputs:
    y: np.ndarray          # 0/1, float
    s: np.ndarray          # вероятности
    mask: np.ndarray       # булева маска исходной длины: какие строки использованы
    info: dict


def prepare_inputs(y_true, y_score) -> PreparedInputs:
    """Приводит входы к массивам. Строки с пропуском в y_true или y_score исключаются и считаются.

    Некорректные входы (разная длина, метки не 0/1, скор вне [0, 1], нет данных) - ValueError:
    молча исправлять их нельзя, иначе метрики будут посчитаны на чём-то другом.
    """
    yt = pd.to_numeric(pd.Series(y_true).reset_index(drop=True), errors="coerce")
    st = pd.to_numeric(pd.Series(y_score).reset_index(drop=True), errors="coerce")
    if len(yt) != len(st):
        raise ValueError(f"Разная длина: y_true={len(yt)}, y_score={len(st)}")
    mask = (np.isfinite(yt) & np.isfinite(st)).to_numpy()
    if not mask.any():
        raise ValueError("Нет ни одной строки с заданными y_true и y_score")
    y = yt.to_numpy(float)[mask]
    s = st.to_numpy(float)[mask]
    bad_labels = set(np.unique(y)) - {0.0, 1.0}
    if bad_labels:
        raise ValueError(f"y_true должен содержать только 0 и 1, найдено: {sorted(bad_labels)[:5]}")
    if s.min() < 0 or s.max() > 1:
        raise ValueError(f"y_score должен быть вероятностью в [0, 1], диапазон: [{s.min():.4g}, {s.max():.4g}]")
    info = {"n_input": int(len(yt)), "n_used": int(mask.sum()), "n_dropped_missing": int((~mask).sum()),
            "n_positive": int(y.sum()), "n_negative": int(len(y) - y.sum()),
            "default_rate": float(y.mean())}
    return PreparedInputs(y, s, mask, info)


def align_optional(values, mask: np.ndarray, name: str) -> pd.Series:
    """Выравнивает необязательный вход (даты, группы) с использованными строками."""
    series = pd.Series(values).reset_index(drop=True)
    if len(series) != len(mask):
        raise ValueError(f"Длина {name} ({len(series)}) не совпадает с y_true ({len(mask)})")
    return series[mask].reset_index(drop=True)
