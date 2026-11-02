"""Проверка и подготовка входов: y_true (0/1) и y_score (вероятность или ранговый балл)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SCORE_KINDS = ("probability", "score")


@dataclass
class PreparedInputs:
    y: np.ndarray          # 0/1, float
    s: np.ndarray          # скор: вероятность или ранговый балл
    mask: np.ndarray       # булева маска исходной длины: какие строки использованы
    info: dict
    score_kind: str = "probability"


def prepare_inputs(y_true, y_score, score_kind: str = "probability",
                   higher_is_riskier: bool = True) -> PreparedInputs:
    """Приводит входы к массивам. Строки с пропуском в y_true или y_score исключаются и считаются.

    score_kind:
      - "probability": скор должен лежать в [0, 1]; для Brier, LogLoss и калибровки
        вероятностная шкала обязательна.
      - "score": ранговый балл (например, int_rate или grade), любой числовой диапазон.
        Brier, LogLoss и калибровка для него не определены и получают not_computable.

    higher_is_riskier=False: скор инвертируется, чтобы большее значение всегда означало
    более высокий риск. Для вероятности это 1 - s, для рангового балла - -s.
    Молчаливой нормализации скора нет: если скор вне [0, 1] и score_kind="probability",
    это ошибка, а не повод делить на 100.

    Некорректные входы (разная длина, метки не 0/1, нет данных) - ValueError:
    молча исправлять их нельзя, иначе метрики будут посчитаны на чём-то другом.
    """
    if score_kind not in SCORE_KINDS:
        raise ValueError(f"score_kind должен быть одним из {SCORE_KINDS}, получено {score_kind!r}")
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
    if score_kind == "probability" and (s.min() < 0 or s.max() > 1):
        raise ValueError(f"y_score должен быть вероятностью в [0, 1] при score_kind='probability', "
                         f"диапазон: [{s.min():.4g}, {s.max():.4g}]. Если это ранговый балл "
                         f"(например, ставка или grade), укажите score_kind='score'.")
    if not higher_is_riskier:
        s = (1 - s) if score_kind == "probability" else -s
    info = {"n_input": int(len(yt)), "n_used": int(mask.sum()), "n_dropped_missing": int((~mask).sum()),
            "n_positive": int(y.sum()), "n_negative": int(len(y) - y.sum()),
            "default_rate": float(y.mean()),
            "score_kind": score_kind, "higher_is_riskier": bool(higher_is_riskier)}
    return PreparedInputs(y, s, mask, info, score_kind)


def align_optional(values, mask: np.ndarray, name: str) -> pd.Series:
    """Выравнивает необязательный вход (даты, группы) с использованными строками."""
    series = pd.Series(values).reset_index(drop=True)
    if len(series) != len(mask):
        raise ValueError(f"Длина {name} ({len(series)}) не совпадает с y_true ({len(mask)})")
    return series[mask].reset_index(drop=True)
