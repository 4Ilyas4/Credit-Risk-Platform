"""Проверки целевой переменной: бинарность, дисбаланс, достаточность дефолтов."""
from __future__ import annotations

import pandas as pd

from src.validation.result import MetricResult, NOT_COMPUTABLE, OK, WARNING

MIN_POSITIVES = 100


def check_target(y: pd.Series) -> list[MetricResult]:
    series = pd.Series(y)
    n_miss = int(series.isna().sum())
    vals = set(series.dropna().unique())
    if not vals <= {0, 1, 0.0, 1.0, True, False}:
        return [MetricResult("Целевая переменная", None, NOT_COMPUTABLE, "Проверка бинарности target.",
                             f"Target содержит значения кроме 0 и 1: {sorted(map(str, vals))[:5]}.",
                             "Метки нужно привести к 0 (нет дефолта) и 1 (дефолт).")]
    yy = series.dropna().astype(float)
    n_pos, n_neg = int(yy.sum()), int((yy == 0).sum())
    rate = n_pos / len(yy) if len(yy) else float("nan")
    one_class = n_pos == 0 or n_neg == 0
    if one_class:
        note = "В данных один класс: метрики качества не определены."
    elif rate < 0.05:
        note = "Сильный дисбаланс: ориентируйтесь на PR-AUC и калибровку, а не на accuracy."
    else:
        note = "Дисбаланс умеренный."
    return [
        MetricResult(
            "Доля дефолтов", rate, WARNING if one_class or rate < 0.01 or rate > 0.5 else OK,
            "Доля наблюдений с target = 1.",
            f"Дефолтов {n_pos}, недефолтов {n_neg}, доля {rate:.3%}. {note}",
            "Допустимая доля дефолтов зависит от продукта. Предупреждение при доле < 1% или > 50%.",
            {"n_positive": n_pos, "n_negative": n_neg, "n_missing_target": n_miss}),
        MetricResult(
            "Достаточность дефолтов", float(n_pos), WARNING if n_pos < MIN_POSITIVES else OK,
            "Число дефолтов в выборке: от него зависит ширина доверительных интервалов.",
            f"Дефолтов {n_pos}" + (f" (меньше {MIN_POSITIVES}): оценки неустойчивы." if n_pos < MIN_POSITIVES
                                   else ": объёма достаточно для оценки AUC с умеренной точностью."),
            f"Порог {MIN_POSITIVES} - рабочий ориентир."),
    ]
