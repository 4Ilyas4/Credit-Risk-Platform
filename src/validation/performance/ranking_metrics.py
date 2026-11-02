"""Ранжирующие метрики: ROC-AUC, Gini, KS, PR-AUC. Собственная реализация на numpy."""
from __future__ import annotations

import numpy as np

from src.validation.result import MetricNotComputable, MetricResult, OK, WARNING, safe_result


def require_both_classes(y: np.ndarray) -> tuple[int, int]:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        missing = "дефолтов" if n_pos == 0 else "недефолтов"
        raise MetricNotComputable(
            f"В выборке нет {missing} (n_pos={n_pos}, n_neg={n_neg}). Ранжирующая метрика не определена.")
    return n_pos, n_neg


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Ранги с усреднением для совпадающих значений (нумерация с 1)."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    counts = np.diff(np.r_[starts, len(xs)])
    ranks_sorted = np.repeat(starts + (counts + 1) / 2.0, counts)
    ranks = np.empty(len(x))
    ranks[order] = ranks_sorted
    return ranks


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Вероятность того, что случайный дефолт получил скор выше случайного недефолта (ничья = 0.5)."""
    n_pos, n_neg = require_both_classes(y)
    ranks = average_ranks(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def gini(y: np.ndarray, s: np.ndarray) -> float:
    return 2.0 * roc_auc(y, s) - 1.0


def _grouped_counts(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Число дефолтов и недефолтов для каждого уникального значения скора (по возрастанию)."""
    uniq, inv = np.unique(s, return_inverse=True)
    inv = inv.ravel()
    pos = np.bincount(inv, weights=y, minlength=len(uniq))
    total = np.bincount(inv, minlength=len(uniq))
    return pos, total - pos


def ks_statistic(y: np.ndarray, s: np.ndarray) -> float:
    """Максимальное расстояние между кумулятивными распределениями скора дефолтов и недефолтов."""
    n_pos, n_neg = require_both_classes(y)
    pos, neg = _grouped_counts(y, s)
    return float(np.max(np.abs(np.cumsum(pos) / n_pos - np.cumsum(neg) / n_neg)))


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision: площадь под кривой precision-recall (ступенчатая, как в scikit-learn)."""
    n_pos, _ = require_both_classes(y)
    pos, neg = _grouped_counts(y, s)
    tp = np.cumsum(pos[::-1])
    fp = np.cumsum(neg[::-1])
    precision = tp / (tp + fp)
    recall = tp / n_pos
    return float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))


def _interp_auc(v: float) -> tuple[str, str]:
    if v < 0.5:
        return (f"AUC={v:.3f} ниже 0.5: ранжирование хуже случайного. Проверьте направление скора "
                "(возможно, он инвертирован)."), WARNING
    if v < 0.55:
        return f"AUC={v:.3f}: ранжирование практически не отличается от случайного.", WARNING
    if v < 0.65:
        return f"AUC={v:.3f}: слабое ранжирование.", OK
    if v < 0.75:
        return f"AUC={v:.3f}: умеренное ранжирование, типично для кредитного скоринга.", OK
    if v < 0.85:
        return f"AUC={v:.3f}: хорошее ранжирование.", OK
    return (f"AUC={v:.3f}: очень высокое значение для кредитного скоринга. Проверьте признаки "
            "на утечку информации из будущего."), WARNING


def _interp_gini(v: float) -> tuple[str, str]:
    return f"Gini={v:.3f} эквивалентен AUC={(v + 1) / 2:.3f}.", (WARNING if v < 0.1 else OK)


def _interp_ks(v: float) -> tuple[str, str]:
    if v < 0.1:
        return f"KS={v:.3f}: скоры дефолтов и недефолтов почти не различаются.", WARNING
    return f"KS={v:.3f}: максимальное расхождение распределений скора двух классов.", OK


def _interp_pr(v: float, base: float) -> tuple[str, str]:
    lift = v / base if base > 0 else float("nan")
    return (f"PR-AUC={v:.3f} при доле дефолтов {base:.3f} (у случайной модели PR-AUC примерно равен "
            f"доле дефолтов). Выигрыш в {lift:.2f} раза."), (WARNING if lift < 1.1 else OK)


def ranking_results(y: np.ndarray, s: np.ndarray) -> list[MetricResult]:
    base = float(np.mean(y))
    rule = "Границы интерпретации - рабочие ориентиры для кредитного скоринга, а не строгие стандарты."
    return [
        safe_result("ROC-AUC", lambda: roc_auc(y, s),
                    "Вероятность того, что случайный дефолт получил скор выше случайного недефолта.",
                    _interp_auc, rule + " Не зависит от порога и не оценивает калибровку."),
        safe_result("Gini", lambda: gini(y, s), "Gini = 2*AUC - 1. Шкала от -1 до 1, 0 - случайное ранжирование.",
                    _interp_gini, "Нет собственной информации сверх AUC."),
        safe_result("KS", lambda: ks_statistic(y, s),
                    "Максимальная разница между накопленными долями дефолтов и недефолтов по скору.",
                    _interp_ks, "Отражает одну точку максимального расхождения, чувствительна к размеру выборки."),
        safe_result("PR-AUC", lambda: pr_auc(y, s),
                    "Площадь под кривой precision-recall. Полезна при редких дефолтах.",
                    lambda v: _interp_pr(v, base),
                    "Зависит от доли дефолтов, поэтому значения нельзя сравнивать между выборками с разной долей."),
    ]
