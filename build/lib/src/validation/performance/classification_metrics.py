"""Метрики по вероятностям и по порогу: Brier, Log Loss, Precision, Recall, F1, Balanced Accuracy."""
from __future__ import annotations

import numpy as np

from src.validation.performance.ranking_metrics import require_both_classes
from src.validation.result import MetricNotComputable, MetricResult, NOT_COMPUTABLE, OK, WARNING, safe_result


def brier_score(y: np.ndarray, s: np.ndarray) -> float:
    return float(np.mean((s - y) ** 2))


def log_loss(y: np.ndarray, s: np.ndarray, eps: float | None = None) -> float:
    """Скор клипуется в [eps, 1-eps], eps = машинный эпсилон float64 (как в scikit-learn 1.8)."""
    eps = np.finfo(np.float64).eps if eps is None else eps
    p = np.clip(s, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def confusion_counts(y: np.ndarray, s: np.ndarray, threshold: float) -> dict[str, int]:
    """Предсказание = 1, если скор >= порога."""
    pred = s >= threshold
    return {"tp": int(np.sum(pred & (y == 1))), "fp": int(np.sum(pred & (y == 0))),
            "fn": int(np.sum(~pred & (y == 1))), "tn": int(np.sum(~pred & (y == 0)))}


def _ratio(num: float, den: float, what: str) -> float:
    if den == 0:
        raise MetricNotComputable(f"{what}: знаменатель равен нулю, метрика не определена.")
    return num / den


def precision(c: dict) -> float:
    return _ratio(c["tp"], c["tp"] + c["fp"], "Precision (нет ни одного предсказанного дефолта)")


def recall(c: dict) -> float:
    return _ratio(c["tp"], c["tp"] + c["fn"], "Recall (в выборке нет дефолтов)")


def f1(c: dict) -> float:
    p, r = precision(c), recall(c)
    return _ratio(2 * p * r, p + r, "F1 (Precision и Recall равны нулю)")


def balanced_accuracy(c: dict) -> float:
    tpr = _ratio(c["tp"], c["tp"] + c["fn"], "Balanced Accuracy (нет дефолтов)")
    tnr = _ratio(c["tn"], c["tn"] + c["fp"], "Balanced Accuracy (нет недефолтов)")
    return (tpr + tnr) / 2


def probability_results(y: np.ndarray, s: np.ndarray, score_kind: str = "probability") -> list[MetricResult]:
    """Brier и Log Loss. Для рангового скора (score_kind='score') не определены:
    они требуют вероятностной шкалы, а не просто ранжирования."""
    if score_kind != "probability":
        note = ("Скор — ранговый балл, а не вероятность: его значения нельзя интерпретировать "
                "как вероятность дефолта, поэтому квадратичная и логарифмическая ошибки не определены. "
                "Для вероятностной интерпретации преобразуйте балл в вероятность (например, калибровкой "
                "на исторических данных) или укажите score_kind='probability'.")
        lim = "Ранговые метрики (AUC, Gini, KS) при этом считаются: они зависят только от порядка."
        return [MetricResult("Brier Score", None, NOT_COMPUTABLE,
                             "Средний квадрат разницы между предсказанной вероятностью и исходом.",
                             note, lim),
                MetricResult("Log Loss", None, NOT_COMPUTABLE,
                             "Среднее отрицательное логарифмическое правдоподобие.",
                             note, lim)]
    base = float(np.mean(y))
    prior_brier = base * (1 - base)
    prior_ll = log_loss(y, np.full(len(y), base))
    return [
        safe_result(
            "Brier Score", lambda: brier_score(y, s),
            "Средний квадрат разницы между предсказанной вероятностью и исходом. Меньше - лучше.",
            lambda v: (f"Brier={v:.4f}. Константный прогноз (доля дефолтов {base:.3f}) даёт {prior_brier:.4f}. "
                       + ("Модель не лучше константы." if v >= prior_brier else
                          f"Улучшение относительно константы: {(1 - v / prior_brier) * 100:.1f}%."),
                       WARNING if v >= prior_brier else OK),
            "Смешивает качество ранжирования и калибровку. Зависит от доли дефолтов."),
        safe_result(
            "Log Loss", lambda: log_loss(y, s),
            "Среднее отрицательное логарифмическое правдоподобие. Сильно штрафует уверенные ошибки.",
            lambda v: (f"LogLoss={v:.4f}. Константный прогноз даёт {prior_ll:.4f}.",
                       WARNING if v >= prior_ll else OK),
            "Скор клипуется в [eps, 1-eps], eps - машинный эпсилон (2.2e-16): скор ровно 0 или 1 с неверным "
            "исходом даёт большой штраф. Чувствителен к редким уверенным ошибкам."),
    ]


def threshold_results(y: np.ndarray, s: np.ndarray, threshold: float | None = None,
                      score_kind: str = "probability") -> list[MetricResult]:
    """Метрики по порогу. По умолчанию порог = доля дефолтов в выборке (скор выше среднего риска).

    Для рангового скора (score_kind='score') порог по умолчанию — квантиль скора,
    соответствующий доле дефолтов: в «дефолт» попадает верхняя доля, равная доле дефолтов.
    """
    require = True
    try:
        require_both_classes(y)
    except MetricNotComputable as exc:
        require = False
        msg = str(exc)
    if threshold is None:
        if score_kind == "probability":
            used = float(np.mean(y))
            src = "по умолчанию равен доле дефолтов в выборке"
        else:
            used = float(np.quantile(s, 1 - np.mean(y)))
            src = "по умолчанию равен квантилю скора, соответствующему доле дефолтов"
    else:
        used = float(threshold)
        src = "задан пользователем"
    limits = (f"Порог {used:.3f} ({src}). Результат сильно зависит от порога, "
              "а бизнес-порог решения здесь неизвестен.")
    explain_suffix = f" Порог {used:.3f}."
    if not require:
        names = ["Precision", "Recall", "F1", "Balanced Accuracy"]
        return [MetricResult(n, None, "not_computable", n + explain_suffix, msg, limits) for n in names]
    c = confusion_counts(y, s, used)
    det = {"threshold": used, **c}
    out = [
        safe_result("Precision", lambda: precision(c), "Доля настоящих дефолтов среди предсказанных." + explain_suffix,
                    lambda v: (f"Precision={v:.3f}", OK), limits, det),
        safe_result("Recall", lambda: recall(c), "Доля найденных дефолтов среди всех настоящих." + explain_suffix,
                    lambda v: (f"Recall={v:.3f}", OK), limits, det),
        safe_result("F1", lambda: f1(c), "Гармоническое среднее Precision и Recall." + explain_suffix,
                    lambda v: (f"F1={v:.3f}", OK), limits, det),
        safe_result("Balanced Accuracy", lambda: balanced_accuracy(c),
                    "Среднее доли найденных дефолтов и доли найденных недефолтов." + explain_suffix,
                    lambda v: (f"Balanced Accuracy={v:.3f}; 0.5 соответствует случайному прогнозу.",
                               WARNING if v <= 0.52 else OK), limits, det),
        MetricResult("Confusion Matrix", None, OK,
                     "Число TP, FP, FN, TN при выбранном пороге." + explain_suffix,
                     f"TP={c['tp']}, FP={c['fp']}, FN={c['fn']}, TN={c['tn']}.", limits, det),
    ]
    return out
