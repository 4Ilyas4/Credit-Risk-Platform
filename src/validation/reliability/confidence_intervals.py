"""Доверительные интервалы метрик качества в формате результата отчёта."""
from __future__ import annotations

import numpy as np

from src.validation.performance.classification_metrics import brier_score
from src.validation.performance.ranking_metrics import ks_statistic, pr_auc, roc_auc
from src.validation.reliability.bootstrap import auto_n_boot, bootstrap_metrics, minimum_positives_warning
from src.validation.result import MetricNotComputable, MetricResult, NOT_COMPUTABLE, OK, WARNING

# Gini не пересчитывается в каждой выборке: Gini = 2*AUC - 1, интервал получается линейным преобразованием.
BASE_METRICS = {"ROC-AUC": roc_auc, "KS": ks_statistic, "PR-AUC": pr_auc, "Brier Score": brier_score}
NAMES = ["ROC-AUC", "Gini", "KS", "PR-AUC", "Brier Score"]


def _gini_from_auc(r: dict) -> dict:
    out = dict(r)
    out.update(value=2 * r["value"] - 1, ci_low=2 * r["ci_low"] - 1, ci_high=2 * r["ci_high"] - 1,
               std_error=2 * r["std_error"])
    return out


def ci_results(y: np.ndarray, s: np.ndarray, n_boot: int | None = None, alpha: float = 0.05, seed: int = 42,
               groups: np.ndarray | None = None) -> list[MetricResult]:
    """Bootstrap-интервалы. Без groups - стратифицированный, с groups - по целым группам.

    n_boot=None: 500 повторов при ≤ 50 000 строк, иначе 200 (скорость).
    """
    n_boot = auto_n_boot(len(y), n_boot)
    few = minimum_positives_warning(int(y.sum()))
    raw = bootstrap_metrics(y, s, BASE_METRICS, n_boot, alpha, seed, groups)
    raw["Gini"] = _gini_from_auc(raw["ROC-AUC"]) if isinstance(raw.get("ROC-AUC"), dict) else raw["ROC-AUC"]
    method_note = ("Стратифицированный bootstrap предполагает независимость наблюдений. Если кредиты одного "
                   "заёмщика или одного периода зависимы, интервал получится слишком узким. "
                   if groups is None else
                   "Групповой bootstrap учитывает зависимость внутри группы, но требует достаточно групп. ")
    reduced = " (на большой выборке число повторов уменьшено ради скорости)" if n_boot < 500 else ""
    lim = (method_note + f"Интервал перцентильный, {int((1 - alpha) * 100)}%, {n_boot} повторов{reduced}. "
           "Показывает неопределённость из-за конечной выборки, но не из-за смены периода или популяции.")
    out = []
    for name in NAMES:
        r = raw[name]
        if isinstance(r, MetricNotComputable):
            out.append(MetricResult(f"Доверительный интервал: {name}", None, NOT_COMPUTABLE,
                                    "Bootstrap-интервал метрики.", str(r), lim))
            continue
        width = r["ci_high"] - r["ci_low"]
        wide = width > 0.1 and name != "Brier Score"
        text = (f"{name}={r['value']:.4f}, {int((1 - alpha) * 100)}% интервал [{r['ci_low']:.4f}, "
                f"{r['ci_high']:.4f}], ширина {width:.4f}, метод: {r['method']}.")
        if wide:
            text += " Интервал широкий: различия в качестве меньше этой ширины нельзя считать значимыми."
        if few:
            text += " " + few
        out.append(MetricResult(f"Доверительный интервал: {name}", r["value"], WARNING if wide or few else OK,
                                "Bootstrap-интервал метрики: диапазон значений при повторной выборке данных.",
                                text, lim, r))
    return out
