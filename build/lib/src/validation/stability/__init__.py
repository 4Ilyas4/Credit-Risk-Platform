"""Стабильность: PSI скора и признаков, backtesting по периодам."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.result import MetricNotComputable, MetricResult, NOT_COMPUTABLE, OK, WARNING
from src.validation.stability.drift_detection import drift_types_note, feature_psi_table
from src.validation.stability.psi import numeric_psi, psi_level
from src.validation.stability.temporal_backtesting import backtest_by_period, trend_summary

SECTION = "Stability"
PSI_LIMIT = ("Ориентиры PSI (0.1 / 0.25) — общепринятые в мониторинге кредитных моделей, но условные. "
             "PSI зависит от числа корзин и размера выборки; на малых выборках даёт шум.")


def score_psi_result(s_ref: np.ndarray, s_cur: np.ndarray, label: str = "") -> MetricResult:
    name = "PSI предсказаний" + (f" ({label})" if label else "")
    try:
        r = numeric_psi(s_ref, s_cur)
    except MetricNotComputable as exc:
        return MetricResult(name, None, NOT_COMPUTABLE, "Сдвиг распределения предсказанных вероятностей.", str(exc), PSI_LIMIT)
    lvl = psi_level(r["psi"])
    text = {"stable": "Распределение скора стабильно.", "moderate": "Умеренный сдвиг распределения скора.",
            "significant": "Значительный сдвиг распределения скора (prediction drift)."}[lvl]
    return MetricResult(name, r["psi"], OK if lvl == "stable" else WARNING,
                        "Сравнение распределения скора нового периода с референсным (корзины по квантилям референса).",
                        f"PSI={r['psi']:.3f}. {text}", PSI_LIMIT + " Не говорит, изменилось ли качество модели.",
                        {"table": r["table"], "n_ref": r["n_ref"], "n_cur": r["n_cur"]})


def feature_psi_result(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str] | None = None,
                       top_k: int = 20) -> MetricResult:
    table = feature_psi_table(ref, cur, features)
    if table.empty:
        return MetricResult("PSI признаков", None, NOT_COMPUTABLE, "Сдвиг распределений признаков.",
                            "Нет общих признаков в референсных и текущих данных.", PSI_LIMIT)
    valid = table[table["psi"].notna()]
    moved = valid[valid["psi"] >= 0.1]
    sig = valid[valid["psi"] >= 0.25]
    skipped = int(table["psi"].isna().sum())
    return MetricResult(
        "PSI признаков", float(valid["psi"].max()) if len(valid) else None, WARNING if len(moved) else OK,
        "Максимальный PSI среди признаков: сдвиг распределения признаков между референсом и текущими данными.",
        f"Признаков с PSI ≥ 0.1: {len(moved)} из {len(valid)}, из них ≥ 0.25: {len(sig)}. "
        + (f"Не рассчитано для {skipped}. " if skipped else "") + drift_types_note(),
        PSI_LIMIT + " Сдвиг признака (data drift) не обязательно ухудшает модель.",
        {"table": table.head(top_k)})


def backtest_results(y: np.ndarray, s: np.ndarray, dates: pd.Series, freq: str = "Q") -> list[MetricResult]:
    try:
        table = backtest_by_period(y, s, dates, freq)
    except MetricNotComputable as exc:
        return [MetricResult("Backtesting по периодам", None, NOT_COMPUTABLE, "Метрики по временным периодам.", str(exc), "")]
    lim = ("Периоды с малым числом дефолтов или недефолтов исключены из AUC/Gini/KS. Последние периоды могут быть "
           "незрелыми (дефолты ещё не наступили), тогда падение качества кажущееся. "
           "Пороги (разброс AUC > 0.05, падение > 0.03) — рабочие ориентиры из практики мониторинга моделей.")
    out = [MetricResult("Backtesting по периодам", None, OK,
                        "Метрики качества по календарным периодам.", f"Периодов: {len(table)}, достаточных: "
                        f"{int(table['sufficient'].sum())}.", lim, {"table": table})]
    try:
        t = trend_summary(table, "auc")
        warn = t["range"] > 0.05 or t["change"] < -0.03
        out.append(MetricResult(
            "Динамика ROC-AUC", t["range"], WARNING if warn else OK,
            "Разброс AUC между периодами (максимум минус минимум).",
            f"AUC от {t['min']:.3f} до {t['max']:.3f}, первый период {t['first']:.3f}, последний {t['last']:.3f} "
            f"(изменение {t['change']:+.3f}). " + ("Качество заметно колеблется." if warn else "Качество стабильно."),
            lim + " Пороги (разброс > 0.05, падение > 0.03) условные.", t))
    except MetricNotComputable as exc:
        out.append(MetricResult("Динамика ROC-AUC", None, NOT_COMPUTABLE, "Разброс AUC между периодами.", str(exc), lim))
    return out

