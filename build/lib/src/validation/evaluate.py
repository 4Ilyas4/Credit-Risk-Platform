"""Оценка готовых предсказаний: качество, калибровка, стабильность, надёжность, ошибки."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.calibration.calibration_metrics import calibration_results
from src.validation.diagnostics.error_analysis import error_results
from src.validation.inputs import align_optional, prepare_inputs
from src.validation.performance.classification_metrics import probability_results, threshold_results
from src.validation.performance.ranking_metrics import ranking_results
from src.validation.reliability.confidence_intervals import ci_results
from src.validation.result import MetricResult, NOT_COMPUTABLE, WARNING, ValidationReport
from src.validation.stability import backtest_results, feature_psi_result, score_psi_result


def _split_reference(reference_data, y, s, dates, X):
    """Возвращает (s_ref, X_ref) из reference_data. Допустимо: DataFrame со столбцами y_score и/или признаками."""
    if reference_data is None:
        return None, None
    if not isinstance(reference_data, pd.DataFrame):
        raise TypeError("reference_data должен быть pandas.DataFrame")
    s_ref = None
    if "y_score" in reference_data.columns:
        s_ref = pd.to_numeric(reference_data["y_score"], errors="coerce").dropna().to_numpy(float)
    x_ref = reference_data.drop(columns=["y_score", "y_true"], errors="ignore")
    return s_ref, (x_ref if x_ref.shape[1] else None)


def evaluate_predictions(y_true, y_score, X: pd.DataFrame | None = None, dates=None, groups=None,
                         reference_data: pd.DataFrame | None = None, threshold: float | None = None,
                         n_boot: int | None = None, seed: int = 42,
                         score_kind: str = "probability", higher_is_riskier: bool = True) -> ValidationReport:
    """Оценивает готовые предсказания. Необязательные входы включают дополнительные проверки:

    X - анализ признаков в ошибках и PSI признаков (при reference_data); dates - backtesting по периодам;
    groups - групповой bootstrap и ошибки по сегментам; reference_data - PSI скора и признаков.
    Если в y_true один класс, ранжирующие метрики возвращают not_computable, а не число.

    score_kind: "probability" (скор в [0, 1]) или "score" (ранговый балл). Для рангового балла
    Brier, LogLoss и калибровка получают not_computable.
    higher_is_riskier=False: скор инвертируется, чтобы большее значение означало больший риск.
    """
    prep = prepare_inputs(y_true, y_score, score_kind=score_kind, higher_is_riskier=higher_is_riskier)
    y, s, mask = prep.y, prep.s, prep.mask
    report = ValidationReport(meta=dict(prep.info), artifacts={"y": y, "s": s})
    if prep.info["n_dropped_missing"]:
        report.warnings.append(f"[Входные данные] Исключено строк с пропуском в y_true или y_score: "
                               f"{prep.info['n_dropped_missing']}.")
    if X is not None and len(X) != len(mask):
        raise ValueError(f"Длина X ({len(X)}) не совпадает с y_true ({len(mask)})")
    Xv = None if X is None else X.reset_index(drop=True).loc[mask].reset_index(drop=True)
    dv = None if dates is None else align_optional(dates, mask, "dates")
    gv = None if groups is None else align_optional(groups, mask, "groups").to_numpy()

    report.add("Model Performance", ranking_results(y, s))
    report.add("Model Performance", probability_results(y, s, score_kind))
    report.add("Model Performance", threshold_results(y, s, threshold, score_kind))
    report.add("Calibration", calibration_results(y, s, score_kind=score_kind))
    report.add("Reliability", ci_results(y, s, n_boot=n_boot, seed=seed, groups=gv, score_kind=score_kind))

    skipped = []
    if dv is not None:
        report.add("Stability", backtest_results(y, s, dv))
    else:
        skipped.append("dates: backtesting по периодам")
    s_ref, x_ref = _split_reference(reference_data, y, s, dv, Xv)
    if s_ref is not None:
        report.add("Stability", score_psi_result(s_ref, s))
    elif reference_data is None:
        skipped.append("reference_data: PSI предсказаний и признаков")
    if x_ref is not None and Xv is not None:
        report.add("Stability", feature_psi_result(x_ref, Xv))
    elif reference_data is not None and Xv is None:
        skipped.append("X: PSI признаков")
    report.add("Error Analysis", error_results(y, s, threshold, dv, gv, Xv))
    if gv is None:
        skipped.append("groups: ошибки по сегментам, групповой bootstrap")
    if skipped:
        report.meta["skipped_checks"] = "Не выполнено из-за отсутствия входов: " + "; ".join(skipped) + "."
    return report
