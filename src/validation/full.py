"""Полная проверка: объединяет все доступные проверки в один отчёт."""
from __future__ import annotations

import pandas as pd

from src.validation.data_quality import validate_dataset
from src.validation.evaluate import evaluate_predictions
from src.validation.explainability.shap_analysis import shap_results
from src.validation.informativeness import assess_informativeness
from src.validation.result import ValidationReport

SECTION_ORDER = ["Data Quality", "Data Informativeness", "Model Performance", "Calibration",
                 "Stability", "Reliability", "Error Analysis", "Explainability"]
_Y, _D = "__y__", "__date__"


def _feature_frame(X: pd.DataFrame, y_true, y_score, dates) -> pd.DataFrame:
    """Признаки + target (+ даты) для строк, где известны y_true и y_score."""
    y = pd.to_numeric(pd.Series(y_true).reset_index(drop=True), errors="coerce")
    s = pd.to_numeric(pd.Series(y_score).reset_index(drop=True), errors="coerce")
    if len(X) != len(y):
        raise ValueError(f"Длина X ({len(X)}) не совпадает с y_true ({len(y)})")
    keep = (y.notna() & s.notna()).to_numpy()
    frame = X.reset_index(drop=True).loc[keep].reset_index(drop=True)
    frame[_Y] = y[keep].to_numpy()
    if dates is not None:
        frame[_D] = pd.Series(dates).reset_index(drop=True).loc[keep].to_numpy()
    return frame


def run_full_validation(y_true, y_score, X: pd.DataFrame | None = None, dates=None, groups=None,
                        reference_data: pd.DataFrame | None = None, threshold: float | None = None,
                        n_boot: int | None = None, seed: int = 42, model=None,
                        score_kind: str = "probability", higher_is_riskier: bool = True) -> ValidationReport:
    """Объединяет все доступные проверки.

    Обязательны y_true и y_score. Без X не выполняются проверки данных и информативности признаков,
    без dates - backtesting, без reference_data - PSI, без groups - групповые проверки, без model - SHAP.
    Список пропущенных проверок лежит в report.meta["skipped_checks"].
    Базовую модель на данных пользователя пакет не обучает и не утверждает, что восстановил его модель.

    score_kind: "probability" или "score" (ранговый балл). Для рангового балла Brier, LogLoss
    и калибровка получают not_computable. higher_is_riskier=False инвертирует скор.
    """
    predictions = evaluate_predictions(y_true, y_score, X=X, dates=dates, groups=groups,
                                       reference_data=reference_data, threshold=threshold,
                                       n_boot=n_boot, seed=seed,
                                       score_kind=score_kind, higher_is_riskier=higher_is_riskier)
    skipped = [predictions.meta.pop("skipped_checks")] if "skipped_checks" in predictions.meta else []
    report = ValidationReport()
    if X is not None:
        frame = _feature_frame(X, y_true, y_score, dates)
        date_col = _D if dates is not None else None
        data = validate_dataset(frame, target=_Y, date_col=date_col, features=list(X.columns))
        info = assess_informativeness(frame[list(X.columns)], frame[_Y],
                                      dates=frame[_D] if date_col else None)
        for part in (data, info):
            skipped.append(part.meta.pop("skipped_checks", ""))
            report.merge(part)
    else:
        skipped.append("X: проверки качества данных и информативности признаков")
    report.merge(predictions)
    report.add("Explainability", shap_results(model, X))
    report.sections = {k: report.sections[k] for k in SECTION_ORDER if k in report.sections}
    clean = [s for s in skipped if s]
    if clean:
        report.meta["skipped_checks"] = " | ".join(clean)
    return report
