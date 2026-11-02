"""Тесты сводки отчёта (Этап A, Шаг 3).

Проверяют:
- summarize возвращает три карточки (качество, стабильность, надёжность) со статусом = худший внутри;
- списки «внимание» и «не проверено»;
- schema_version в JSON-отчёте;
- поле pillar в MetricResult.
"""
import numpy as np
import pytest

from src.validation import evaluate_predictions, run_full_validation
from src.validation.reporting.report import summarize
from src.validation.result import NOT_COMPUTABLE, OK, PILLAR_QUALITY, PILLAR_RELIABILITY, PILLAR_STABILITY, WARNING
from synthetic_data import (drift, few_defaults, good_model, immature_target, one_class, prediction_drift)


def cards(report):
    return summarize(report)["cards"]


def test_pillar_assigned_by_section():
    df = good_model(n=2000)
    rep = evaluate_predictions(df["default"], df["score"], n_boot=50)
    for m in rep.sections["Model Performance"]:
        assert m.pillar == PILLAR_QUALITY
    for m in rep.sections["Calibration"]:
        assert m.pillar == PILLAR_QUALITY
    for m in rep.sections["Reliability"]:
        assert m.pillar == PILLAR_RELIABILITY
    # pillar попадает в JSON
    d = rep.to_dict()
    assert d["schema_version"] == "1.0"
    assert all(r["pillar"] == PILLAR_QUALITY for r in d["sections"]["Model Performance"])


def test_good_model_without_dates_cards():
    df = good_model(n=2000)
    rep = evaluate_predictions(df["default"], df["score"], n_boot=50)
    c = cards(rep)
    assert c["quality"]["status"] == OK
    assert c["reliability"]["status"] == OK
    # стабильность не проверена: нет дат и референса
    assert c["stability"]["status"] == NOT_COMPUTABLE
    assert c["stability"]["n_metrics"] == 0
    # на чистом наборе нет предупреждений
    s = summarize(rep)
    assert s["attention"] == []
    # но есть пропущенные проверки
    assert any("Пропущено" in n for n in s["not_checked"])


def test_good_model_with_dates_all_ok():
    df = good_model()
    rep = evaluate_predictions(df["default"], df["score"], dates=df["date"], n_boot=50)
    c = cards(rep)
    assert c["quality"]["status"] == OK
    assert c["stability"]["status"] == OK
    assert c["reliability"]["status"] == OK


def test_one_class_cards_not_computable():
    df = one_class()
    rep = evaluate_predictions(df["default"], df["score"], n_boot=50)
    c = cards(rep)
    assert c["quality"]["status"] == NOT_COMPUTABLE
    assert c["reliability"]["status"] == NOT_COMPUTABLE
    assert c["quality"]["n_not_computable"] > 0
    s = summarize(rep)
    assert any("ROC-AUC" in n for n in s["not_checked"])


def test_few_defaults_reliability_warning():
    df = few_defaults()
    rep = evaluate_predictions(df["default"], df["score"], n_boot=50)
    c = cards(rep)
    assert c["reliability"]["status"] == WARNING
    assert c["reliability"]["n_warning"] > 0
    s = summarize(rep)
    assert any("интервал" in n for n in s["attention"])


def test_drift_stability_warning():
    df = drift()
    from test_validation_synthetic import reference_frame, split_by_date
    ref, cur = split_by_date(df)
    rep = evaluate_predictions(cur["default"], cur["score"], X=cur[["income"]],
                               dates=cur["date"],
                               reference_data=reference_frame(ref, ["income"]), n_boot=50)
    c = cards(rep)
    assert c["stability"]["status"] == WARNING
    assert c["stability"]["n_warning"] >= 1


def test_immature_target_quality_warning_via_maturity():
    df = immature_target()
    df["income"] = np.random.default_rng(0).normal(50, 10, len(df))
    rep = run_full_validation(df["default"], df["score"], X=df[["income"]],
                              dates=df["date"], n_boot=50)
    c = cards(rep)
    # незрелость target — предупреждение в Data Quality (качество данных)
    assert c["quality"]["status"] == WARNING
    assert any("Зрелость target" in n for n in summarize(rep)["attention"])


def test_summarize_empty_report():
    from src.validation.result import ValidationReport
    s = summarize(ValidationReport())
    for pillar in (PILLAR_QUALITY, PILLAR_STABILITY, PILLAR_RELIABILITY):
        assert s["cards"][pillar]["status"] == NOT_COMPUTABLE
        assert s["cards"][pillar]["n_metrics"] == 0
    assert s["attention"] == [] and s["not_checked"] == []