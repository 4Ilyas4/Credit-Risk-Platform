"""Тесты пакета валидации на синтетических данных с известными свойствами.

Каждый генератор из synthetic_data закладывает ровно один дефект. Тест проверяет,
что дефект даёт ожидаемый статус (warning или not_computable), а на чистом наборе
(good_model) ложных тревог нет. Это главный аргумент на собеседовании: инструмент
доказуемо находит заложенные дефекты.
"""
import pandas as pd
import pytest

from src.validation import assess_informativeness, evaluate_predictions, run_full_validation, validate_dataset
from src.validation.result import NOT_COMPUTABLE, OK, WARNING
from synthetic_data import (good_model, weak_model, drift, leakage, immature_target,
                            one_class, few_defaults, prediction_drift)


def get(report, section, name):
    return report.get(section, name)


def split_by_date(df, share=0.5):
    """Делит набор на референсную (раннюю) и текущую (позднюю) части по дате."""
    dates = pd.to_datetime(df["date"])
    cut = dates.quantile(share)
    return df[dates <= cut], df[dates > cut]


def reference_frame(ref, features=()):
    """DataFrame для reference_data: y_score + признаки."""
    cols = {"y_score": ref["score"].to_numpy()}
    cols.update({f: ref[f].to_numpy() for f in features})
    return pd.DataFrame(cols)


# ---------- чистая модель: ложных тревог нет ----------

def test_good_model_high_auc_and_no_warnings():
    df = good_model()
    rep = evaluate_predictions(df["default"], df["score"], n_boot=50)
    auc = get(rep, "Model Performance", "ROC-AUC")
    assert auc.status == OK and auc.value > 0.7
    # на чистом наборе ни одна метрика качества не должна давать warning
    for m in rep.sections["Model Performance"]:
        assert m.status != WARNING, m.name
    for m in rep.sections["Calibration"]:
        assert m.status != WARNING, m.name
    # доверительные интервалы не должны быть широкими
    for m in rep.sections["Reliability"]:
        assert m.status != WARNING, m.name


def test_good_model_stable_and_mature():
    df = good_model()
    rep = evaluate_predictions(df["default"], df["score"], dates=df["date"], n_boot=50)
    trend = get(rep, "Stability", "Динамика ROC-AUC")
    assert trend.status == OK, trend.interpretation
    dq = validate_dataset(df, target="default", date_col="date")
    maturity = get(dq, "Data Quality", "Зрелость target")
    assert maturity.status == OK, maturity.interpretation


def test_good_model_full_run_no_fatal_warnings():
    df = good_model()
    rep = run_full_validation(df["default"], df["score"], X=df[["income"]],
                              dates=df["date"], n_boot=50)
    # ключевые проверки посчитаны, а не упали
    assert get(rep, "Model Performance", "ROC-AUC").value > 0.7
    assert get(rep, "Data Quality", "Зрелость target").status == OK


# ---------- слабая модель ----------

def test_weak_model_low_auc_but_computable():
    df = weak_model()
    auc = get(evaluate_predictions(df["default"], df["score"], n_boot=50),
              "Model Performance", "ROC-AUC")
    assert auc.value is not None and auc.value < 0.65
    assert auc.status == OK  # слабая, но корректная: не сломана


# ---------- дрейф ----------

def test_drift_detected_by_psi():
    df = drift()
    ref, cur = split_by_date(df)
    rep = evaluate_predictions(cur["default"], cur["score"], X=cur[["income"]],
                               dates=cur["date"],
                               reference_data=reference_frame(ref, ["income"]), n_boot=50)
    psi_score = get(rep, "Stability", "PSI предсказаний")
    assert psi_score.status == WARNING and psi_score.value >= 0.1, psi_score.interpretation
    psi_feat = get(rep, "Stability", "PSI признаков")
    assert psi_feat.status == WARNING, psi_feat.interpretation
    table = pd.DataFrame(psi_feat.details["table"])
    assert table.loc[table["feature"] == "income", "psi"].iloc[0] >= 0.1


def test_prediction_drift_psi_but_quality_ok():
    df = prediction_drift()
    ref, cur = split_by_date(df)
    rep = evaluate_predictions(cur["default"], cur["score"], dates=cur["date"],
                               reference_data=reference_frame(ref), n_boot=50)
    psi = get(rep, "Stability", "PSI предсказаний")
    assert psi.status == WARNING and psi.value >= 0.1, psi.interpretation
    auc = get(rep, "Model Performance", "ROC-AUC")
    assert auc.value > 0.7  # дрейф предсказаний, а не качества


# ---------- утечка ----------

def test_leakage_flagged():
    df = leakage()
    info = assess_informativeness(df[["leaky", "income"]], df["default"])
    leak = get(info, "Data Informativeness", "Проверка на утечку")
    assert leak.status == WARNING
    assert "leaky" in leak.interpretation
    # обычный признак не должен быть подозрительным
    flagged = leak.details.get("flagged", [])
    assert "leaky" in flagged
    assert "income" not in flagged


# ---------- недозревший target ----------

def test_immature_target_detected():
    df = immature_target()
    dq = validate_dataset(df, target="default", date_col="date")
    maturity = get(dq, "Data Quality", "Зрелость target")
    assert maturity.status == WARNING, maturity.interpretation
    assert maturity.value is not None and maturity.value < 0.6


# ---------- один класс ----------

def test_one_class_not_computable():
    df = one_class()
    rep = evaluate_predictions(df["default"], df["score"], n_boot=50)
    for name in ("ROC-AUC", "Gini", "KS", "PR-AUC"):
        m = get(rep, "Model Performance", name)
        assert m.status == NOT_COMPUTABLE and m.value is None, name
    assert get(rep, "Reliability", "Доверительный интервал: ROC-AUC").status == NOT_COMPUTABLE
    # Brier определена и без второго класса: не подставляем фейковые значения
    assert get(rep, "Model Performance", "Brier Score").value is not None


# ---------- малое число дефолтов ----------

def test_few_defaults_warning():
    df = few_defaults()
    dq = validate_dataset(df, target="default")
    assert get(dq, "Data Quality", "Достаточность дефолтов").status == WARNING
    rep = evaluate_predictions(df["default"], df["score"], n_boot=50)
    ci = get(rep, "Reliability", "Доверительный интервал: ROC-AUC")
    assert ci.status == WARNING  # интервал широкий из-за малого числа дефолтов
    assert any("меньше 100" in w for w in rep.warnings)


# ---------- все наборы: прогон без падений ----------

@pytest.mark.parametrize("builder", [good_model, weak_model, drift, leakage,
                                     immature_target, one_class, few_defaults, prediction_drift])
def test_all_generators_full_run_no_crash(builder):
    df = builder()
    kwargs = {}
    if "date" in df.columns:
        kwargs["dates"] = df["date"]
    rep = run_full_validation(df["default"], df["score"], **kwargs, n_boot=50)
    assert rep is not None
    assert isinstance(rep.sections, dict) and len(rep.sections) > 0
