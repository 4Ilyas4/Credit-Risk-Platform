"""Десять контролируемых сценариев A3. Для каждого задано ожидаемое поведение валидатора."""
import numpy as np
import pandas as pd
import pytest

from src.validation import assess_informativeness, evaluate_predictions, run_full_validation, validate_dataset
from src.validation.result import NOT_COMPUTABLE, OK, WARNING


def make(n=6000, rate=0.2, signal=0.25, seed=0, noise=0.12):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(float)
    s = np.clip(rate + signal * (y - rate) + rng.normal(0, noise, n), 0.001, 0.999)
    return y, s


def get(report, section, name):
    return report.get(section, name)


def test_1_well_separated_classes():
    y, s = make(signal=0.6, noise=0.1)
    r = evaluate_predictions(y, s, n_boot=50)
    assert get(r, "Model Performance", "ROC-AUC").value > 0.9
    assert get(r, "Model Performance", "KS").value > 0.6
    assert get(r, "Model Performance", "Gini").value == pytest.approx(2 * get(r, "Model Performance", "ROC-AUC").value - 1)


def test_2_random_predictions():
    rng = np.random.default_rng(1)
    y = (rng.random(8000) < 0.2).astype(float)
    r = evaluate_predictions(y, rng.random(8000), n_boot=50)
    auc = get(r, "Model Performance", "ROC-AUC")
    assert abs(auc.value - 0.5) < 0.03 and auc.status == WARNING
    assert get(r, "Model Performance", "Gini").status == WARNING


def test_3_bad_predictions_inverted_score():
    y, s = make(signal=0.5, noise=0.1)
    r = evaluate_predictions(y, 1 - s, n_boot=50)
    auc = get(r, "Model Performance", "ROC-AUC")
    assert auc.value < 0.1 and auc.status == WARNING
    assert "инвертирован" in auc.interpretation


def test_4_imbalanced_classes():
    y, s = make(n=20000, rate=0.01, signal=0.02, noise=0.03)  # умеренное разделение при 1% дефолтов
    data = pd.DataFrame({"target": y, "x": s})
    dq = validate_dataset(data, target="target")
    assert "Сильный дисбаланс" in dq.get("Data Quality", "Доля дефолтов").interpretation
    r = evaluate_predictions(y, s, n_boot=50)
    auc = get(r, "Model Performance", "ROC-AUC").value
    pr = get(r, "Model Performance", "PR-AUC")
    assert auc > 0.6  # AUC выглядит приемлемо
    assert pr.value < 0.15 and "доле дефолтов" in pr.interpretation  # а PR-AUC показывает реальную сложность


def test_5_high_calibration_error():
    y, s = make(signal=0.3)
    r = evaluate_predictions(y, np.clip(s * 0.4, 0.001, 0.999), n_boot=50)  # риск занижен в 2.5 раза
    assert get(r, "Calibration", "ECE").status == WARNING
    citl = get(r, "Calibration", "Calibration in the large")
    assert citl.status == WARNING and "занижает" in citl.interpretation
    assert get(r, "Model Performance", "ROC-AUC").value > 0.7  # ранжирование при этом не пострадало


def test_6_feature_distribution_shift():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.normal(0, 1, 5000), "y_score": rng.beta(2, 8, 5000)})
    y, s = make(n=5000)
    cur = pd.DataFrame({"a": rng.normal(1.5, 1, 5000), "b": rng.normal(0, 1, 5000)})
    r = evaluate_predictions(y, s, X=cur, reference_data=ref, n_boot=50)
    psi = get(r, "Stability", "PSI признаков")
    table = pd.DataFrame(psi.details["table"]).set_index("feature")
    assert table.loc["a", "psi"] > 0.25 and table.loc["a", "level"] == "significant"
    assert table.loc["b", "psi"] < 0.1
    assert psi.status == WARNING


def test_7_changed_predictive_relationship():
    rng = np.random.default_rng(3)
    n = 12000
    dates = pd.Series(pd.date_range("2013-01-01", periods=n, freq="3h"))
    half = np.arange(n) < n // 2
    x = rng.normal(size=n)
    logit = np.where(half, 1.5 * x, -0.2 * x) - 1.6  # связь признака с target пропала во второй половине
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    s = np.clip(1 / (1 + np.exp(-(1.5 * x - 1.6))), 0.001, 0.999)  # модель обучена на старой связи
    r = evaluate_predictions(y, s, dates=dates, n_boot=50)
    trend = get(r, "Stability", "Динамика ROC-AUC")
    assert trend.status == WARNING and trend.details["change"] < -0.1
    info = assess_informativeness(pd.DataFrame({"x": x}), y, dates=dates)
    iv = get(info, "Data Informativeness", "Стабильность IV по периодам")
    assert iv.status == WARNING


def test_8_missing_values_and_duplicates():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"id": np.arange(2000), "x": rng.normal(size=2000), "y": rng.integers(0, 2, 2000)})
    df.loc[:1200, "x"] = np.nan  # 60% пропусков
    df = pd.concat([df, df.iloc[:100]], ignore_index=True)  # 100 полных дубликатов
    rep = validate_dataset(df, target="y", id_col="id")
    assert rep.get("Data Quality", "Пропуски").status == WARNING
    assert rep.get("Data Quality", "Полные дубликаты строк").details["n_duplicates"] == 100
    assert rep.get("Data Quality", "Повторы идентификатора").status == WARNING
    assert any("Пропуски" in w for w in rep.warnings)


def test_9_one_class_missing_no_fake_values():
    s = np.linspace(0.05, 0.95, 500)
    for label in (0.0, 1.0):
        r = evaluate_predictions(np.full(500, label), s, n_boot=50)
        for name in ("ROC-AUC", "Gini", "KS", "PR-AUC"):
            m = get(r, "Model Performance", name)
            assert m.value is None and m.status == NOT_COMPUTABLE, name
        assert get(r, "Model Performance", "Brier Score").value is not None  # она определена и без второго класса
        assert get(r, "Reliability", "Доверительный интервал: ROC-AUC").status == NOT_COMPUTABLE
        assert any("не рассчитана" in w for w in r.warnings)
    assert assess_informativeness(pd.DataFrame({"x": s}), np.zeros(500)).sections["Data Informativeness"][0].status \
        == NOT_COMPUTABLE


def test_10_too_few_observations():
    y, s = make(n=60, rate=0.1)
    dq = validate_dataset(pd.DataFrame({"y": y, "x": s}), target="y")
    assert dq.get("Data Quality", "Достаточность наблюдений").status == WARNING
    assert dq.get("Data Quality", "Достаточность дефолтов").status == WARNING
    r = evaluate_predictions(y, s, n_boot=50)
    assert any("неустойчив" in w or "ненадёжн" in w for w in r.warnings)
