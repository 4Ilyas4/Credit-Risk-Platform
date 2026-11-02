"""Тесты корректности логики (Этап A, Шаг 2).

Проверяют:
- score_kind="score": Brier, LogLoss и калибровка получают not_computable, ранжирующие метрики считаются;
- higher_is_riskier=False: «перевёрнутый» скор даёт верные статусы (AUC > 0.5);
- бинаризация цели только по явному default_value;
- референс по умолчанию «ранние 70% / поздние 30% по дате».
"""
import numpy as np
import pandas as pd
import pytest

from src.validation import evaluate_predictions
from src.validation.result import NOT_COMPUTABLE, OK, WARNING
from api_helpers import BASE_MAPPING, csv_bytes, make_frame, start_run, upload
from synthetic_data import good_model


def get(report, section, name):
    return report.get(section, name)


# ---------- ранговый скор (score_kind="score") ----------

def test_rank_score_probability_metrics_not_computable():
    df = good_model(n=2000)
    rank = np.clip(df["score"] * 100, 1, 99)  # ранговый балл 1..99, выше = рискованнее
    rep = evaluate_predictions(df["default"], rank, score_kind="score", n_boot=50)
    # вероятностные метрики не определены для рангового балла
    assert get(rep, "Model Performance", "Brier Score").status == NOT_COMPUTABLE
    assert get(rep, "Model Performance", "Log Loss").status == NOT_COMPUTABLE
    assert get(rep, "Calibration", "Calibration in the large").status == NOT_COMPUTABLE
    assert get(rep, "Calibration", "ECE").status == NOT_COMPUTABLE
    assert get(rep, "Calibration", "Calibration Curve").status == NOT_COMPUTABLE
    # ранжирующие метрики считаются
    auc = get(rep, "Model Performance", "ROC-AUC")
    assert auc.value > 0.7 and auc.status == OK
    assert get(rep, "Reliability", "Доверительный интервал: ROC-AUC").value is not None
    # Brier CI не включается для рангового балла
    names = [m.name for m in rep.sections["Reliability"]]
    assert "Доверительный интервал: Brier Score" not in names
    # в метаданных отмечен тип скора
    assert rep.meta["score_kind"] == "score"


def test_rank_score_threshold_uses_score_quantile():
    df = good_model(n=2000)
    rank = np.clip(df["score"] * 100, 1, 99)
    rep = evaluate_predictions(df["default"], rank, score_kind="score", n_boot=50)
    thr = get(rep, "Model Performance", "Precision")
    # порог по умолчанию — квантиль скора, соответствующий доле дефолтов (а не доля дефолтов)
    expected = float(np.quantile(rank, 1 - df["default"].mean()))
    assert thr.details["threshold"] == pytest.approx(expected, abs=1e-9)
    assert thr.details["threshold"] != pytest.approx(df["default"].mean())  # не доля дефолтов
    assert "квантилю" in thr.limitations


def test_probability_score_out_of_range_raises():
    df = good_model(n=500)
    with pytest.raises(ValueError, match="score_kind='probability'"):
        evaluate_predictions(df["default"], df["score"] * 100, n_boot=50)


# ---------- перевёрнутый скор (higher_is_riskier=False) ----------

def test_inverted_score_higher_is_riskier_false():
    df = good_model(n=2000)
    inv = 1 - df["score"]  # больше = меньше риск
    rep = evaluate_predictions(df["default"], inv, higher_is_riskier=False, n_boot=50)
    auc = get(rep, "Model Performance", "ROC-AUC")
    assert auc.value > 0.7 and auc.status == OK
    # калибровка после инверсии тоже корректна (1 - s вернуло исходную вероятность)
    assert get(rep, "Calibration", "Calibration in the large").status == OK
    assert rep.meta["higher_is_riskier"] is False


def test_inverted_rank_score_higher_is_riskier_false():
    df = good_model(n=2000)
    rank = np.clip((1 - df["score"]) * 100, 1, 99)  # выше = меньше риск
    rep = evaluate_predictions(df["default"], rank, score_kind="score", higher_is_riskier=False, n_boot=50)
    auc = get(rep, "Model Performance", "ROC-AUC")
    assert auc.value > 0.7 and auc.status == OK


# ---------- бинаризация цели по default_value ----------

def test_binary_target_with_default_value():
    from src.api.columns import binary_target
    s = pd.Series(["Fully Paid", "Charged Off", "Current", "Charged Off", None])
    out = binary_target(s, "loan_status", default_value="Charged Off")
    assert out.fillna(-1).tolist() == [0.0, 1.0, 0.0, 1.0, -1.0]
    assert out.isna().sum() == 1  # пропуск остался пропуском


def test_binary_target_without_default_value_rejects_text():
    from src.api.errors import InvalidRequest
    from src.api.columns import binary_target
    s = pd.Series(["Fully Paid", "Charged Off"])
    with pytest.raises(InvalidRequest, match="default_value"):
        binary_target(s, "loan_status")


# ---------- референс по умолчанию: ранние 70% / поздние 30% ----------

def test_default_reference_early_70(client, dataset):
    m = {**BASE_MAPPING, "date": "issue_d"}
    run = start_run(client, dataset, m).json()
    done = client.get(f"/validation-runs/{run['run_id']}").json()
    assert done["status"] == "completed", done["error"]
    # проверяются поздние 30% строк (3000 * 0.3 = 900)
    assert done["summary"]["n_used"] == 900
    # PSI посчитан по референсу по умолчанию
    names = {mm["name"] for mm in done["result"]["sections"].get("Stability", [])}
    assert "PSI предсказаний" in names
    # в отчёте есть пояснение про референс по умолчанию
    notes = done["result"]["meta"].get("notes", [])
    assert any("ранние 70%" in n for n in notes)


def test_explicit_reference_disables_default(client, dataset):
    # если референс задан явно, дефолтный не применяется
    m = {**BASE_MAPPING, "date": "issue_d"}
    run = start_run(client, dataset, m, reference={"period": {"end": "2014-06-30"}}).json()
    done = client.get(f"/validation-runs/{run['run_id']}").json()
    assert done["status"] == "completed", done["error"]
    notes = done["result"]["meta"].get("notes", [])
    assert not any("ранние 70%" in n for n in notes)


# ---------- score_kind через API ----------

def test_api_rejects_out_of_range_probability(client):
    df = make_frame()
    df["score"] = df["score"] * 100  # вне [0,1]
    ds = upload(client, csv_bytes(df)).json()
    r = start_run(client, ds)
    assert r.status_code == 422 and "score_kind" in r.json()["error"]


def test_api_accepts_rank_score_with_score_kind(client):
    df = make_frame()
    df["score"] = df["score"] * 100
    ds = upload(client, csv_bytes(df)).json()
    m = {**BASE_MAPPING, "score_kind": "score"}
    run = start_run(client, ds, m).json()
    done = client.get(f"/validation-runs/{run['run_id']}").json()
    assert done["status"] == "completed", done["error"]
    brier = next(mm for mm in done["result"]["sections"]["Model Performance"] if mm["name"] == "Brier Score")
    assert brier["status"] == "not_computable"
    auc = next(mm for mm in done["result"]["sections"]["Model Performance"] if mm["name"] == "ROC-AUC")
    assert auc["value"] is not None


def test_api_binarizes_target_with_default_value(client):
    df = make_frame(n=500)
    df["status"] = np.where(df["default"] == 1, "Charged Off", "Fully Paid")
    ds = upload(client, csv_bytes(df)).json()
    m = {"y_true": "status", "y_score": "score", "default_value": "Charged Off"}
    run = start_run(client, ds, m).json()
    done = client.get(f"/validation-runs/{run['run_id']}").json()
    assert done["status"] == "completed", done["error"]
    assert done["summary"]["default_rate"] == pytest.approx(0.2, abs=0.03)


# ---------- исправление: инверсия скора применяется и к референсу ----------

def test_inverted_score_psi_consistent_with_normal(client, dataset):
    """higher_is_riskier=False не должен давать ложный PSI-дрейф: инверсия применяется
    и к референсу, и к текущим данным, поэтому PSI совпадает с обычным режимом."""
    m = {**BASE_MAPPING, "date": "issue_d"}
    ref_spec = {"period": {"end": "2014-06-30"}}

    def psi(higher_is_riskier):
        run = start_run(client, dataset, {**m, "higher_is_riskier": higher_is_riskier},
                        reference=ref_spec).json()
        done = client.get(f"/validation-runs/{run['run_id']}").json()
        assert done["status"] == "completed", done["error"]
        return next(mm for mm in done["result"]["sections"]["Stability"]
                    if mm["name"] == "PSI предсказаний")["value"]

    # PSI инвариантен к согласованной монотонной инверсии обеих частей
    assert psi(False) == pytest.approx(psi(True), abs=1e-6)


def test_inverted_score_auc_high_after_inversion(client):
    """Подаём скор, где большее значение = меньше риск, с higher_is_riskier=False:
    инструмент инвертирует его обратно, и AUC должен быть высоким."""
    df = make_frame()
    df["score"] = 1 - df["score"]  # теперь высокий скор = низкий риск
    ds = upload(client, csv_bytes(df)).json()
    m = {**BASE_MAPPING, "higher_is_riskier": False}
    run = start_run(client, ds, m).json()
    done = client.get(f"/validation-runs/{run['run_id']}").json()
    assert done["status"] == "completed", done["error"]
    auc = next(mm for mm in done["result"]["sections"]["Model Performance"]
               if mm["name"] == "ROC-AUC")
    assert auc["value"] > 0.7  # инверсия вернула исходное ранжирование


def test_reference_dataset_rank_score(client):
    """Отдельный датасет-референс с ранговым скором: score_kind передаётся в probability()."""
    df = make_frame(500)
    df["score"] = df["score"] * 100  # ранговый балл вне [0,1]
    ds = upload(client, csv_bytes(df)).json()
    ref = upload(client, csv_bytes(make_frame(300).assign(score=lambda d: d.score * 100))).json()
    m = {**BASE_MAPPING, "score_kind": "score"}
    run = start_run(client, ds, m, reference={"dataset_id": ref["dataset_id"]}).json()
    done = client.get(f"/validation-runs/{run['run_id']}").json()
    assert done["status"] == "completed", done["error"]
    names = {mm["name"] for mm in done["result"]["sections"].get("Stability", [])}
    assert "PSI предсказаний" in names