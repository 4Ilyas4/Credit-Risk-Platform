import json

import numpy as np
import pandas as pd
import pytest

from src.validation import (assess_informativeness, evaluate_predictions, run_full_validation,
                            validate_dataset)
from src.validation.inputs import prepare_inputs
from src.validation.result import NOT_COMPUTABLE, MetricResult, ValidationReport, to_jsonable


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(float)
    s = np.clip(0.2 + 0.25 * (y - 0.2) + rng.normal(0, 0.12, n), 0.001, 0.999)
    X = pd.DataFrame({"a": rng.normal(size=n) + y, "b": rng.normal(size=n), "c": rng.choice(["u", "v"], n)})
    dates = pd.Series(pd.date_range("2014-01-01", periods=n, freq="6h"))
    return y, s, X, dates


def test_prepare_inputs_rejects_bad_inputs():
    with pytest.raises(ValueError, match="Разная длина"):
        prepare_inputs([0, 1], [0.1])
    with pytest.raises(ValueError, match="только 0 и 1"):
        prepare_inputs([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        prepare_inputs([0, 1], [0.1, 1.5])
    with pytest.raises(ValueError, match="Нет ни одной строки"):
        prepare_inputs([np.nan], [np.nan])


def test_x_length_mismatch_is_clear_value_error():
    y, s, X, _ = _data()
    with pytest.raises(ValueError, match="Длина X"):
        evaluate_predictions(y, s, X=X.iloc[:10], n_boot=30)
    with pytest.raises(ValueError, match="dates"):
        evaluate_predictions(y, s, dates=pd.Series(pd.date_range("2014-01-01", periods=10)), n_boot=30)


def test_prepare_inputs_drops_missing_and_counts_them():
    p = prepare_inputs([0, 1, np.nan, 1], [0.1, 0.9, 0.5, np.nan])
    assert p.info["n_used"] == 2 and p.info["n_dropped_missing"] == 2


def test_evaluate_predictions_works_with_only_required_inputs():
    y, s, *_ = _data()
    r = evaluate_predictions(y, s, n_boot=30)
    assert "Stability" not in r.sections  # без dates и reference_data стабильность не считается
    assert "skipped_checks" in r.meta and "dates" in r.meta["skipped_checks"]


def test_optional_inputs_enable_extra_checks():
    y, s, X, dates = _data()
    groups = np.random.default_rng(1).integers(0, 30, len(y))
    r = evaluate_predictions(y, s, X=X, dates=dates, groups=groups, n_boot=30,
                             reference_data=X.assign(y_score=s))
    names = {m.name for items in r.sections.values() for m in items}
    assert {"Динамика ROC-AUC", "PSI предсказаний", "PSI признаков", "Ошибки по сегментам"} <= names
    assert r.get("Reliability", "Доверительный интервал: ROC-AUC").details["method"].startswith("cluster")


def test_missing_rows_in_y_are_excluded_consistently_with_x_and_dates():
    y, s, X, dates = _data()
    y = y.copy()
    y[:100] = np.nan
    r = evaluate_predictions(y, s, X=X, dates=dates, n_boot=30)
    assert r.meta["n_used"] == len(y) - 100
    assert any("Исключено строк" in w for w in r.warnings)


def test_modules_work_independently():
    y, s, X, dates = _data()
    assert validate_dataset(X.assign(t=y)).sections  # без target и дат
    assert assess_informativeness(X[["a", "b"]], y).sections  # без dates
    assert evaluate_predictions(y, s, n_boot=30).sections  # без X


def test_run_full_validation_sections_in_order_and_shap_explained():
    y, s, X, dates = _data()
    r = run_full_validation(y, s, X=X, dates=dates, n_boot=30)
    assert list(r.sections) == ["Data Quality", "Data Informativeness", "Model Performance", "Calibration",
                                "Stability", "Reliability", "Error Analysis", "Explainability"]
    shap = r.get("Explainability", "SHAP")
    assert shap.status == NOT_COMPUTABLE and "неизвестной" in shap.interpretation


def test_run_full_validation_without_x_skips_data_checks():
    y, s, *_ = _data()
    r = run_full_validation(y, s, n_boot=30)
    assert "Data Quality" not in r.sections and "X:" in r.meta["skipped_checks"]


def test_every_result_has_explanation_interpretation_limitations():
    y, s, X, dates = _data()
    r = run_full_validation(y, s, X=X, dates=dates, n_boot=30)
    for section, items in r.sections.items():
        for m in items:
            assert m.explanation is not None and m.interpretation and m.limitations is not None, (section, m.name)
            if m.name != "Целевая переменная":
                assert m.limitations.strip() or m.status == NOT_COMPUTABLE, (section, m.name)


def test_report_is_reproducible_with_seed():
    y, s, *_ = _data()
    a = evaluate_predictions(y, s, n_boot=40, seed=7).to_dict()
    b = evaluate_predictions(y, s, n_boot=40, seed=7).to_dict()
    assert a == b


def test_to_jsonable_handles_nan_inf_numpy_and_dataframe():
    out = to_jsonable({"a": np.float64("nan"), "b": np.inf, "c": np.int64(3), "d": pd.DataFrame({"x": [1.0, np.nan]})})
    assert out == {"a": None, "b": None, "c": 3, "d": [{"x": 1.0}, {"x": None}]}
    json.dumps(out)


def test_save_html_pdf_json(tmp_path):
    y, s, X, dates = _data()
    r = run_full_validation(y, s, X=X, dates=dates, n_boot=30)
    html = r.save(str(tmp_path / "r.html"))
    pdf = r.save(str(tmp_path / "r.pdf"))
    js = r.save(str(tmp_path / "r.json"))
    text = open(html, encoding="utf-8").read()
    assert "ROC-AUC" in text and "Ограничения" in text and "data:image/png;base64" in text
    assert open(pdf, "rb").read(4) == b"%PDF"
    assert json.loads(open(js, encoding="utf-8").read())["sections"]["Model Performance"]
    with pytest.raises(ValueError):
        r.save(str(tmp_path / "r.docx"))


def test_html_escapes_untrusted_text(tmp_path):
    rep = ValidationReport()
    rep.add("Data Quality", MetricResult("<script>alert(1)</script>", 1.0, "ok", "<b>x</b>", "<i>y</i>", "z"))
    text = open(rep.save(str(tmp_path / "x.html")), encoding="utf-8").read()
    assert "<script>alert(1)</script>" not in text and "&lt;script&gt;" in text


def test_report_for_one_class_data_still_renders(tmp_path):
    r = evaluate_predictions(np.zeros(400), np.linspace(0.05, 0.95, 400), n_boot=30)
    text = open(r.save(str(tmp_path / "one.html")), encoding="utf-8").read()
    assert "не рассчитано" in text
    r.save(str(tmp_path / "one.pdf"))
