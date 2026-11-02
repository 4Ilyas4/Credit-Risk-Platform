import numpy as np
import pandas as pd
import pytest

from src.models.baseline import prior_rate_metrics, run, score_metrics
from src.models.features import (CATEGORICAL, NUMERIC, QuantileClipper, add_derived,
                                 assert_no_leakage, feature_lists)
from src.models.split import SplitConfig, assign_split, expanding_window_folds


def _dates():
    return pd.Series(pd.to_datetime(["2012-05-01", "2013-06-01", "2014-02-01", "2014-12-01",
                                     "2015-07-01", "2016-01-01"]))


def test_assign_split_boundaries_and_order():
    s = assign_split(_dates())
    assert s.tolist() == ["train", "train", "valid", "valid", "test", "late"]


def test_assign_split_rejects_bad_boundaries():
    with pytest.raises(ValueError):
        assign_split(_dates(), SplitConfig("2015-01-01", "2014-01-01", "2016-01-01"))


def test_expanding_window_train_always_before_valid():
    dates = pd.Series(pd.date_range("2010-01-01", "2014-12-01", freq="MS"))
    folds = expanding_window_folds(dates, [2012, 2013, 2014])
    for train, valid in folds:
        assert dates[train].max() < dates[valid].min()
        assert not (train & valid).any()
    assert folds[1][0].sum() > folds[0][0].sum()  # окно расширяется


def test_expanding_window_empty_fold_raises():
    dates = pd.Series(pd.date_range("2012-01-01", "2012-12-01", freq="MS"))
    with pytest.raises(ValueError):
        expanding_window_folds(dates, [2012])  # до 2012 года данных нет


def test_score_metrics_single_class_gives_nan_not_fake_value():
    m = score_metrics(np.zeros(10), np.linspace(0.1, 0.9, 10))
    assert np.isnan(m["auc"]) and np.isnan(m["gini"]) and np.isnan(m["ks"])
    assert np.isfinite(m["brier"])


def test_score_metrics_perfect_and_prior_baseline():
    y = np.array([0, 0, 1, 1])
    m = score_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["auc"] == 1.0 and m["gini"] == 1.0 and m["ks"] == 1.0
    prior = prior_rate_metrics(pd.Series([0, 1, 0, 0]), pd.Series(y))
    assert prior["auc"] == 0.5 and prior["gini"] == 0.0


def test_assert_no_leakage_blocks_leaky_columns():
    assert_no_leakage(NUMERIC + CATEGORICAL)  # рабочий набор чистый
    with pytest.raises(ValueError):
        assert_no_leakage(["loan_amnt", "recoveries"])


def test_lc_score_exclusion():
    numeric, categorical = feature_lists(exclude_lc_score=True)
    assert "int_rate" not in numeric
    assert "grade" not in categorical and "sub_grade" not in categorical


def test_quantile_clipper_uses_train_quantiles_only():
    clipper = QuantileClipper(0.0, 0.5).fit(np.array([[1.0], [2.0], [3.0]]))
    out = clipper.transform(np.array([[100.0], [0.0]]))
    assert out.ravel().tolist() == [2.0, 1.0]  # границы взяты из fit, а не из новых данных


def _synthetic(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(rng.choice(pd.date_range("2010-01-01", "2016-12-01", freq="MS"), n))
    fico = rng.normal(700, 30, n)
    target = (rng.random(n) < 1 / (1 + np.exp((fico - 650) / 25))).astype("int8")
    df = pd.DataFrame({
        "id": np.arange(n).astype(str), "issue_d": dates, "loan_status": "x", "target": target,
        "earliest_cr_line": "Jan-2000", "annual_inc": rng.lognormal(11, 0.5, n).astype("float32"),
        "revol_bal": rng.lognormal(9, 1, n).astype("float32"), "fico_range_low": fico.astype("float32"),
        "loan_amnt": rng.uniform(1000, 30000, n).astype("float32"),
        "int_rate": rng.uniform(5, 25, n).astype("float32"),
        "installment": rng.uniform(50, 900, n).astype("float32"), "dti": rng.uniform(0, 40, n).astype("float32"),
        "delinq_2yrs": rng.integers(0, 3, n).astype("float32"),
        "inq_last_6mths": rng.integers(0, 4, n).astype("float32"),
        "open_acc": rng.integers(2, 25, n).astype("float32"), "pub_rec": rng.integers(0, 2, n).astype("float32"),
        "revol_util": rng.uniform(0, 100, n).astype("float32"), "total_acc": rng.integers(5, 50, n).astype("float32"),
        "term": rng.choice([" 36 months", " 60 months"], n), "grade": rng.choice(list("ABCD"), n),
        "sub_grade": rng.choice(["A1", "B2", "C3", "D4"], n), "emp_length": rng.choice(["1 year", None], n),
        "home_ownership": rng.choice(["RENT", "OWN"], n), "verification_status": rng.choice(["Verified", "Not Verified"], n),
        "purpose": rng.choice(["car", "other"], n), "addr_state": rng.choice(["CA", "NY"], n)})
    return df


def test_add_derived_has_no_target_or_leakage_columns():
    out = add_derived(_synthetic(200))
    assert list(out.columns) == NUMERIC + CATEGORICAL
    assert out["emp_length"].notna().all()
    assert out["credit_history_years"].min() > 0


def test_add_derived_missing_column_raises():
    with pytest.raises(KeyError):
        add_derived(_synthetic(50).drop(columns=["dti"]))


def test_run_end_to_end_on_synthetic(tmp_path):
    df = _synthetic()
    src = tmp_path / "clean.parquet"
    df.to_parquet(src, index=False)
    cfg = SplitConfig("2013-12-31", "2014-12-31", "2015-12-31")
    res = run(src, tmp_path / "exp", tmp_path / "pred", cfg, c_grid=(0.1, 1.0), cv_years=[2012, 2013])
    full = res["variants"]["full"]
    assert full["test"]["auc"] > 0.6  # в синтетике есть реальный сигнал
    assert full["prior_rate_test"]["auc"] == 0.5  # тривиальный ориентир не ранжирует
    pred = pd.read_parquet(tmp_path / "pred" / "predictions_full.parquet")
    assert set(pred["split"]) == {"valid", "test", "late"}
    assert pred["y_score"].between(0, 1).all() and not pred["id"].duplicated().any()
    assert (tmp_path / "exp" / "baseline_metrics.json").exists()
    # test не пересекается по времени с обучением
    assert pred.loc[pred.split == "test", "issue_d"].min() > pd.Timestamp("2014-12-31")
