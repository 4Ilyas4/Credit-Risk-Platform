import numpy as np
import pandas as pd
import pytest

from src.validation.informativeness.information_value import information_value, iv_level
from src.validation.performance.ranking_metrics import roc_auc
from src.validation.reliability.bootstrap import bootstrap_metric
from src.validation.result import MetricNotComputable
from src.validation.stability.psi import categorical_psi, numeric_psi, psi_from_proportions, psi_level


def test_psi_hand_computed_and_identity():
    expected = (0.25 - 0.5) * np.log(0.5) + (0.75 - 0.5) * np.log(1.5)
    assert psi_from_proportions(np.array([0.5, 0.5]), np.array([0.25, 0.75])) == pytest.approx(expected)
    assert psi_from_proportions(np.array([0.2, 0.8]), np.array([0.2, 0.8])) == 0.0


def test_psi_handles_empty_bin_without_infinity():
    v = psi_from_proportions(np.array([0.5, 0.5, 0.0]), np.array([0.5, 0.0, 0.5]))
    assert np.isfinite(v) and v > 0.25


def test_numeric_psi_same_vs_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20000)
    same = numeric_psi(ref, rng.normal(0, 1, 20000))["psi"]
    shifted = numeric_psi(ref, rng.normal(1.0, 1, 20000))["psi"]
    assert same < 0.02 and psi_level(same) == "stable"
    assert shifted > 0.25 and psi_level(shifted) == "significant"


def test_psi_too_few_or_constant_not_computable():
    with pytest.raises(MetricNotComputable):
        numeric_psi(np.arange(10), np.arange(10))
    with pytest.raises(MetricNotComputable):
        numeric_psi(np.ones(100), np.arange(100))


def test_categorical_psi_new_category():
    ref = ["a"] * 100 + ["b"] * 100
    res = categorical_psi(ref, ["a"] * 100 + ["c"] * 100)
    assert res["new_categories"] == ["c"] and res["psi"] > 0.25


def test_iv_hand_computed():
    x = pd.Series(["A"] * 100 + ["B"] * 100)
    y = np.array([1] * 10 + [0] * 90 + [1] * 30 + [0] * 70, dtype=float)
    res = information_value(x, y, smooth=0.0)
    # bad: A=10, B=30 из 40; good: A=90, B=70 из 160
    woe_a, woe_b = np.log((90 / 160) / (10 / 40)), np.log((70 / 160) / (30 / 40))
    expected = (90 / 160 - 10 / 40) * woe_a + (70 / 160 - 30 / 40) * woe_b
    assert res["iv"] == pytest.approx(expected, abs=1e-12)
    assert res["table"].set_index("bin").loc["A", "woe"] == pytest.approx(woe_a)
    assert woe_a > 0  # группа A надёжнее среднего: положительный WoE


def test_iv_useless_feature_and_one_class():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=20000))
    y = (rng.random(20000) < 0.2).astype(float)
    assert iv_level(information_value(x, y)["iv"]) == "useless"
    with pytest.raises(MetricNotComputable):
        information_value(x, np.zeros(20000))


def test_iv_missing_is_separate_bin():
    x = pd.Series([1.0, 2.0, np.nan, np.nan] * 50)
    y = np.array([0, 0, 1, 1] * 50, dtype=float)
    assert "missing" in set(information_value(x, y, n_bins=2)["table"]["bin"])


def _make(rng, n, signal=0.1):
    y = (rng.random(n) < 0.2).astype(float)
    return y, np.clip(0.2 + signal * (y - 0.2) + rng.normal(0, 0.15, n), 0, 1)


def test_bootstrap_interval_covers_point_and_is_reproducible():
    y, s = _make(np.random.default_rng(1), 3000)
    a = bootstrap_metric(y, s, roc_auc, n_boot=200, seed=1)
    b = bootstrap_metric(y, s, roc_auc, n_boot=200, seed=1)
    assert a["ci_low"] < a["value"] < a["ci_high"]
    assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]
    assert a["n_failed"] == 0 and a["method"] == "stratified"


def test_bootstrap_interval_narrows_with_more_data():
    rng = np.random.default_rng(3)
    small = bootstrap_metric(*_make(rng, 500), roc_auc, 200)
    large = bootstrap_metric(*_make(rng, 20000), roc_auc, 200)
    assert (small["ci_high"] - small["ci_low"]) > 3 * (large["ci_high"] - large["ci_low"])


def test_cluster_bootstrap_wider_than_stratified_for_dependent_data():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(40), 100)
    effect = rng.normal(0, 1.5, 40)[groups]  # общий эффект группы делает наблюдения зависимыми
    y = (rng.random(4000) < 1 / (1 + np.exp(-(effect - 1.5)))).astype(float)
    s = 1 / (1 + np.exp(-(effect + rng.normal(0, 1.0, 4000))))
    strat = bootstrap_metric(y, s, roc_auc, 300)
    clus = bootstrap_metric(y, s, roc_auc, 300, groups=groups)
    assert clus["std_error"] > strat["std_error"]
    assert clus["method"].startswith("cluster")


def test_bootstrap_one_class_raises_and_few_groups_raise():
    with pytest.raises(MetricNotComputable):
        bootstrap_metric(np.zeros(100), np.linspace(0, 1, 100), roc_auc, 50)
    y, s = _make(np.random.default_rng(0), 500)
    with pytest.raises(MetricNotComputable):
        bootstrap_metric(y, s, roc_auc, 50, groups=np.repeat([0, 1, 2], [200, 200, 100]))
