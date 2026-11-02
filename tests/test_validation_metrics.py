import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.calibration import calibration_curve as sk_calibration_curve

from src.validation.calibration.calibration_metrics import (
    calibration_curve, calibration_in_the_large, expected_calibration_error)
from src.validation.performance.classification_metrics import (
    balanced_accuracy, brier_score, confusion_counts, f1, log_loss, precision, recall)
from src.validation.performance.ranking_metrics import (
    average_ranks, gini, ks_statistic, pr_auc, roc_auc)
from src.validation.result import MetricNotComputable


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.2).astype(float)
    s = np.clip(0.2 + 0.15 * (y - 0.2) + rng.normal(0, 0.15, 5000), 0, 1)
    s[:500] = np.round(s[:500], 1)  # много совпадающих значений: проверка обработки ничьих
    return y, s


def test_average_ranks_with_ties():
    assert average_ranks(np.array([10.0, 20.0, 20.0, 30.0])).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_ranking_metrics_match_sklearn(data):
    y, s = data
    assert roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s), abs=1e-12)
    assert pr_auc(y, s) == pytest.approx(skm.average_precision_score(y, s), abs=1e-12)
    fpr, tpr, _ = skm.roc_curve(y, s)
    assert ks_statistic(y, s) == pytest.approx(np.max(tpr - fpr), abs=1e-12)
    assert gini(y, s) == pytest.approx(2 * skm.roc_auc_score(y, s) - 1, abs=1e-12)


def test_auc_hand_computed():
    y = np.array([0, 0, 1, 1.0])
    assert roc_auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == 0.75  # 3 из 4 пар упорядочены верно
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5  # полная ничья
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0  # инвертированный скор


def test_probability_and_threshold_metrics_match_sklearn(data):
    y, s = data
    assert brier_score(y, s) == pytest.approx(skm.brier_score_loss(y, s), abs=1e-12)
    assert log_loss(y, s) == pytest.approx(skm.log_loss(y, s), abs=1e-12)
    c = confusion_counts(y, s, 0.25)
    pred = s >= 0.25
    assert precision(c) == pytest.approx(skm.precision_score(y, pred))
    assert recall(c) == pytest.approx(skm.recall_score(y, pred))
    assert f1(c) == pytest.approx(skm.f1_score(y, pred))
    assert balanced_accuracy(c) == pytest.approx(skm.balanced_accuracy_score(y, pred))
    assert sum(c.values()) == len(y)


def test_brier_hand_computed():
    assert brier_score(np.array([1.0, 0.0]), np.array([0.8, 0.3])) == pytest.approx((0.04 + 0.09) / 2)


@pytest.mark.parametrize("fn", [roc_auc, gini, ks_statistic, pr_auc])
@pytest.mark.parametrize("label", [0.0, 1.0])
def test_one_class_is_not_computable_not_fake_value(fn, label):
    with pytest.raises(MetricNotComputable):
        fn(np.full(50, label), np.linspace(0.05, 0.95, 50))


def test_precision_undefined_when_nothing_predicted():
    c = confusion_counts(np.array([0, 1.0, 0, 1]), np.array([0.1, 0.2, 0.1, 0.2]), threshold=0.9)
    with pytest.raises(MetricNotComputable):
        precision(c)


def test_calibration_matches_sklearn(data):
    y, s = data
    for strategy in ("uniform", "quantile"):
        pt, pp = sk_calibration_curve(y, s, n_bins=10, strategy=strategy)
        c = calibration_curve(y, s, 10, strategy)
        assert len(c) == len(pt)
        np.testing.assert_allclose(c["observed"], pt, atol=1e-12)
        np.testing.assert_allclose(c["predicted"], pp, atol=1e-12)


def test_ece_perfect_and_hand_computed():
    # 10 наблюдений с прогнозом 0.3 (факт 0.3) и 10 с прогнозом 0.9 (факт 0.9): идеальная калибровка
    y = np.array([1.0] * 3 + [0.0] * 7 + [1.0] * 9 + [0.0] * 1)
    s = np.array([0.3] * 10 + [0.9] * 10)
    assert expected_calibration_error(y, s, 10, "uniform") == pytest.approx(0.0, abs=1e-12)
    # один и тот же прогноз 0.5 для всех: одна корзина, факт 0.6, ECE = |0.5 - 0.6| = 0.1
    assert expected_calibration_error(y, np.full(20, 0.5), 10, "uniform") == pytest.approx(0.1)


def test_calibration_in_the_large():
    r = calibration_in_the_large(np.array([1.0, 0, 0, 0]), np.array([0.1, 0.1, 0.1, 0.1]))
    assert r["difference"] == pytest.approx(-0.15) and r["ratio"] == pytest.approx(0.4)
