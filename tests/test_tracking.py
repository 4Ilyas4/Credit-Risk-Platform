import numpy as np
import pytest

pytest.importorskip("mlflow")

from src.models.tracking import file_sha256, flatten_metrics, flatten_params, log_run  # noqa: E402


def _result():
    metrics = {"n": 10, "default_rate": 0.2, "brier": 0.1, "log_loss": 0.4,
               "auc": 0.7, "gini": 0.4, "ks": 0.3}
    nan_metrics = {**metrics, "auc": float("nan"), "gini": float("nan"), "ks": float("nan")}
    info = {"features": ["a", "b"], "best_C": 0.1, "valid": metrics, "test": metrics,
            "late_not_for_quality": nan_metrics, "prior_rate_test": metrics}
    return {"split_config": {"train_end": "2013-12-31"}, "variants": {"full": info}}


def test_flatten_metrics_skips_nan():
    flat = flatten_metrics(_result())
    assert flat["full.test.auc"] == 0.7
    assert "full.late_not_for_quality.auc" not in flat  # NaN не превращается в число
    assert "full.late_not_for_quality.brier" in flat
    assert all(np.isfinite(v) for v in flat.values())


def test_file_sha256_changes_with_content(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"data1")
    b.write_bytes(b"data2")
    assert file_sha256(a) != file_sha256(b)
    assert file_sha256(a) == file_sha256(a)


def test_flatten_params_contains_data_fingerprint(tmp_path):
    f = tmp_path / "d.bin"
    f.write_bytes(b"x")
    params = flatten_params(_result(), f)
    assert params["data.sha256"] == file_sha256(f)
    assert params["full.best_C"] == "0.1" and params["full.n_features"] == "2"
    assert "git.commit" in params


def test_log_run_keeps_every_run(tmp_path):
    import mlflow

    data = tmp_path / "d.bin"
    data.write_bytes(b"x")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    id1 = log_run(_result(), data, run_name="r1", tracking_uri=uri)
    id2 = log_run(_result(), data, run_name="r2", tracking_uri=uri)
    assert id1 != id2  # второй запуск не затирает первый
    runs = mlflow.search_runs(experiment_names=["credit-risk-baseline"])
    assert len(runs) == 2
    assert runs["metrics.full.test.auc"].tolist() == [0.7, 0.7]
