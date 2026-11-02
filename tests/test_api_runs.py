import json

import numpy as np
import pandas as pd
import pytest

from api_helpers import BASE_MAPPING, csv_bytes, make_frame, start_run, upload


def test_minimal_run_completes_and_returns_result(client, dataset):
    r = start_run(client, dataset)
    assert r.status_code == 202, r.text
    run = r.json()
    assert len(run["run_id"]) == 32 and run["dataset_id"] == dataset["dataset_id"]
    done = client.get(f"/validation-runs/{run['run_id']}").json()
    assert done["status"] == "completed" and done["error"] is None
    assert 0.6 < done["summary"]["auc"] < 0.95
    assert {"validation_seconds", "report_seconds"} <= set(done["timings"])
    sections = done["result"]["sections"]
    assert "Model Performance" in sections and "Data Quality" not in sections  # признаки не переданы
    assert "X:" in done["summary"]["skipped_checks"]


def test_full_run_with_all_optional_inputs(client, dataset):
    mapping = {**BASE_MAPPING, "date": "issue_d", "group": "segment", "features": ["income", "age"]}
    run = start_run(client, dataset, mapping, reference={"period": {"end": "2014-06-30"}},
                    current_period={"start": "2014-07-01"}).json()
    done = client.get(f"/validation-runs/{run['run_id']}").json()
    assert done["status"] == "completed", done["error"]
    names = {m["name"] for items in done["result"]["sections"].values() for m in items}
    assert {"Динамика ROC-AUC", "PSI предсказаний", "PSI признаков", "Information Value"} <= names
    assert done["result"]["sections"]["Data Quality"]


def test_reference_from_another_dataset(client, dataset):
    ref = upload(client, csv_bytes(make_frame(1500, seed=9).assign(score=lambda d: d.score * 0.8))).json()
    mapping = {**BASE_MAPPING, "features": ["income"]}
    run = start_run(client, dataset, mapping, reference={"dataset_id": ref["dataset_id"]}).json()
    done = client.get(f"/validation-runs/{run['run_id']}").json()
    assert done["status"] == "completed", done["error"]
    assert any(m["name"] == "PSI предсказаний" for m in done["result"]["sections"]["Stability"])


def test_get_unknown_run_is_404(client):
    assert client.get("/validation-runs/" + "0" * 32).status_code == 404
    assert client.get("/validation-runs/not-an-id").status_code == 404


def test_unknown_dataset_is_422(client):
    r = client.post("/validation-runs", json={"dataset_id": "0" * 32, "mapping": BASE_MAPPING})
    assert r.status_code == 422 and "не найден" in r.json()["error"]


def test_current_period_limits_rows_inclusive(client, dataset):
    m = {**BASE_MAPPING, "date": "issue_d"}
    run = start_run(client, dataset, m, current_period={"start": "2014-01-01", "end": "2014-01-31"}).json()
    done = client.get(f"/validation-runs/{run['run_id']}").json()
    expected = int((pd.to_datetime(make_frame().issue_d) < "2014-02-01").sum())
    assert done["summary"]["n_used"] == expected


def test_results_are_reproducible_across_runs(client, dataset):
    a = client.get(f"/validation-runs/{start_run(client, dataset).json()['run_id']}").json()["result"]
    b = client.get(f"/validation-runs/{start_run(client, dataset).json()['run_id']}").json()["result"]
    assert a == b


def test_different_seed_changes_only_intervals(client, dataset):
    def run(seed):
        r = start_run(client, dataset, options={"n_boot": 30, "seed": seed}).json()["run_id"]
        return client.get(f"/validation-runs/{r}").json()["result"]["sections"]
    a, b = run(1), run(2)
    auc = lambda s: next(m for m in s["Model Performance"] if m["name"] == "ROC-AUC")["value"]  # noqa: E731
    ci = lambda s: next(m for m in s["Reliability"] if "ROC-AUC" in m["name"])["details"]["ci_low"]  # noqa: E731
    assert auc(a) == auc(b) and ci(a) != ci(b)


def test_api_result_matches_direct_library_call(client, dataset, frame):
    from src.validation import evaluate_predictions
    direct = evaluate_predictions(frame["default"], frame["score"], n_boot=30).get("Model Performance", "ROC-AUC").value
    rid = start_run(client, dataset).json()["run_id"]
    assert client.get(f"/validation-runs/{rid}").json()["summary"]["auc"] == pytest.approx(direct, abs=1e-12)


def test_one_class_dataset_completes_with_not_computable(client):
    ds = upload(client, csv_bytes(make_frame().assign(default=0))).json()
    done = client.get(f"/validation-runs/{start_run(client, ds).json()['run_id']}").json()
    assert done["status"] == "completed"
    auc = next(m for m in done["result"]["sections"]["Model Performance"] if m["name"] == "ROC-AUC")
    assert auc["value"] is None and auc["status"] == "not_computable"
    assert done["summary"]["auc"] is None and done["summary"]["n_not_computable"] > 0


def test_missing_values_in_target_and_score_are_dropped_and_reported(client):
    df = make_frame()
    df.loc[:49, "default"] = np.nan
    df.loc[50:79, "score"] = np.nan
    ds = upload(client, csv_bytes(df)).json()
    done = client.get(f"/validation-runs/{start_run(client, ds).json()['run_id']}").json()
    assert done["status"] == "completed" and done["summary"]["n_used"] == len(df) - 80
    assert any("Исключено строк" in w for w in done["result"]["warnings"])


def test_all_target_missing_fails_with_clear_message(client):
    ds = upload(client, csv_bytes(make_frame().assign(default=np.nan))).json()
    done = client.get(f"/validation-runs/{start_run(client, ds).json()['run_id']}").json()
    assert done["status"] == "failed" and done["result"] is None
    assert "Traceback" not in json.dumps(done) and done["error"]
