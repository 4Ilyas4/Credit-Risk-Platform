"""Проверка входных данных запроса: отсутствующие поля, неверные типы, ошибки сопоставления столбцов."""
import numpy as np
import pandas as pd
import pytest

from api_helpers import BASE_MAPPING, csv_bytes, start_run, upload


@pytest.mark.parametrize("body,field", [
    ({"mapping": BASE_MAPPING}, "dataset_id"),
    ({"dataset_id": "x"}, "mapping"),
    ({"dataset_id": "x", "mapping": {"y_true": "default"}}, "mapping.y_score"),
    ({"dataset_id": "x", "mapping": {"y_true": "", "y_score": "score"}}, "mapping.y_true"),
])
def test_missing_required_fields_are_422_with_field_name(client, body, field):
    r = client.post("/validation-runs", json=body)
    assert r.status_code == 422
    assert field in [d["field"] for d in r.json()["detail"]]


@pytest.mark.parametrize("patch", [
    {"mapping": {"y_true": 5, "y_score": "score"}},
    {"mapping": {"y_true": "default", "y_score": "score", "features": "income"}},
    {"options": {"threshold": "high"}},
    {"options": {"n_boot": 5}},
    {"options": {"n_boot": 10**6}},
    {"options": {"threshold": 1.5}},
    {"options": {"seed": -1}},
    {"mapping": {"y_true": "default", "y_score": "score", "typo_field": "x"}},
    {"unknown_top_level": 1},
    {"current_period": {"start": "not-a-date"}},
    {"current_period": {}},
    {"current_period": {"start": "2015-01-01", "end": "2014-01-01"}},
    {"reference": {}},
    {"reference": {"dataset_id": "a" * 32, "period": {"start": "2014-01-01"}}},
])
def test_wrong_types_and_values_are_422(client, dataset, patch):
    body = {"dataset_id": dataset["dataset_id"], "mapping": BASE_MAPPING, **patch}
    r = client.post("/validation-runs", json=body)
    assert r.status_code == 422, (patch, r.text)
    assert r.json()["error"]


def test_malformed_json_body_is_422_not_500(client):
    r = client.post("/validation-runs", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


@pytest.mark.parametrize("mapping,fragment", [
    ({"y_true": "nope", "y_score": "score"}, "не найден"),
    ({"y_true": "default", "y_score": "default"}, "разными"),
    ({"y_true": "segment", "y_score": "score"}, "числовым"),
    ({"y_true": "income", "y_score": "score"}, "только 0 и 1"),
    ({"y_true": "default", "y_score": "income"}, "[0, 1]"),
    ({"y_true": "default", "y_score": "segment"}, "числовым"),
    ({"y_true": "default", "y_score": "score", "date": "income"}, "числовой"),
    ({"y_true": "default", "y_score": "score", "date": "segment"}, "как дату"),
    ({"y_true": "default", "y_score": "score", "features": ["nope"]}, "не найден"),
    ({"y_true": "default", "y_score": "score", "features": ["income", "income"]}, "повторы"),
    ({"y_true": "default", "y_score": "score", "features": ["default"]}, "нельзя использовать"),
    ({"y_true": "default", "y_score": "score", "group": "nope"}, "не найден"),
])
def test_mapping_errors_are_clear_and_create_no_run(client, dataset, settings, mapping, fragment):
    r = start_run(client, dataset, mapping)
    assert r.status_code == 422 and fragment in r.json()["error"], r.text
    runs = settings.storage_dir / "runs"
    assert not runs.exists() or not list(runs.glob("*.json"))


def test_too_many_features_is_422(client, settings):
    cols = {f"f{i}": np.arange(30.0) for i in range(settings.max_features + 1)}
    df = pd.DataFrame({"default": [0, 1] * 15, "score": np.linspace(0.1, 0.9, 30), **cols})
    ds = upload(client, csv_bytes(df)).json()
    r = start_run(client, ds, {**BASE_MAPPING, "features": list(cols)})
    assert r.status_code == 422 and "много признаков" in r.json()["error"]


def test_periods_require_date_column_and_must_have_rows(client, dataset):
    r = start_run(client, dataset, BASE_MAPPING, current_period={"start": "2014-01-01"})
    assert r.status_code == 422 and "mapping.date" in r.json()["error"]
    m = {**BASE_MAPPING, "date": "issue_d"}
    assert "нет строк" in start_run(client, dataset, m, current_period={"start": "2030-01-01"}).json()["error"]
    overlap = start_run(client, dataset, m, current_period={"start": "2014-01-01", "end": "2015-01-01"},
                        reference={"period": {"start": "2014-06-01", "end": "2014-12-31"}})
    assert "пересекаются" in overlap.json()["error"]


def test_reference_dataset_without_score_column_is_422(client, dataset):
    other = upload(client, csv_bytes(pd.DataFrame({"x": range(50)}))).json()
    r = start_run(client, dataset, reference={"dataset_id": other["dataset_id"]})
    assert r.status_code == 422 and "нет столбца скора" in r.json()["error"]
