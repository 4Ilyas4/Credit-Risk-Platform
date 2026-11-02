"""Общие функции тестов API. Импортируются как `api_helpers` (pytest добавляет tests/ в sys.path).

Не использовать `from tests.conftest import ...`: в site-packages есть посторонний пакет `tests`.
"""
import io

import numpy as np
import pandas as pd

BASE_MAPPING = {"y_true": "default", "y_score": "score"}


def make_frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    score = np.clip(0.2 + 0.25 * (y - 0.2) + rng.normal(0, 0.12, n), 0.001, 0.999)
    return pd.DataFrame({
        "loan_id": np.arange(n),
        "issue_d": pd.date_range("2014-01-01", periods=n, freq="6h").strftime("%Y-%m-%d"),
        "default": y, "score": score,
        "income": rng.normal(50, 10, n) + 3 * y, "age": rng.normal(40, 8, n),
        "segment": rng.choice(["a", "b", "c"], n)})


def csv_bytes(df: pd.DataFrame, **kw) -> bytes:
    return df.to_csv(index=False, **kw).encode("utf-8")


def upload(client, content: bytes, name="data.csv", ctype="text/csv"):
    return client.post("/datasets", files={"file": (name, io.BytesIO(content), ctype)})


def start_run(client, dataset, mapping=None, options=None, **extra):
    body = {"dataset_id": dataset["dataset_id"], "mapping": mapping or BASE_MAPPING,
            "options": options or {"n_boot": 30}, **extra}
    return client.post("/validation-runs", json=body)
