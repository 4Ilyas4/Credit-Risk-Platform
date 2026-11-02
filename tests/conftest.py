import pytest

from api_helpers import csv_bytes, make_frame, upload
from src.api.app import create_app
from src.api.settings import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(storage_dir=tmp_path / "storage", max_upload_mb=5, max_rows=100_000, max_columns=50,
                    max_features=20, max_cells=2_000_000)


@pytest.fixture
def client(settings):
    from fastapi.testclient import TestClient
    with TestClient(create_app(settings, inline_jobs=True), raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def frame():
    return make_frame()


@pytest.fixture
def dataset(client, frame):
    r = upload(client, csv_bytes(frame))
    assert r.status_code == 201, r.text
    return r.json()
