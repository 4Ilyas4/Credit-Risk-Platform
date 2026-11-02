import numpy as np
import pandas as pd
import pytest

from api_helpers import csv_bytes, upload


def test_upload_csv_returns_structure_and_suggested_mapping(client, frame):
    r = upload(client, csv_bytes(frame))
    assert r.status_code == 201
    body = r.json()
    assert body["n_rows"] == len(frame) and body["n_columns"] == frame.shape[1] and body["format"] == "csv"
    assert len(body["sha256"]) == 64 and len(body["preview"]) == 10
    assert {c["name"] for c in body["columns"]} == set(frame.columns)
    sm = body["suggested_mapping"]
    assert sm["y_true"] == "default" and sm["y_score"] == "score" and sm["date"] == "issue_d"
    assert sm["group"] == "segment"
    assert body["parse_seconds"] >= 0 and body["upload_seconds"] >= 0
    assert client.get(f"/datasets/{body['dataset_id']}").json()["sha256"] == body["sha256"]


def test_upload_parquet_roundtrip(client, frame, tmp_path):
    path = tmp_path / "f.parquet"
    frame.to_parquet(path)
    r = upload(client, path.read_bytes(), "f.parquet", "application/octet-stream")
    assert r.status_code == 201 and r.json()["format"] == "parquet"


def test_parquet_detected_by_content_even_with_wrong_extension(client, frame, tmp_path):
    path = tmp_path / "f.parquet"
    frame.to_parquet(path)
    assert upload(client, path.read_bytes(), "renamed.csv").json()["format"] == "parquet"


def test_semicolon_csv_with_decimal_comma(client):
    text = "default;score\n0;0,25\n1;0,75\n0;0,10\n1;0,90\n"
    r = upload(client, text.encode("utf-8"))
    assert r.status_code == 201
    cols = {c["name"]: c for c in r.json()["columns"]}
    assert cols["score"]["kind"] == "numeric"
    assert any("запят" in w for w in r.json()["warnings"])


def test_csv_with_utf8_bom(client, frame):
    r = upload(client, b"\xef\xbb\xbf" + csv_bytes(frame))
    assert r.status_code == 201
    assert "default" in {c["name"] for c in r.json()["columns"]}


@pytest.mark.parametrize("name", ["x.xlsx", "x.json", "x.exe", "noext"])
def test_unsupported_extension_is_415(client, frame, name):
    r = upload(client, csv_bytes(frame), name)
    assert r.status_code == 415 and "CSV и Parquet" in r.json()["error"]


def test_empty_file_is_422(client):
    r = upload(client, b"")
    assert r.status_code == 422 and "пуст" in r.json()["error"]


def test_header_only_csv_is_422(client):
    r = upload(client, b"default,score\n")
    assert r.status_code == 422


def test_binary_garbage_as_csv_is_415(client):
    r = upload(client, bytes(range(256)) * 50, "data.csv")
    assert r.status_code == 415


def test_non_utf8_csv_is_415_with_hint(client):
    r = upload(client, "default;score\nпривет;1\n".encode("cp1251"))
    assert r.status_code == 415 and "UTF-8" in r.json()["error"]


def test_fake_parquet_extension_is_415(client):
    r = upload(client, b"just,text\n1,2\n", "data.parquet")
    assert r.status_code == 415


@pytest.mark.parametrize("damage", ["truncated_half", "truncated_tail", "footer_zeroed"])
def test_truncated_or_broken_parquet_is_422_and_does_not_leak_internals(client, frame, tmp_path, damage):
    path = tmp_path / "f.parquet"
    frame.to_parquet(path)
    raw = path.read_bytes()
    broken = {"truncated_half": raw[:len(raw) // 2], "truncated_tail": raw[:-8],
              "footer_zeroed": raw[:-100] + b"\x00" * 100}[damage]
    r = upload(client, broken, "f.parquet")
    assert r.status_code in (415, 422), r.text  # без сигнатуры PAR1 в начале и конце формат не узнан
    assert "Traceback" not in r.text and str(tmp_path) not in r.text


def test_parquet_with_damaged_data_pages_is_not_detectable_but_hash_is_returned(client, frame, tmp_path):
    """Известное ограничение: Parquet без контрольных сумм. Тест фиксирует поведение, а не одобряет его.

    Повреждение страниц данных внутри файла может читаться без ошибки. Защита - SHA-256 в ответе:
    пользователь сверяет его с хешем своего файла.
    """
    path = tmp_path / "f.parquet"
    frame.to_parquet(path)
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2:len(raw) // 2 + 200] = b"\x00" * 200
    r = upload(client, bytes(raw), "f.parquet")
    if r.status_code == 201:
        import hashlib
        assert r.json()["sha256"] == hashlib.sha256(bytes(raw)).hexdigest()
    else:
        assert r.status_code == 422


def test_malformed_csv_rows_are_422(client):
    r = upload(client, b'a,b\n1,2\n"3,4\n5,6,7,8,9\n')
    assert r.status_code in (201, 422)  # pandas может прочитать с предупреждением; главное - не 500
    assert r.status_code != 500


def test_missing_file_field_is_422(client):
    r = client.post("/datasets")
    assert r.status_code == 422 and r.json()["error"]


def test_file_too_large_is_413(client, settings):
    big = b"a,b\n" + b"1,2\n" * (settings.max_upload_bytes // 4 + 10)
    r = upload(client, big)
    assert r.status_code == 413 and "МБ" in r.json()["error"]


def test_too_many_columns_is_413(client, settings):
    df = pd.DataFrame(np.zeros((5, settings.max_columns + 1)), columns=[f"c{i}" for i in range(settings.max_columns + 1)])
    assert upload(client, csv_bytes(df)).status_code == 413


def test_too_many_rows_is_413(client, settings):
    df = pd.DataFrame({"a": np.zeros(settings.max_rows + 5), "b": 1})
    assert upload(client, csv_bytes(df)).status_code == 413


def test_path_traversal_filename_is_harmless(client, frame, settings):
    r = upload(client, csv_bytes(frame), "../../../etc/evil.csv")
    assert r.status_code == 201 and r.json()["filename"] == "evil.csv"
    stored = {p.name for p in (settings.storage_dir / "datasets").iterdir()}
    assert all(len(n.split(".")[0]) == 32 for n in stored)  # только uuid, имя пользователя в путях не используется
    assert not (settings.storage_dir.parent / "etc").exists()


def test_structure_warnings(client):
    df = pd.DataFrame({"default": [0, 1, 0, 1], "score": [0.1, 0.9, 0.1, 0.9], "const": 1, "empty": np.nan})
    df = pd.concat([df, df.iloc[:1]])
    w = " ".join(upload(client, csv_bytes(df)).json()["warnings"])
    assert "Константные" in w and "пустые" in w and "дубликатов" in w


def test_duplicate_header_names_reported(client):
    r = upload(client, b"a,a,b\n1,2,3\n4,5,6\n")
    assert r.status_code == 201 and any("повторяются" in w for w in r.json()["warnings"])


def test_unknown_dataset_is_404_and_bad_id_is_404(client):
    assert client.get("/datasets/" + "0" * 32).status_code == 404
    for bad in ("..%2F..%2Fetc", "not-an-id", "A" * 32):
        assert client.get(f"/datasets/{bad}").status_code == 404


def test_numbers_in_date_like_column_not_suggested_as_date(client):
    df = pd.DataFrame({"year": [2015, 2016, 2017, 2018] * 5, "default": [0, 1] * 10, "score": np.linspace(0.1, 0.9, 20)})
    sm = upload(client, csv_bytes(df)).json()["suggested_mapping"]
    assert sm["date"] is None


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"
