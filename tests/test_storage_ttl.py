"""TTL-очистка: загруженные данные не хранятся бессрочно (Этап D, Шаг 8)."""
import json
import time

import pandas as pd


def _write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def test_cleanup_removes_old_datasets_and_linked_runs(tmp_path):
    from src.api.storage import Storage, now_iso

    storage = Storage(tmp_path)
    # свежий набор (не трогаем)
    fresh_id = storage.save_dataset(pd.DataFrame({"a": [1, 2]}), {"created_at": now_iso()})
    # устаревший набор
    old_id = storage.save_dataset(pd.DataFrame({"a": [1, 2]}), {"created_at": now_iso()})
    # запуск, связанный с устаревшим набором
    run = storage.create_run(old_id, {"mapping": {}})
    (storage.root / "reports").mkdir(exist_ok=True)
    (storage.root / "reports" / f"{run['run_id']}.html").write_text("<html></html>", encoding="utf-8")

    # ставим устаревшему набору created_at давно в прошлом
    old_ts = time.time() - 100 * 3600
    meta_path = storage.root / "datasets" / f"{old_id}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["created_at"] = "__STALE__"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    # перезапишем корректным ISO в прошлом (используем прямой вызов _write_json-подобной записи)
    meta["created_at"] = pd.Timestamp(old_ts, unit="s", tz="UTC").isoformat()
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    removed = storage.cleanup_old(max_age_hours=24.0)
    assert removed == 1
    assert not (storage.root / "datasets" / f"{old_id}.json").exists()
    assert not (storage.root / "datasets" / f"{old_id}.parquet").exists()
    assert not (storage.root / "runs" / f"{run['run_id']}.json").exists()
    assert not (storage.root / "reports" / f"{run['run_id']}.html").exists()
    # свежий набор и его данные на месте
    assert (storage.root / "datasets" / f"{fresh_id}.json").exists()
    assert (storage.root / "datasets" / f"{fresh_id}.parquet").exists()


def test_cleanup_leaves_fresh_datasets(tmp_path):
    from src.api.storage import Storage, now_iso

    storage = Storage(tmp_path)
    storage.save_dataset(pd.DataFrame({"a": [1]}), {"created_at": now_iso()})
    assert storage.cleanup_old(max_age_hours=24.0) == 0
    assert len(list((tmp_path / "datasets").glob("*.json"))) == 1


def test_parse_iso_handles_naive_and_aware():
    from src.api.storage import Storage

    s = Storage(tmp_path := __import__("tempfile").mkdtemp())
    assert s._parse_iso("2026-01-01T00:00:00Z") is not None
    assert s._parse_iso("2026-01-01T00:00:00+00:00") is not None
    assert s._parse_iso("2026-01-01T00:00:00") is not None
    assert s._parse_iso("garbage") is None
    assert s._parse_iso(None) is None