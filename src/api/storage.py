"""Локальное файловое хранилище: наборы данных, запуски, результаты, отчёты.

Идентификаторы - uuid4 hex. Они всегда проверяются перед построением пути, поэтому обход каталогов
через идентификатор невозможен. Имя файла пользователя в путях не используется.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.validation.result import to_jsonable

ID_RE = re.compile(r"^[0-9a-f]{32}$")
RUN_FILE_RE = re.compile(r"^[0-9a-f]{32}\.json$")


class NotFound(Exception):
    """Объект не найден или идентификатор имеет недопустимый вид."""


def new_id() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = threading.Lock()

    def _path(self, kind: str, ident: str, suffix: str) -> Path:
        if not isinstance(ident, str) or not ID_RE.match(ident):
            raise NotFound(str(ident)[:40])
        directory = self.root / kind
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{ident}{suffix}"

    def _write_json(self, path: Path, obj: dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(to_jsonable(obj), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # атомарная замена: читатель не увидит наполовину записанный файл

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            raise NotFound(path.name)
        return json.loads(path.read_text(encoding="utf-8"))

    def tmp_path(self) -> Path:
        directory = self.root / "tmp"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{new_id()}.part"

    # --- наборы данных ---
    def save_dataset(self, df: pd.DataFrame, meta: dict) -> str:
        ident = new_id()
        path = self._path("datasets", ident, ".parquet")
        try:
            df.to_parquet(path, index=False)
        except Exception:  # смешанные типы в столбце: приводим текстовые столбцы к строкам и пробуем ещё раз
            fixed = df.copy()
            for col in fixed.select_dtypes(include="object").columns:
                fixed[col] = fixed[col].astype("string")
            fixed.to_parquet(path, index=False)
        self._write_json(self._path("datasets", ident, ".json"), {**meta, "dataset_id": ident})
        return ident

    def dataset_meta(self, ident: str) -> dict:
        return self._read_json(self._path("datasets", ident, ".json"))

    def load_dataset(self, ident: str) -> pd.DataFrame:
        path = self._path("datasets", ident, ".parquet")
        if not path.exists():
            raise NotFound(ident)
        return pd.read_parquet(path)

    # --- запуски ---
    def create_run(self, dataset_id: str, request: dict) -> dict:
        record = {"run_id": new_id(), "dataset_id": dataset_id, "status": "queued", "created_at": now_iso(),
                  "started_at": None, "finished_at": None, "request": request, "timings": None,
                  "summary": None, "report_formats": [], "error": None}
        self._write_json(self._path("runs", record["run_id"], ".json"), record)
        return record

    def get_run(self, ident: str) -> dict:
        return self._read_json(self._path("runs", ident, ".json"))

    def update_run(self, ident: str, **fields) -> dict:
        with self._lock:
            record = self.get_run(ident)
            record.update(fields)
            self._write_json(self._path("runs", ident, ".json"), record)
            return record

    def save_result(self, ident: str, result: dict) -> None:
        self._write_json(self._path("runs", ident, ".result.json"), result)

    def load_result(self, ident: str) -> dict:
        return self._read_json(self._path("runs", ident, ".result.json"))

    def report_path(self, ident: str, fmt: str) -> Path:
        if fmt not in ("html", "pdf", "json"):
            raise NotFound(fmt)
        return self._path("reports", ident, f".{fmt}")

    @staticmethod
    def _parse_iso(value: str | None) -> float | None:
        """Дата/время из created_at в timestamp (UTC). Нечитаемое -> None."""
        if not value:
            return None
        s = value if value.endswith(("Z", "+00:00", "+0000")) else value + "Z"
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00").replace("+0000", "+00:00")).timestamp()
        except ValueError:
            return None

    def cleanup_old(self, max_age_hours: float = 24.0) -> int:
        """TTL-очистка: удаляет наборы данных старше max_age_hours и их запуски и отчёты.

        Цель: загруженные пользователем данные не хранятся бессрочно. Вызывается при
        старте приложения (и может вызываться периодически). Возвращает число удалённых наборов.
        """
        cutoff = time.time() - max_age_hours * 3600
        ds_dir = self.root / "datasets"
        if not ds_dir.exists():
            return 0
        removed = 0
        for json_path in ds_dir.glob("*.json"):
            ident = json_path.stem
            try:
                ts = self._parse_iso(json.loads(json_path.read_text(encoding="utf-8")).get("created_at"))
            except Exception:
                ts = None
            if ts is None or ts >= cutoff:
                continue
            # связанные запуски (записи ссылаются на dataset_id)
            run_ids = []
            runs_dir = self.root / "runs"
            if runs_dir.exists():
                for rp in runs_dir.glob("*.json"):
                    try:
                        rec = json.loads(rp.read_text(encoding="utf-8"))
                        if rec.get("dataset_id") == ident and rec.get("run_id"):
                            run_ids.append(rec["run_id"])
                    except Exception:
                        pass
            for p in (json_path, ds_dir / f"{ident}.parquet"):
                p.unlink(missing_ok=True)
            for rid in run_ids:
                for suffix in (".json", ".result.json"):
                    (runs_dir / f"{rid}{suffix}").unlink(missing_ok=True)
                for fmt in ("html", "pdf", "json"):
                    (self.root / "reports" / f"{rid}.{fmt}").unlink(missing_ok=True)
            removed += 1
        return removed

    def fail_stale_runs(self) -> int:
        """После перезапуска сервера запуски в статусе queued/running уже не выполняются: помечаем failed."""
        directory = self.root / "runs"
        count = 0
        if directory.exists():
            for path in directory.iterdir():
                if not RUN_FILE_RE.match(path.name):
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("status") in ("queued", "running"):
                    self.update_run(record["run_id"], status="failed", finished_at=now_iso(),
                                    error="Запуск прерван перезапуском сервера")
                    count += 1
        tmp = self.root / "tmp"
        if tmp.exists():
            for leftover in tmp.glob("*.part"):
                leftover.unlink(missing_ok=True)
        return count
