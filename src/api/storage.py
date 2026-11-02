"""Локальное файловое хранилище: наборы данных, запуски, результаты, отчёты.

Идентификаторы - uuid4 hex. Они всегда проверяются перед построением пути, поэтому обход каталогов
через идентификатор невозможен. Имя файла пользователя в путях не используется.
"""
from __future__ import annotations

import json
import os
import re
import threading
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
