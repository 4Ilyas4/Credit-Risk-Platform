"""Приём файлов: формат определяется по содержимому, размеры ограничены, структура проверяется."""
from __future__ import annotations

import csv
import hashlib
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.api.describe import describe_columns, structure_warnings, suggest_mapping
from src.api.errors import FileTooLarge, InvalidDataset, UnsupportedFile
from src.api.settings import Settings

PARQUET_MAGIC = b"PAR1"
SNIFF_BYTES = 64 * 1024


def display_name(raw: str | None) -> str:
    """Имя файла только для показа. В путях не используется. Убираем каталоги и управляющие символы."""
    name = (raw or "upload").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    return (name or "upload")[:200]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def detect_format(path: Path, filename: str) -> str:
    """Формат по содержимому: parquet узнаётся по сигнатуре, CSV - по расширению и читаемому тексту."""
    with open(path, "rb") as f:
        head = f.read(SNIFF_BYTES)
    if not head:
        raise InvalidDataset("Файл пустой.")
    if head[:4] == PARQUET_MAGIC:
        return "parquet"
    ext = Path(filename).suffix.lower()
    if ext == ".parquet":
        raise UnsupportedFile("Файл имеет расширение .parquet, но не является Parquet.")
    if ext not in (".csv", ".txt"):
        raise UnsupportedFile(f"Поддерживаются CSV и Parquet. Получено расширение {ext or 'без расширения'!r}.")
    if b"\x00" in head:
        raise UnsupportedFile("Файл содержит бинарные данные и не похож на CSV.")
    try:
        head.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        truncated_tail = len(head) == SNIFF_BYTES and exc.start >= SNIFF_BYTES - 4  # обрезанный символ на краю
        if not truncated_tail:
            raise UnsupportedFile("CSV не в кодировке UTF-8. Сохраните файл как UTF-8.") from exc
    return "csv"


def _sniff_csv(path: Path) -> tuple[str, str]:
    """Разделитель и десятичный знак. Русский Excel сохраняет CSV с ';' и запятой в числах."""
    with open(path, "rb") as f:
        sample = f.read(SNIFF_BYTES).decode("utf-8-sig", errors="ignore")
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    decimal = "."
    if sep == ";" and re.search(r"\d,\d", sample) and not re.search(r"\d\.\d", sample):
        decimal = ","
    return sep, decimal


def _check_limits(rows: int, cols: int, settings: Settings) -> None:
    if rows == 0:
        raise InvalidDataset("В файле нет строк с данными.")
    if cols == 0:
        raise InvalidDataset("В файле нет столбцов.")
    if rows > settings.max_rows:
        raise FileTooLarge(f"Слишком много строк: больше {settings.max_rows}.")
    if cols > settings.max_columns:
        raise FileTooLarge(f"Слишком много столбцов: {cols}, максимум {settings.max_columns}.")
    if rows * cols > settings.max_cells:
        raise FileTooLarge(f"Слишком большая таблица: {rows}x{cols} ячеек, максимум {settings.max_cells}.")


def read_table(path: Path, fmt: str, settings: Settings) -> tuple[pd.DataFrame, list[str]]:
    """Читает файл. Для Parquet лимиты проверяются по метаданным до загрузки в память."""
    notes: list[str] = []
    try:
        if fmt == "parquet":
            meta = pq.read_metadata(path)  # сам закрывает файл; ParquetFile держит дескриптор при ошибке
            _check_limits(meta.num_rows, meta.num_columns, settings)
            df = pd.read_parquet(path)
        else:
            sep, decimal = _sniff_csv(path)
            if sep != ",":
                notes.append(f"Определён разделитель {sep!r}.")
            if decimal == ",":
                notes.append("Определена десятичная запятая: числа вида 0,25 прочитаны как 0.25.")
            df = pd.read_csv(path, sep=sep, decimal=decimal, encoding="utf-8-sig", low_memory=False,
                             nrows=settings.max_rows + 1)
            _check_limits(len(df), df.shape[1], settings)
    except (FileTooLarge, InvalidDataset):
        raise
    except Exception as exc:  # битые файлы дают разные типы ошибок pandas/pyarrow; детали наружу не отдаём
        raise InvalidDataset(f"Не удалось прочитать файл как {fmt.upper()} ({type(exc).__name__}).") from exc
    df.columns = [str(c) for c in df.columns]
    return df, notes


def inspect_upload(path: Path, filename: str, settings: Settings) -> tuple[pd.DataFrame, dict]:
    """Полная проверка загруженного файла. Возвращает таблицу и метаданные для ответа API."""
    t0 = time.perf_counter()
    fmt = detect_format(path, filename)
    df, notes = read_table(path, fmt, settings)
    columns = describe_columns(df)
    head = df.head(10)
    preview = head.astype(object).where(head.notna(), None).to_dict("records")
    meta = {"filename": display_name(filename), "format": fmt, "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path), "n_rows": int(len(df)), "n_columns": int(df.shape[1]),
            "columns": [c.model_dump() for c in columns], "preview": preview,
            "warnings": notes + structure_warnings(df, columns),
            "suggested_mapping": suggest_mapping(columns).model_dump(),
            "parse_seconds": round(time.perf_counter() - t0, 3)}
    return df, meta
