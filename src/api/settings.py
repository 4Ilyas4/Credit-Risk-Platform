"""Настройки API. Читаются из переменных окружения, значения по умолчанию безопасны для локальной работы."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} должен быть целым числом, получено {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} должен быть положительным, получено {value}")
    return value


@dataclass(frozen=True)
class Settings:
    storage_dir: Path = Path("data/api")
    max_upload_mb: int = 200
    max_rows: int = 5_000_000
    max_columns: int = 500
    max_features: int = 200
    max_concurrent_runs: int = 2
    max_cells: int = 100_000_000  # строки * столбцы: защита от сжатых файлов, которые раздуваются в памяти

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            storage_dir=Path(os.environ.get("VALIDATOR_STORAGE_DIR", "data/api")),
            max_upload_mb=_positive_int("VALIDATOR_MAX_UPLOAD_MB", 200),
            max_rows=_positive_int("VALIDATOR_MAX_ROWS", 5_000_000),
            max_columns=_positive_int("VALIDATOR_MAX_COLUMNS", 500),
            max_features=_positive_int("VALIDATOR_MAX_FEATURES", 200),
            max_concurrent_runs=_positive_int("VALIDATOR_MAX_CONCURRENT_RUNS", 2),
            max_cells=_positive_int("VALIDATOR_MAX_CELLS", 100_000_000))
