"""Учёт экспериментов в MLflow (локальное хранилище SQLite). Отдельный слой: baseline работает и без него."""
from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path

DEFAULT_URI = "sqlite:///experiments/mlflow.db"
EXPERIMENT = "credit-risk-baseline"


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Отпечаток файла данных: по нему видно, что два запуска использовали одни и те же данные."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    """Короткий хеш коммита или 'no-commit', если коммитов ещё нет или git недоступен."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "no-commit"
    except (OSError, subprocess.SubprocessError):
        return "no-commit"


def flatten_metrics(result: dict) -> dict[str, float]:
    """Плоский словарь метрик для MLflow. Нечисловые значения и NaN/inf пропускаются."""
    flat: dict[str, float] = {}
    for variant, info in result["variants"].items():
        for part in ("valid", "test", "late_not_for_quality", "prior_rate_test"):
            for name, value in info[part].items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    flat[f"{variant}.{part}.{name}"] = float(value)
    return flat


def flatten_params(result: dict, data_path: str | Path) -> dict[str, str]:
    params = {f"split.{k}": v for k, v in result["split_config"].items()}
    params["data.path"] = str(data_path)
    params["data.sha256"] = file_sha256(data_path)
    params["git.commit"] = git_commit()
    for variant, info in result["variants"].items():
        params[f"{variant}.best_C"] = str(info["best_C"])
        params[f"{variant}.n_features"] = str(len(info["features"]))
    return params


def log_run(result: dict, data_path: str | Path, artifacts: list[str | Path] | None = None,
            run_name: str | None = None, tracking_uri: str = DEFAULT_URI) -> str:
    """Записывает параметры, метрики и файлы одного запуска. Возвращает run_id."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(flatten_params(result, data_path))
        mlflow.log_metrics(flatten_metrics(result))
        for path in artifacts or []:
            if Path(path).exists():
                mlflow.log_artifact(str(path))
        return run.info.run_id
