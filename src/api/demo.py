"""Демонстрационный набор для кнопки «Демо» в интерфейсе.

Это синтетические данные с известными свойствами: хорошая модель, но с лёгким
завышением риска (видна в калибровке) и датами (работают периоды и референс).
Инструмент проверяет скоры, а не создаёт их: здесь скор генерируется только
для демонстрации пути «файл → вердикт».
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.api.ingest import dataset_meta
from src.api.settings import Settings

SEED = 42
FILENAME = "demo_scoring_results.csv"
FORMAT = "csv"


def demo_frame(n: int = 3000, seed: int = SEED) -> pd.DataFrame:
    """Синтетические результаты скоринга.

    Скор — сигмоид от скрытого фактора x с лёгким завышением риска (p * 1.12,
    клип в [0.001, 0.999]): калибровка даст warning, AUC останется ~0.8.
    Это честная демонстрация: видны и сильные стороны, и предупреждение.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(1.2 * x - 1.386)))  # ~20% дефолтов
    y = (rng.random(n) < p).astype(int)
    score = np.clip(p * 1.12, 0.001, 0.999)  # слегка завышает риск
    dates = pd.date_range("2020-01-01", periods=n, freq="3h")
    df = pd.DataFrame({
        "loan_id": np.arange(n),
        "issue_d": dates.strftime("%Y-%m-%d"),
        "default": y, "score": score,
        "income": np.round(rng.normal(50, 10, n) + 3 * y, 1),
        "age": np.round(rng.normal(40, 8, n), 1),
        "dti": np.round(rng.normal(18, 7, n), 2),
        "segment": rng.choice(["Стандарт", "Prime", "Subprime"], n, p=[0.5, 0.35, 0.15]),
    })
    return df


def demo_dataset(settings: Settings) -> dict:
    """Собирает демо-набор в формате ответа `POST /datasets` (без сохранения в хранилище).

    Возвращает (meta, df): идентификатор назначается в app.py при сохранении.
    """
    df = demo_frame()
    meta = dataset_meta(df, FILENAME, FORMAT, int(df.memory_usage(deep=True).sum()), _fake_sha(df),
                        notes=[f"Демо-набор: {len(df)} строк, сгенерирован с seed={SEED}"], settings=settings)
    meta["parse_seconds"] = 0.0
    return meta, df


def _fake_sha(df: pd.DataFrame) -> str:
    """Стабильный хэш демо-данных: SHA-256 от представления строк с тем же seed."""
    return hashlib.sha256(demo_frame().to_csv(index=False).encode()).hexdigest()