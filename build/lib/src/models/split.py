"""Временное разделение данных. Случайное разделение и случайный K-Fold не используются."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    """Границы периодов по дате выдачи (включительно).

    Границы выбраны по зрелости: у кредитов 2016+ большая доля ещё в статусе Current,
    поэтому они не участвуют в оценке качества (см. docs/decisions.md, D-006).
    """
    train_end: str = "2013-12-31"
    valid_end: str = "2014-12-31"
    test_end: str = "2015-12-31"


def assign_split(dates: pd.Series, cfg: SplitConfig = SplitConfig()) -> pd.Series:
    """train / valid / test / late. 'late' - незрелый период, только для мониторинга."""
    train_end, valid_end, test_end = (pd.Timestamp(x) for x in
                                      (cfg.train_end, cfg.valid_end, cfg.test_end))
    if not train_end < valid_end < test_end:
        raise ValueError("Границы периодов должны возрастать: train_end < valid_end < test_end")
    labels = np.select(
        [dates <= train_end, dates <= valid_end, dates <= test_end],
        ["train", "valid", "test"], default="late")
    return pd.Series(labels, index=dates.index, name="split")


def expanding_window_folds(dates: pd.Series, valid_years: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Расширяющееся окно: для каждого года - обучение на всех годах до него, проверка на этом году.

    Возвращает булевы маски (train, valid). Обучающая часть всегда раньше проверочной.
    """
    folds = []
    for year in sorted(valid_years):
        train = (dates < pd.Timestamp(year=year, month=1, day=1)).to_numpy()
        valid = (dates.dt.year == year).to_numpy()
        if train.sum() == 0 or valid.sum() == 0:
            raise ValueError(f"Пустая часть для года {year}")
        folds.append((train, valid))
    return folds
