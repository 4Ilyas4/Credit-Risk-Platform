"""Генераторы синтетических датасетов с известными свойствами для тестов пакета валидации.

Каждый генератор закладывает ровно один дефект (или ни одного для good_model).
Ожидаемый статус дефекта описан в docstring и проверяется в test_validation_synthetic.py.
Это эталонные наборы: если тест на них падает, значит, валидатор не находит заложенный
дефект (или даёт ложную тревогу на чистом наборе), а не «данные такие».

Скоры строятся как сигмоид от скрытого признака: по построению они калиброваны
(mean(score) ≈ доля дефолтов), поэтому на чистом наборе калибровка не даёт ложных тревог.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# logit(0.2) = ln(0.25) ≈ -1.386: средняя доля дефолтов около 20%.
_INTERCEPT = -1.386


def _dates(n: int, start: str = "2015-01-01", freq: str = "12h") -> pd.Series:
    """Даты с двумя наблюдениями в день: при n=2000 это ~2.7 года, ~11 кварталов."""
    return pd.Series(pd.date_range(start, periods=n, freq=freq))


def _latent(n: int, rng: np.random.Generator, signal: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    """Вероятность дефолта по скрытому признаку: p = sigmoid(signal*x + intercept).

    По построению p калибрована (mean(p) ≈ 0.2), AUC ≈ Φ(signal/√2):
    signal=1.2 → AUC ≈ 0.80, signal=0.3 → AUC ≈ 0.58.
    """
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(signal * x + _INTERCEPT)))
    return p, x


def good_model(n=6000, seed=0):
    """Хорошая модель: AUC ≈ 0.78, калибровка идеальна, стабильна по периодам. Дефектов нет.

    n=6000 и freq="2h" дают ~6 кварталов с ~1000 строк и ~200 дефолтов в каждом:
    разброс AUC между периодами < 0.05 (иначе на малых периодах шум выглядит как дрейф).
    """
    rng = np.random.default_rng(seed)
    p, _ = _latent(n, rng)
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"default": y, "score": p,
                         "income": rng.normal(50, 10, n), "date": _dates(n, freq="2h")})


def weak_model(n=2000, seed=1):
    """Слабая модель: AUC ≈ 0.61 (слабое, но корректное ранжирование, статус ok)."""
    rng = np.random.default_rng(seed)
    p, _ = _latent(n, rng, signal=0.4)
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"default": y, "score": p, "date": _dates(n)})


def drift(n=2000, seed=2):
    """Дрейф: во второй половине скор сдвинут (+0.25) и признак income сдвинут (+20).

    Ожидание: PSI предсказаний и PSI признаков → warning.
    """
    rng = np.random.default_rng(seed)
    p, _ = _latent(n, rng)
    y = (rng.random(n) < p).astype(int)
    late = np.arange(n) >= n // 2
    score = np.clip(p + 0.25 * late, 0.001, 0.999)
    income = rng.normal(50, 10, n)
    income[: n // 2] += 20
    return pd.DataFrame({"default": y, "score": score, "income": income, "date": _dates(n)})


def leakage(n=2000, seed=3):
    """Утечка: признак leaky полностью повторяет target.

    Ожидание: «Проверка на утечку» → warning, leaky в списке подозрительных.
    """
    rng = np.random.default_rng(seed)
    p, _ = _latent(n, rng)
    y = (rng.random(n) < p).astype(int)
    df = pd.DataFrame({"default": y, "score": p, "income": rng.normal(50, 10, n)})
    df["leaky"] = y  # утечка: признак недоступен на момент выдачи
    return df


def immature_target(n=2000, seed=4):
    """Недозревший target: в последних ~2 кварталах дефолты ещё не наступили.

    Ожидание: «Зрелость target» → warning (отношение доли дефолтов < 0.6).
    """
    rng = np.random.default_rng(seed)
    p, _ = _latent(n, rng)
    y = (rng.random(n) < p).astype(int)
    n_late = n // 6  # ≈ 2 квартала при freq="12h"
    y[-n_late:] = 0
    return pd.DataFrame({"default": y, "score": p, "date": _dates(n)})


def one_class(n=2000, seed=5):
    """Один класс: дефолтов нет.

    Ожидание: ROC-AUC, Gini, KS, PR-AUC и их интервалы → not_computable.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    score = np.clip(rng.random(n), 0.001, 0.999)
    return pd.DataFrame({"default": y, "score": score})


def few_defaults(n=5000, seed=6):
    """Малое число дефолтов (60): оценки неустойчивы.

    Ожидание: «Достаточность дефолтов» → warning, доверительные интервалы → warning (широкие).
    """
    rng = np.random.default_rng(seed)
    n_pos = 60
    y = np.r_[np.ones(n_pos, dtype=int), np.zeros(n - n_pos, dtype=int)]
    rng.shuffle(y)
    score = np.clip(0.05 + 0.8 * y + rng.normal(0, 0.1, n), 0.001, 0.999)
    return pd.DataFrame({"default": y, "score": score})


def prediction_drift(n=2000, seed=7):
    """Дрейф предсказаний без изменения качества: скор сдвинут на +0.35 во второй половине.

    Ожидание: PSI предсказаний → warning, ROC-AUC остаётся > 0.7 (качество не пострадало).
    """
    rng = np.random.default_rng(seed)
    p, _ = _latent(n, rng)
    y = (rng.random(n) < p).astype(int)
    late = np.arange(n) >= n // 2
    score = np.clip(p + 0.35 * late, 0.001, 0.999)
    return pd.DataFrame({"default": y, "score": score, "date": _dates(n)})
