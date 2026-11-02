"""Временная согласованность и зрелость целевой переменной."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.result import MetricResult, NOT_COMPUTABLE, OK, WARNING


def check_dates(dates: pd.Series) -> MetricResult:
    parsed = pd.to_datetime(dates, errors="coerce")
    bad = float(parsed.isna().mean())
    if parsed.notna().sum() == 0:
        return MetricResult("Временная согласованность", None, NOT_COMPUTABLE, "Разбор столбца дат.",
                            "Ни одну дату не удалось разобрать.", "Формат дат должен распознаваться pandas.")
    future = int((parsed > pd.Timestamp.now()).sum())
    return MetricResult(
        "Временная согласованность", bad, WARNING if bad > 0 or future else OK,
        "Доля значений, которые не удалось разобрать как дату.",
        f"Диапазон дат {parsed.min().date()} - {parsed.max().date()}; не разобрано {bad:.2%}; "
        f"дат в будущем: {future}.",
        "Не проверяет, что дата - это именно дата выдачи, а не дата погашения или обновления.",
        {"min": parsed.min(), "max": parsed.max(), "n_future": future})


def check_maturity(dates: pd.Series, y: pd.Series, freq: str = "Q", window: int = 4,
                   drop_ratio: float = 0.6) -> MetricResult:
    """Признак незрелого target: доля дефолтов в последних периодах резко ниже, чем раньше.

    Кредиты свежих периодов ещё не успели дойти до дефолта, поэтому их доля дефолтов занижена.
    Это косвенный признак. Он не заменяет проверку статусов кредитов и сроков.
    """
    df = pd.DataFrame({"d": pd.to_datetime(dates, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if df.empty:
        return MetricResult("Зрелость target", None, NOT_COMPUTABLE, "Динамика доли дефолтов по периодам.",
                            "Нет строк с датой и target.", "")
    df["p"] = df["d"].dt.to_period(freq)
    by = df.groupby("p")["y"].agg(n="size", default_rate="mean")
    by = by[by["n"] >= 30]
    lim = ("Падение доли дефолтов может быть реальным улучшением портфеля, а не незрелостью. "
           "Различить их по одним только датам и исходам нельзя.")
    if len(by) < window + 1:
        return MetricResult("Зрелость target", None, NOT_COMPUTABLE,
                            "Сравнение доли дефолтов последнего периода с предыдущими.",
                            f"Нужно минимум {window + 1} периодов с ≥ 30 наблюдениями, есть {len(by)}.", lim,
                            {"table": by.reset_index().astype({"p": str}).to_dict("records")})
    last = float(by["default_rate"].iloc[-1])
    ref = float(by["default_rate"].iloc[-window - 1:-1].median())
    ratio = last / ref if ref > 0 else float("nan")
    flag = bool(ref > 0 and ratio < drop_ratio)
    return MetricResult(
        "Зрелость target", ratio, WARNING if flag else OK,
        f"Доля дефолтов в последнем периоде ({by.index[-1]}) относительно медианы {window} предыдущих.",
        (f"Отношение {ratio:.2f} (последний период {last:.3f}, медиана предыдущих {ref:.3f}). "
         + ("Свежие периоды, вероятно, незрелые: дефолты ещё не успели наступить. Исключите их из оценки качества."
            if flag else "Признаков незрелости последнего периода нет.")),
        lim + f" Порог {drop_ratio} - рабочий ориентир.",
        {"table": by.reset_index().astype({"p": str}).to_dict("records")})
