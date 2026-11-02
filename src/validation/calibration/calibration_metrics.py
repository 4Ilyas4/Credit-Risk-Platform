"""Калибровка: кривая, ECE, калибровка в целом. Собственная реализация на numpy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.result import MetricNotComputable, MetricResult, OK, WARNING, safe_result


def _bin_ids(s: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Номера корзин. Значение ровно на границе попадает в нижнюю корзину (как в scikit-learn).

    При совпадающих квантилях корзины не объединяются: пустые корзины просто не попадают в таблицу.
    """
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    elif strategy == "quantile":
        edges = np.quantile(s, np.linspace(0.0, 1.0, n_bins + 1))[1:-1]
    else:
        raise ValueError("strategy должен быть 'uniform' или 'quantile'")
    return np.searchsorted(edges, s, side="left")


def calibration_curve(y: np.ndarray, s: np.ndarray, n_bins: int = 10,
                      strategy: str = "quantile") -> pd.DataFrame:
    """Таблица по корзинам: число наблюдений, средний прогноз, фактическая доля дефолтов."""
    if len(y) == 0:
        raise MetricNotComputable("Нет наблюдений.")
    ids = _bin_ids(s, n_bins, strategy)
    df = pd.DataFrame({"bin": ids, "y": y, "s": s})
    g = df.groupby("bin").agg(n=("y", "size"), predicted=("s", "mean"), observed=("y", "mean"))
    return g.reset_index()


def expected_calibration_error(y: np.ndarray, s: np.ndarray, n_bins: int = 10,
                               strategy: str = "uniform") -> float:
    """ECE = sum_k (n_k / n) * |observed_k - predicted_k|."""
    curve = calibration_curve(y, s, n_bins, strategy)
    return float(np.sum(curve["n"] / curve["n"].sum() * np.abs(curve["observed"] - curve["predicted"])))


def calibration_in_the_large(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    mean_pred, mean_obs = float(np.mean(s)), float(np.mean(y))
    return {"mean_predicted": mean_pred, "observed_rate": mean_obs, "difference": mean_pred - mean_obs,
            "ratio": mean_pred / mean_obs if mean_obs > 0 else float("nan")}


def calibration_results(y: np.ndarray, s: np.ndarray, n_bins: int = 10) -> list[MetricResult]:
    limits_bins = (f"Зависит от числа корзин ({n_bins}) и способа разбиения. "
                   "В корзинах с малым числом наблюдений оценка шумная.")
    out: list[MetricResult] = []
    citl = calibration_in_the_large(y, s)

    def interp_citl(_):
        d = citl["difference"]
        text = (f"Средний прогноз {citl['mean_predicted']:.4f}, фактическая доля дефолтов "
                f"{citl['observed_rate']:.4f} (разница {d:+.4f}).")
        if abs(d) > 0.02 and abs(d) / max(citl["observed_rate"], 1e-9) > 0.1:
            return text + (" Модель занижает риск." if d < 0 else " Модель завышает риск."), WARNING
        return text, OK

    out.append(safe_result(
        "Calibration in the large", lambda: citl["difference"],
        "Разница между средним прогнозом и фактической долей дефолтов.", interp_citl,
        "Не показывает, где именно прогноз ошибается по шкале риска. Порог предупреждения: |разница| > 0.02 "
        "и больше 10% от фактической доли - рабочий ориентир.", citl))
    ece = lambda: expected_calibration_error(y, s, n_bins, "uniform")  # noqa: E731
    out.append(safe_result(
        "ECE", ece, "Среднее взвешенное расхождение между прогнозом и фактом по корзинам вероятности.",
        lambda v: (f"ECE={v:.4f}: в среднем прогноз отличается от факта на {v * 100:.1f} п.п.",
                   WARNING if v > 0.05 else OK),
        limits_bins + " Порог 0.05 - рабочий ориентир."))
    try:
        curve = calibration_curve(y, s, n_bins, "quantile")
        out.append(MetricResult(
            "Calibration Curve", None, OK,
            "Средний прогноз и фактическая доля дефолтов по равным по размеру корзинам прогноза.",
            "Точки на диагонали означают хорошую калибровку.", limits_bins,
            {"table": curve.to_dict("records")}))
    except MetricNotComputable as exc:
        out.append(MetricResult("Calibration Curve", None, "not_computable", "Кривая калибровки.",
                                str(exc), limits_bins))
    return out
