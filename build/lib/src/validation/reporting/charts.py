"""Графики отчёта (matplotlib). Каждая функция возвращает PNG в байтах или None, если данных нет."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def roc_chart(y: np.ndarray, s: np.ndarray) -> bytes | None:
    """ROC-кривая. Строится только если в данных есть оба класса."""
    if len(np.unique(y)) < 2:
        return None
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    distinct = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tps = np.cumsum(ys)[distinct]
    fps = (1 + distinct) - tps
    tpr, fpr = np.r_[0, tps / ys.sum()], np.r_[0, fps / (len(ys) - ys.sum())]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot(fpr, tpr, lw=2, label="Модель")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Случайно")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC-кривая")
    ax.legend(loc="lower right")
    return _png(fig)


def calibration_chart(table: list[dict]) -> bytes | None:
    if not table:
        return None
    t = pd.DataFrame(table)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    hi = float(max(t["predicted"].max(), t["observed"].max())) * 1.05
    ax.plot([0, hi], [0, hi], "--", color="grey", label="Идеальная калибровка")
    ax.plot(t["predicted"], t["observed"], "o-", label="Модель")
    ax.set(xlabel="Средний прогноз", ylabel="Фактическая доля дефолтов", title="Калибровка")
    ax.legend(loc="upper left")
    return _png(fig)


def score_distribution_chart(y: np.ndarray, s: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bins = np.linspace(0, float(max(s.max(), 1e-6)), 40)
    ax.hist(s[y == 0], bins=bins, alpha=0.6, density=True, label="Нет дефолта")
    if (y == 1).any():
        ax.hist(s[y == 1], bins=bins, alpha=0.6, density=True, label="Дефолт")
    ax.set(xlabel="Предсказанная вероятность", ylabel="Плотность", title="Распределение скора")
    ax.legend()
    return _png(fig)


def backtest_chart(table: list[dict]) -> bytes | None:
    t = pd.DataFrame(table)
    t = t[t["sufficient"]] if len(t) else t
    if len(t) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(t["period"], t["auc"], "o-", label="ROC-AUC")
    ax.plot(t["period"], t["ks"], "s-", label="KS")
    ax.set(xlabel="Период", ylabel="Значение", title="Качество по периодам")
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    return _png(fig)


def psi_chart(table: list[dict]) -> bytes | None:
    t = pd.DataFrame(table)
    if t.empty or "psi" not in t:
        return None
    t = t.dropna(subset=["psi"]).head(15)
    if t.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 0.35 * len(t) + 1.2))
    ax.barh(t["feature"][::-1], t["psi"][::-1])
    ax.axvline(0.1, color="orange", ls="--", lw=1)
    ax.axvline(0.25, color="red", ls="--", lw=1)
    ax.set(xlabel="PSI", title="PSI признаков (0.1 и 0.25 - ориентиры)")
    return _png(fig)


def charts_for(report) -> dict[str, list[tuple[str, bytes]]]:
    """Графики по разделам: {раздел: [(подпись, png), ...]}."""
    out: dict[str, list[tuple[str, bytes]]] = {}

    def add(section: str, caption: str, png: bytes | None) -> None:
        if png:
            out.setdefault(section, []).append((caption, png))

    y, s = report.artifacts.get("y"), report.artifacts.get("s")
    if y is not None and s is not None:
        add("Model Performance", "ROC-кривая", roc_chart(y, s))
        add("Model Performance", "Распределение скора", score_distribution_chart(y, s))
    for sec, name, fn, caption in (
            ("Calibration", "Calibration Curve", calibration_chart, "Кривая калибровки"),
            ("Stability", "Backtesting по периодам", backtest_chart, "Динамика качества"),
            ("Stability", "PSI признаков", psi_chart, "PSI признаков")):
        for r in report.sections.get(sec, []):
            tbl = r.details.get("table")
            if isinstance(tbl, pd.DataFrame):
                tbl = tbl.to_dict("records")
            if r.name == name and tbl:
                add(sec, caption, fn(tbl))
    return out
