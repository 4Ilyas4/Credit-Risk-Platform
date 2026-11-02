"""Plotly-графики отчёта: одна реализация для HTML-отчёта и Streamlit-интерфейса.

Каждая функция возвращает plotly.graph_objects.Figure. charts_payload(report)
собирает подписи и JSON графиков для хранения в результате API.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import plotly.graph_objects as go

# Общие стили: единый вид в HTML и UI.
_BG = "#ffffff"
_FONT = "Segoe UI, Arial, sans-serif"
_C0, _C1 = "#1f77b4", "#d62728"
_GREY = "#9e9e9e"


def _layout(title: str, xlabel: str, ylabel: str, height: int = 380) -> go.Layout:
    return go.Layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "y": 0.99, "yanchor": "top"},
        xaxis={"title": xlabel, "gridcolor": "#eee"},
        yaxis={"title": ylabel, "gridcolor": "#eee"},
        height=height, margin={"l": 60, "r": 20, "t": 56, "b": 64},
        paper_bgcolor=_BG, plot_bgcolor=_BG, font={"family": _FONT, "size": 12},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18, "x": 0.02})


def score_distribution(y: np.ndarray, s: np.ndarray, score_kind: str = "probability") -> go.Figure:
    """Распределение скора отдельно для дефолтов и недефолтов."""
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    good, bad = s[y == 0], s[y == 1]
    lo, hi = (0.0, 1.0) if score_kind == "probability" else (float(np.nanmin(s)), float(np.nanmax(s)))
    if np.isnan(lo) or lo == hi:
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
    hist = go.Histogram
    fig = go.Figure(layout=_layout("Распределение скора по классам", "Скор", "Число наблюдений"))
    fig.add_trace(hist(x=good, name="Нет дефолта", opacity=0.55, marker_color=_C0,
                       xbins={"size": (hi - lo) / 40 or 1}))
    fig.add_trace(hist(x=bad, name="Дефолт", opacity=0.6, marker_color=_C1,
                       xbins={"size": (hi - lo) / 40 or 1}))
    return fig


def calibration_chart(table: list[dict]) -> go.Figure:
    """Калибровка по децилям: средний прогноз против фактической доли дефолтов."""
    t = pd.DataFrame(table)
    fig = go.Figure(layout=_layout("Калибровка по децилям", "Средний прогноз", "Фактическая доля дефолтов"))
    hi = float(max(t["predicted"].max(), t["observed"].max())) * 1.05
    fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", name="Идеальная",
                             line={"dash": "dash", "color": _GREY}))
    fig.add_trace(go.Scatter(x=t["predicted"], y=t["observed"], mode="lines+markers",
                             name="Модель", marker_color=_C0, line={"color": _C0}))
    return fig


def backtest_chart(table: list[dict]) -> go.Figure:
    """Метрики качества по периодам: AUC, Gini, KS."""
    t = pd.DataFrame(table)
    fig = go.Figure(layout=_layout("Метрики качества по периодам", "Период", "Значение"))
    for col, color, name in (("auc", _C0, "ROC-AUC"), ("gini", "#2ca02c", "Gini"),
                             ("ks", _C1, "KS")):
        if col in t and t[col].notna().any():
            fig.add_trace(go.Scatter(x=t["period"], y=t[col], mode="lines+markers",
                                     name=name, marker_color=color, line={"color": color}))
    return fig


def psi_chart(table: list[dict]) -> go.Figure:
    """PSI предсказаний: доли референса и текущего периода по корзинам."""
    t = pd.DataFrame(table)
    fig = go.Figure(layout=_layout("PSI предсказаний: распределения", "Корзина", "Доля"))
    fig.add_trace(go.Bar(x=t["bin"], y=t["expected"], name="Референс", opacity=0.55, marker_color=_C0))
    fig.add_trace(go.Bar(x=t["bin"], y=t["actual"], name="Текущий период", opacity=0.75, marker_color=_C1))
    return fig


def maturity_chart(table: list[dict]) -> go.Figure:
    """Зрелость target: доля дефолтов по периодам."""
    t = pd.DataFrame(table)
    fig = go.Figure(layout=_layout("Доля дефолтов по периодам (зрелость target)", "Период", "Доля дефолтов"))
    fig.add_trace(go.Scatter(x=t["period"], y=t["default_rate"], mode="lines+markers",
                             name="Доля дефолтов", marker_color=_C0, line={"color": _C0}))
    return fig


def charts_figures(report) -> dict[str, list[tuple[str, go.Figure]]]:
    """Фигуры по разделам: {section: [(caption, go.Figure), ...]}. Общая для HTML и UI."""
    out: dict[str, list[tuple[str, go.Figure]]] = {}

    def add(section: str, caption: str, fig: go.Figure | None) -> None:
        if fig is not None:
            out.setdefault(section, []).append((caption, fig))

    y, s = report.artifacts.get("y"), report.artifacts.get("s")
    if y is not None and s is not None:
        add("Model Performance", "Распределение скора",
            score_distribution(y, s, report.meta.get("score_kind", "probability")))

    for sec, res_name, fn, caption in (
            ("Calibration", "Calibration Curve", calibration_chart, "Калибровка по децилям"),
            ("Stability", "Backtesting по периодам", backtest_chart, "Метрики по периодам"),
            ("Stability", "PSI предсказаний", psi_chart, "PSI предсказаний"),
            ("Stability", "Backtesting по периодам", maturity_chart, "Доля дефолтов по периодам")):
        for r in report.sections.get(sec, []):
            tbl = r.details.get("table") if isinstance(r.details, dict) else None
            if r.name == res_name and tbl is not None:
                records = tbl.to_dict("records") if isinstance(tbl, pd.DataFrame) else tbl
                if records:
                    add(sec, caption, fn(records))
    return out


def charts_payload(report) -> dict[str, list[dict]]:
    """Фигуры как JSON для API-результата: {section: [{caption, fig_json}]}.

    fig_json — строка plotly JSON (from_json в Streamlit).
    """
    return {sec: [{"caption": cap, "fig_json": fig.to_json()} for cap, fig in items]
            for sec, items in charts_figures(report).items()}