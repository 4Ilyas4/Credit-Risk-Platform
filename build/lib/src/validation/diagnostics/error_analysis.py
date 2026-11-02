"""Анализ ошибок: FP/FN, сегменты, периоды, различия признаков между верными и ошибочными прогнозами."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.result import MetricNotComputable, MetricResult, NOT_COMPUTABLE, OK, WARNING


def error_table(y: np.ndarray, s: np.ndarray, threshold: float) -> pd.DataFrame:
    pred = s >= threshold
    return pd.DataFrame({"y": y, "pred": pred.astype(int),
                         "fp": pred & (y == 0), "fn": ~pred & (y == 1),
                         "correct": pred == (y == 1)})


def segment_errors(y: np.ndarray, s: np.ndarray, segments: pd.Series, threshold: float,
                   min_n: int = 100, max_segments: int = 30) -> pd.DataFrame:
    """FP/FN и разница прогноза с фактом по сегментам. Малые сегменты не показываются (шум)."""
    seg = pd.Series(segments).reset_index(drop=True).astype("string").fillna("missing")
    df = error_table(y, s, threshold)
    df["seg"] = seg.to_numpy()
    df["s"] = s
    g = df.groupby("seg").agg(n=("y", "size"), default_rate=("y", "mean"), mean_score=("s", "mean"),
                              fp_rate=("fp", "mean"), fn_rate=("fn", "mean"))
    g["calibration_gap"] = g["mean_score"] - g["default_rate"]
    g = g[g["n"] >= min_n].sort_values("n", ascending=False).head(max_segments)
    return g.reset_index()


def feature_shift_errors(X: pd.DataFrame, y: np.ndarray, s: np.ndarray, threshold: float,
                         top_k: int = 10) -> pd.DataFrame:
    """Стандартизованная разница средних числовых признаков: ошибочные против верных прогнозов."""
    err = error_table(y, s, threshold)
    wrong = ~err["correct"].to_numpy()
    if wrong.sum() < 30 or (~wrong).sum() < 30:
        raise MetricNotComputable("Слишком мало верных или ошибочных прогнозов для сравнения признаков.")
    rows = []
    for c in X.select_dtypes("number").columns:
        v = X[c].to_numpy(float)
        a, b = v[wrong], v[~wrong]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 30 or len(b) < 30:
            continue
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        if sd > 0:
            rows.append({"feature": c, "std_diff": float((a.mean() - b.mean()) / sd)})
    out = pd.DataFrame(rows, columns=["feature", "std_diff"])
    return out.reindex(out["std_diff"].abs().sort_values(ascending=False).index).head(top_k).reset_index(drop=True)


def error_results(y: np.ndarray, s: np.ndarray, threshold: float | None = None, dates=None, groups=None,
                  X: pd.DataFrame | None = None) -> list[MetricResult]:
    thr = float(np.mean(y)) if threshold is None else float(threshold)
    lim_thr = (f"Порог {thr:.3f} {'задан пользователем' if threshold is not None else 'равен доле дефолтов'}. "
               "Структура ошибок целиком зависит от порога, а бизнес-порог неизвестен.")
    e = error_table(y, s, thr)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    out = []
    if n_pos == 0 or n_neg == 0:
        return [MetricResult("Анализ ошибок", None, NOT_COMPUTABLE, "FP и FN по порогу.",
                             "В выборке один класс: доли FP и FN не определены.", lim_thr)]
    fpr, fnr = float(e["fp"].sum() / n_neg), float(e["fn"].sum() / n_pos)
    out.append(MetricResult(
        "False Positive / False Negative", fnr, OK,
        "FPR - доля недефолтов, отнесённых к дефолтам. FNR - доля пропущенных дефолтов (значение метрики).",
        f"FP={int(e['fp'].sum())} (FPR={fpr:.3f}), FN={int(e['fn'].sum())} (FNR={fnr:.3f}).", lim_thr,
        {"fpr": fpr, "fnr": fnr, "threshold": thr}))
    if groups is not None:
        t = segment_errors(y, s, groups, thr)
        worst = float(t["calibration_gap"].abs().max()) if len(t) else None
        out.append(MetricResult(
            "Ошибки по сегментам", worst, WARNING if worst is not None and worst > 0.05 else OK,
            "Для каждого сегмента: доли FP, FN и разница среднего прогноза с фактической долей дефолтов. "
            "Значение - наибольшая по модулю разница.",
            ("Есть сегменты со смещённым прогнозом." if worst is not None and worst > 0.05
             else "Заметных смещений по сегментам нет.") + f" Сегментов в таблице: {len(t)}.",
            lim_thr + " Сегменты меньше 100 наблюдений не показываются.", {"table": t}))
    if dates is not None:
        p = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce").dt.to_period("Q").astype(str)
        t = segment_errors(y, s, p, thr, min_n=50)
        out.append(MetricResult(
            "Ошибки по периодам", None, OK, "FP, FN и калибровочный разрыв по кварталам.",
            f"Периодов в таблице: {len(t)}.", lim_thr, {"table": t.sort_values("seg")}))
    if X is not None:
        try:
            t = feature_shift_errors(X.reset_index(drop=True), y, s, thr)
            out.append(MetricResult(
                "Признаки: верные и ошибочные прогнозы", float(t["std_diff"].abs().max()) if len(t) else None, OK,
                "Стандартизованная разница средних признака между ошибочными и верными прогнозами.",
                "Большие значения указывают, где модель ошибается чаще.",
                lim_thr + " Это описательная статистика: она не объясняет внутреннюю логику модели.", {"table": t}))
        except MetricNotComputable as exc:
            out.append(MetricResult("Признаки: верные и ошибочные прогнозы", None, NOT_COMPUTABLE, "", str(exc), lim_thr))
    return out
