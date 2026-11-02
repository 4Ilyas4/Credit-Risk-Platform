"""Population Stability Index (PSI): сравнение распределения нового периода с референсным."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.result import MetricNotComputable

EPS = 1e-4  # доля, которой заменяется 0 в пустой корзине, чтобы не было log(0)


def _clean(x) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    return arr[np.isfinite(arr)]


def psi_from_proportions(expected: np.ndarray, actual: np.ndarray, eps: float = EPS) -> float:
    """PSI = sum (a_i - e_i) * ln(a_i / e_i). Нулевые доли заменяются на eps и нормируются."""
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def numeric_psi(reference, current, n_bins: int = 10, min_obs: int = 30) -> dict:
    """PSI числового признака. Границы корзин - квантили референса. Пропуски считаются отдельно."""
    ref_all = pd.to_numeric(pd.Series(reference), errors="coerce")
    cur_all = pd.to_numeric(pd.Series(current), errors="coerce")
    ref, cur = _clean(ref_all), _clean(cur_all)
    if len(ref) < min_obs or len(cur) < min_obs:
        raise MetricNotComputable(
            f"Недостаточно наблюдений для PSI (референс {len(ref)}, текущий {len(cur)}, нужно ≥ {min_obs}).")
    if np.unique(ref).size < 2:
        raise MetricNotComputable("Референсное распределение константно: корзины построить нельзя.")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
    ref_ids = np.searchsorted(edges, ref, side="left")
    cur_ids = np.searchsorted(edges, cur, side="left")
    k = len(edges) + 1
    e = np.bincount(ref_ids, minlength=k) / len(ref)
    a = np.bincount(cur_ids, minlength=k) / len(cur)
    value = psi_from_proportions(e, a)
    table = pd.DataFrame({"bin": range(k), "expected": e, "actual": a})
    return {"psi": value, "n_bins": k, "table": table,
            "missing_ref": float(ref_all.isna().mean()), "missing_cur": float(cur_all.isna().mean()),
            "n_ref": int(len(ref)), "n_cur": int(len(cur))}


def categorical_psi(reference, current, min_obs: int = 30) -> dict:
    """PSI категориального признака: корзина = значение. Новые значения тоже дают вклад."""
    ref = pd.Series(reference).dropna().astype(str)
    cur = pd.Series(current).dropna().astype(str)
    if len(ref) < min_obs or len(cur) < min_obs:
        raise MetricNotComputable(
            f"Недостаточно наблюдений для PSI (референс {len(ref)}, текущий {len(cur)}, нужно ≥ {min_obs}).")
    cats = sorted(set(ref) | set(cur))
    e = ref.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    a = cur.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    table = pd.DataFrame({"category": cats, "expected": e, "actual": a})
    return {"psi": psi_from_proportions(e, a), "n_bins": len(cats), "table": table,
            "new_categories": sorted(set(cur) - set(ref)), "n_ref": int(len(ref)), "n_cur": int(len(cur))}


def psi_level(value: float) -> str:
    """Общепринятые ориентиры: <0.1 стабильно, 0.1-0.25 умеренный сдвиг, >0.25 значительный."""
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "significant"
