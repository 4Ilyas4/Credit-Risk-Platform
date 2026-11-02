"""Дрейф данных и предсказаний: PSI по признакам и скору, изменение связи с target."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.result import MetricNotComputable
from src.validation.stability.psi import categorical_psi, numeric_psi, psi_level


def feature_psi_table(reference: pd.DataFrame, current: pd.DataFrame, features: list[str] | None = None,
                      n_bins: int = 10) -> pd.DataFrame:
    """PSI по каждому общему признаку. Признаки, для которых PSI не определён, получают причину."""
    common = [c for c in reference.columns if c in current.columns]
    cols = [c for c in (features or common) if c in common]
    rows = []
    for c in cols:
        try:
            if pd.api.types.is_numeric_dtype(reference[c]) and pd.api.types.is_numeric_dtype(current[c]):
                r = numeric_psi(reference[c], current[c], n_bins)
                rows.append({"feature": c, "type": "numeric", "psi": r["psi"], "level": psi_level(r["psi"]),
                             "missing_ref": r["missing_ref"], "missing_cur": r["missing_cur"], "note": ""})
            else:
                r = categorical_psi(reference[c], current[c])
                rows.append({"feature": c, "type": "categorical", "psi": r["psi"], "level": psi_level(r["psi"]),
                             "missing_ref": float(reference[c].isna().mean()),
                             "missing_cur": float(current[c].isna().mean()),
                             "note": f"новые категории: {r['new_categories'][:3]}" if r["new_categories"] else ""})
        except MetricNotComputable as exc:
            rows.append({"feature": c, "type": "n/a", "psi": np.nan, "level": "not_computable",
                         "missing_ref": np.nan, "missing_cur": np.nan, "note": str(exc)})
    cols_out = ["feature", "type", "psi", "level", "missing_ref", "missing_cur", "note"]
    if not rows:
        return pd.DataFrame(columns=cols_out)
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


def drift_types_note() -> str:
    """Что именно показывает каждый вид дрейфа. Нужно в отчёте, чтобы не путать понятия."""
    return ("Data drift - изменение распределения признаков. Prediction drift - изменение распределения скора. "
            "Concept drift - изменение связи признаков с исходом. Performance degradation - падение качества. "
            "Первые два видны без исходов, два последних требуют фактических исходов нового периода.")
