"""Контроль временной утечки: признаки, доступные на момент выдачи кредита."""
from __future__ import annotations

import pandas as pd

# Известны заявителю/кредитору в момент выдачи.
AVAILABLE_AT_ORIGINATION = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "addr_state", "dti", "delinq_2yrs", "earliest_cr_line",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc",
]

# Появляются после выдачи: прямая или косвенная информация об исходе.
KNOWN_LEAKAGE = [
    "loan_status", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    "hardship_flag", "debt_settlement_flag", "settlement_status", "settlement_date",
    "settlement_amount", "settlement_percentage", "settlement_term",
    "pymnt_plan", "funded_amnt", "funded_amnt_inv",
]


def split_features(columns: list[str]) -> dict[str, list[str]]:
    """Делит столбцы на разрешённые, запрещённые и требующие ручной проверки."""
    allowed = [c for c in columns if c in AVAILABLE_AT_ORIGINATION]
    leaking = [c for c in columns if c in KNOWN_LEAKAGE]
    unknown = [c for c in columns if c not in AVAILABLE_AT_ORIGINATION and c not in KNOWN_LEAKAGE]
    return {"allowed": allowed, "leakage": leaking, "unknown": unknown}


def flag_suspicious(df: pd.DataFrame, features: list[str], target: str = "target",
                    auc_threshold: float = 0.85) -> pd.DataFrame:
    """Находит числовые признаки с подозрительно сильной одномерной связью с target.

    Это только сигнал для ручной проверки происхождения признака, а не доказательство утечки.
    """
    from sklearn.metrics import roc_auc_score

    rows = []
    y = df[target]
    if y.nunique() < 2:
        raise ValueError("Для расчёта AUC нужны оба класса target")
    for col in features:
        x = pd.to_numeric(df[col], errors="coerce")
        mask = x.notna()
        if mask.sum() < 50 or y[mask].nunique() < 2:
            continue
        auc = roc_auc_score(y[mask], x[mask])
        strength = max(auc, 1 - auc)
        rows.append({"feature": col, "univariate_auc": auc, "strength": strength,
                     "suspicious": strength >= auc_threshold})
    return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)
