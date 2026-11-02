"""Признаки и предобработка. Всё, что оценивается по данным, обучается только на train."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.leakage import split_features

NUMERIC = [
    "loan_amnt", "int_rate", "installment", "log_annual_inc", "dti", "delinq_2yrs",
    "fico_range_low", "inq_last_6mths", "open_acc", "pub_rec", "log_revol_bal",
    "revol_util", "total_acc", "credit_history_years",
]
CATEGORICAL = [
    "term", "grade", "sub_grade", "emp_length", "home_ownership",
    "verification_status", "purpose", "addr_state",
]
# Оценки собственной модели Lending Club. Формально известны при выдаче, но по сути чужой скор.
LC_SCORE_FEATURES = ["int_rate", "grade", "sub_grade"]
# fico_range_high = fico_range_low + 4 в этих данных: берём только low, чтобы не дублировать признак.
RAW_REQUIRED = [
    "issue_d", "earliest_cr_line", "annual_inc", "revol_bal", "fico_range_low",
    "loan_amnt", "int_rate", "installment", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_util", "total_acc",
] + CATEGORICAL


def feature_lists(exclude_lc_score: bool = False) -> tuple[list[str], list[str]]:
    if not exclude_lc_score:
        return list(NUMERIC), list(CATEGORICAL)
    return ([c for c in NUMERIC if c not in LC_SCORE_FEATURES],
            [c for c in CATEGORICAL if c not in LC_SCORE_FEATURES])


def assert_no_leakage(columns: list[str]) -> None:
    """Защита: в модель не должны попадать признаки утечки."""
    leaking = split_features(list(columns))["leakage"]
    if leaking:
        raise ValueError(f"Признаки утечки в наборе признаков: {leaking}")


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Строит признаки модели из сырых столбцов. Без параметров, оцениваемых по данным."""
    missing = [c for c in RAW_REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"Нет столбцов: {missing}")
    out = df[RAW_REQUIRED].copy()
    earliest = pd.to_datetime(out["earliest_cr_line"], format="%b-%Y", errors="coerce")
    out["credit_history_years"] = ((out["issue_d"] - earliest).dt.days / 365.25).astype("float32")
    out["log_annual_inc"] = np.log1p(out["annual_inc"].clip(lower=0)).astype("float32")
    out["log_revol_bal"] = np.log1p(out["revol_bal"].clip(lower=0)).astype("float32")
    out["emp_length"] = out["emp_length"].fillna("missing")
    cols = NUMERIC + CATEGORICAL
    assert_no_leakage(cols)
    return out[cols]


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Обрезает выбросы по квантилям, найденным на обучающей части."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(arr, self.lower, axis=0)
        self.high_ = np.nanquantile(arr, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(numeric: list[str], categorical: list[str], dense: bool = False) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("clip", QuantileClipper()),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                                     sparse_output=not dense)
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        sparse_threshold=0.0 if dense else 0.3)
