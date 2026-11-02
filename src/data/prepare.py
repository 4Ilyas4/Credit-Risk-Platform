"""Подготовка рабочего набора: только нужные столбцы, числовые типы, target. Результат - Parquet."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.data.cleaning import clean, target_summary
from src.data.ingestion import load_parquet, save_parquet
from src.data.leakage import AVAILABLE_AT_ORIGINATION

SERVICE_COLUMNS = ["id", "issue_d", "loan_status"]
NUMERIC_COLUMNS = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc",
]


def working_columns() -> list[str]:
    """Служебные столбцы и признаки, доступные на момент выдачи. Признаки утечки не читаются."""
    return SERVICE_COLUMNS + AVAILABLE_AT_ORIGINATION


def to_numeric(df: pd.DataFrame, columns: list[str] = NUMERIC_COLUMNS) -> pd.DataFrame:
    """Приводит строковые числовые столбцы к float32. Нечитаемые значения -> NaN."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


def prepare(raw_parquet: str | Path, out_path: str | Path) -> pd.DataFrame:
    """raw Parquet -> нужные столбцы -> target и дата -> числовые типы -> processed Parquet."""
    available = set(pq.read_schema(raw_parquet).names)
    raw = load_parquet(raw_parquet, columns=[c for c in working_columns() if c in available])
    cleaned = to_numeric(clean(raw))
    del raw
    save_parquet(cleaned, out_path)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовка рабочего набора (processed Parquet)")
    parser.add_argument("--raw", default="data/raw/loans_raw.parquet")
    parser.add_argument("--out", default="data/processed/loans_clean.parquet")
    args = parser.parse_args()
    df = prepare(args.raw, args.out)
    mem_gb = df.memory_usage(deep=True).sum() / 1e9
    print(f"Строк: {len(df)}, столбцов: {df.shape[1]}, память: {mem_gb:.2f} ГБ, "
          f"доля дефолтов: {df['target'].mean():.4f}, сохранено: {args.out}")
    print(target_summary(df).to_string())


if __name__ == "__main__":
    main()
