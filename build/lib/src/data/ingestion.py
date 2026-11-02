"""Загрузка исходных данных. Исходный файл никогда не изменяется."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, usecols: list[str] | None = None, nrows: int | None = None,
             as_str: bool = False) -> pd.DataFrame:
    """Читает CSV (в том числе .gz/.zip). as_str=True - все значения как строки, без приведения типов."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return pd.read_csv(path, usecols=usecols, nrows=nrows, low_memory=False,
                       dtype=str if as_str else None)


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV -> Parquet (копия исходных данных)")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="data/raw/loans_raw.parquet")
    args = parser.parse_args()
    df = load_csv(args.csv, as_str=True)
    out = save_parquet(df, args.out)
    print(f"Строк: {len(df)}, столбцов: {df.shape[1]}, сохранено: {out}")


if __name__ == "__main__":
    main()
