"""Демонстрационный кейс Lending Club 2007-2011: подготовка данных и прогон валидатора.

Что делает:
1. Загружает сырые данные LC, фильтрует 2007-2011 (≈42 тыс. строк).
2. Бинаризует loan_status: Fully Paid = 0, Charged Off/Default = 1, остальные статусы
   (Current, Late, In Grace Period) исключаются как незрелые.
3. Строит два варианта скора:
   A. LogisticRegression, обученная на 2007-2009, применённая к 2010-2011 (честная проверка «во времени»).
   B. int_rate как ранговый скор (проверяет score_kind='score' и higher_is_riskier).
4. Один прогон с намеренно включёнными столбцами-утечками (total_pymnt, recoveries).
5. Запускает валидатор на каждом варианте и сохраняет HTML/JSON отчёты.

Названия столбцов Lending Club допустимы только в этом скрипте, не в ядре.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Корень проекта в sys.path: скрипт запускается из scripts/ напрямую.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.validation import run_full_validation, validate_dataset

# --- LC-специфичные константы (только здесь, не в ядре) ---
RAW = "data/raw/loans_raw.parquet"
GOOD_STATUSES = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
BAD_STATUSES = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
TRAIN_END = pd.Timestamp("2009-12-31")  # обучение на ранних годах, проверка на поздних
SEED = 42

# Признаки, доступные на момент выдачи (числовые): честная проверка «во времени».
NUMERIC_FEATURES = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc",
]
# Признаки-утечки: появляются после выдачи и раскрывают исход.
LEAKAGE_FEATURES = ["total_pymnt", "recoveries"]


def load_base() -> pd.DataFrame:
    """Загружает сырые данные, фильтрует 2007-2011, бинаризует loan_status.

    Числовые признаки приводятся к float: в сыром CSV LC все значения — строки,
    а валидатор требует числовые признаки для проверки утечки и PSI.
    """
    df = pd.read_parquet(RAW)
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df[df["issue_d"].between("2007-01-01", "2011-12-31")].copy()
    status = df["loan_status"].astype("string").str.strip()
    y = pd.Series(np.nan, index=df.index)
    y[status.isin(GOOD_STATUSES)] = 0
    y[status.isin(BAD_STATUSES)] = 1
    df = df[y.notna()].copy()
    df["y_true"] = y[df.index].astype(int)
    df = to_numeric(df, NUMERIC_FEATURES + LEAKAGE_FEATURES)
    return df.reset_index(drop=True)


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Приводит строковые числовые столбцы LC к float. Нечитаемое -> NaN."""
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def make_lr_scores(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Логистическая регрессия на 2007-2009, предсказание вероятности на всех строках."""
    train = df["issue_d"] <= TRAIN_END
    X = to_numeric(df, features)[features]
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, C=1.0, random_state=SEED)),
    ])
    model.fit(X[train], df.loc[train, "y_true"])
    return model.predict_proba(X)[:, 1]


def build_variants(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Три варианта набора: LR-скор, int_rate-скор, LR-скор с утечкой."""
    score_lr = make_lr_scores(df, NUMERIC_FEATURES)
    out = {}
    # A. Логистическая регрессия (честная проверка «во времени»)
    a = df[["id", "issue_d", "grade", "y_true"]].copy()
    a["y_score"] = score_lr
    a["score_kind"] = "probability"
    a["higher_is_riskier"] = True
    for f in NUMERIC_FEATURES:
        a[f] = df[f].to_numpy()
    out["lr"] = a
    # B. int_rate как ранговый скор (score_kind='score')
    b = df[["id", "issue_d", "grade", "y_true"]].copy()
    b["y_score"] = pd.to_numeric(df["int_rate"], errors="coerce")
    b["score_kind"] = "score"
    b["higher_is_riskier"] = True  # выше ставка = выше риск
    for f in NUMERIC_FEATURES:
        b[f] = df[f].to_numpy()
    out["intrate"] = b
    # C. Утечка: те же признаки + total_pymnt, recoveries
    c = a.copy()
    for f in LEAKAGE_FEATURES:
        c[f] = df[f].to_numpy()
    out["leakage"] = c
    return out


def maturity_demo(out_dir: Path) -> dict:
    """Демонстрация проверки зрелости target на полном диапазоне 2007-2018.

    В 2007-2011 все кредиты давно погашены (доля Current = 0), поэтому признаков
    незрелости нет. Начиная с 2016 доля статуса Current растёт (32% в 2016, 62% в 2017,
    ещё выше в 2018): свежие кредиты не созрели, и проверка «Зрелость target»
    предупреждает (доля дефолтов последнего квартала падает ниже 60% от предыдущих).
    """
    df = pd.read_parquet(RAW, columns=["loan_status", "issue_d"])
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df[df["issue_d"].between("2007-01-01", "2018-12-31")].copy()
    status = df["loan_status"].astype("string").str.strip()
    y = pd.Series(np.nan, index=df.index)
    y[status.isin(GOOD_STATUSES)] = 0
    y[status.isin(BAD_STATUSES)] = 1
    df["y_true"] = y.astype(float)
    df["current"] = status.isin({"Current", "In Grace Period",
                                 "Late (16-30 days)", "Late (31-120 days)"}).astype(int)
    by_year = df.groupby(df["issue_d"].dt.year).agg(
        current_share=("current", "mean"), default_rate=("y_true", "mean"))
    maturity = validate_dataset(df, target="y_true", date_col="issue_d")
    m = maturity.get("Data Quality", "Зрелость target")
    return {"status": m.status, "interpretation": m.interpretation,
            "by_year": by_year.to_dict("index")}


def run_validation(df: pd.DataFrame, label: str, out_dir: Path) -> dict:
    """Прогон валидатора на поздних годах (2010-2011) с референсом из ранних (2007-2009).

    Возвращает сводку находок для README.
    """
    train = df["issue_d"] <= TRAIN_END
    cur = df[~train].copy()
    features = [c for c in df.columns if c not in
                ("id", "issue_d", "grade", "y_true", "y_score", "score_kind", "higher_is_riskier")]
    # референс: y_score + признаки ранних лет
    ref = pd.DataFrame({"y_score": df.loc[train, "y_score"].to_numpy()})
    for f in features:
        ref[f] = df.loc[train, f].to_numpy()
    score_kind = str(df["score_kind"].iloc[0])
    higher_is_riskier = bool(df["higher_is_riskier"].iloc[0])
    report = run_full_validation(
        cur["y_true"], cur["y_score"], X=cur[features], dates=cur["issue_d"],
        reference_data=ref, n_boot=200, seed=SEED,
        score_kind=score_kind, higher_is_riskier=higher_is_riskier)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.save(str(out_dir / f"report_{label}.html"))
    report.save(str(out_dir / f"report_{label}.json"))
    return {
        "label": label, "score_kind": score_kind,
        "n_used": int(len(cur)), "default_rate": float(cur["y_true"].mean()),
        "auc": report.get("Model Performance", "ROC-AUC").value,
        "warnings": report.warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lending Club 2007-2011: подготовка и валидация")
    parser.add_argument("--out", default="data/lending_club", help="каталог для файлов и отчётов")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Загрузка и подготовка данных 2007-2011...")
    df = load_base()
    print(f"  строк: {len(df)}, доля дефолтов: {df['y_true'].mean():.3f}")
    by_year = df.groupby(df["issue_d"].dt.year)["y_true"].agg(["size", "mean"])
    print(f"  по годам: {by_year.to_dict('index')}")

    print("Построение вариантов скора...")
    variants = build_variants(df)
    for label, v in variants.items():
        path = out_dir / f"lc_{label}.parquet"
        v.drop(columns=["score_kind", "higher_is_riskier"]).to_parquet(path, index=False)
        print(f"  {label}: {path} ({len(v)} строк)")

    print("Прогон валидатора...")
    findings = []
    for label in ("lr", "intrate", "leakage"):
        print(f"  [{label}] ...")
        info = run_validation(variants[label], label, out_dir)
        findings.append(info)
        print(f"    AUC={info['auc']:.4f}, n_used={info['n_used']}, "
              f"warnings={len(info['warnings'])}")

    print("Проверка зрелости target (2007-2018)...")
    maturity = maturity_demo(out_dir)
    print(f"    {maturity['status']}: {maturity['interpretation'][:160]}")

    findings.append({"label": "maturity_demo", **maturity})
    (out_dir / "findings.json").write_text(
        json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nГотово. Отчёты и файлы: {out_dir}/")


if __name__ == "__main__":
    main()