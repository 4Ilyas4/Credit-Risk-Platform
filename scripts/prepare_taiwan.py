"""Второй датасет другой формы: Taiwan Credit Card Default (30 000 строк, без дат).

Проверяет контракт пакета валидации на данных, отличных от Lending Club:
- без столбца дат раздел «Во времени» (Stability) честно помечается «не проверено»,
  а не показывает пустые или выдуманные метрики;
- на малой выборке (1000 строк) доверительные интервалы шире и появляются
  предупреждения о достаточности наблюдений и дефолтов.

Скор строится как out-of-fold предсказания логистической регрессии
(cross_val_predict): это честная оценка на данных, где каждая строка
предсказана моделью, не видевшей её при обучении.
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
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.validation import run_full_validation
from src.validation.reporting.report import summarize

RAW = "data/raw/default_of_credit_card_clients.xls"
TARGET_COL = "default payment next month"
SEED = 42
# Признаки UCI Taiwan: лимит, демография, история платежей, суммы.
FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def load() -> pd.DataFrame:
    """Загружает Taiwan XLS, переименовывает target, приводит признаки к числовым."""
    df = pd.read_excel(RAW, header=1)
    df = df.rename(columns={TARGET_COL: "y_true"})
    df = df.drop(columns=["ID"], errors="ignore")
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    return df.dropna(subset=["y_true"]).reset_index(drop=True)


def make_scores(df: pd.DataFrame) -> np.ndarray:
    """Out-of-fold вероятности LogisticRegression: каждая строка не видела модель."""
    X = df[FEATURES]
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)),
    ])
    return cross_val_predict(model, X, df["y_true"], cv=5, method="predict_proba")[:, 1]


def run(df: pd.DataFrame, label: str, out_dir: Path) -> dict:
    """Прогон валидатора на наборе без дат. Сохраняет отчёт и сводку."""
    report = run_full_validation(df["y_true"], df["y_score"],
                                 X=df[FEATURES], n_boot=200, seed=SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.save(str(out_dir / f"report_{label}.html"))
    report.save(str(out_dir / f"report_{label}.json"))
    s = summarize(report)
    # ширина доверительного интервала AUC
    ci = report.get("Reliability", "Доверительный интервал: ROC-AUC")
    ci_width = None if ci.value is None else ci.details["ci_high"] - ci.details["ci_low"]
    sufficiency = next((m for m in report.sections.get("Data Quality", [])
                        if m.name in ("Достаточность наблюдений", "Достаточность дефолтов")), None)
    return {
        "label": label, "n": int(len(df)), "default_rate": float(df["y_true"].mean()),
        "auc": report.get("Model Performance", "ROC-AUC").value,
        "ci_width_auc": ci_width,
        "cards": s["cards"], "attention": len(s["attention"]), "not_checked": len(s["not_checked"]),
        "sufficiency": sufficiency.interpretation if sufficiency else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Taiwan Credit Card Default: прогон валидатора")
    parser.add_argument("--out", default="data/taiwan", help="каталог для отчётов")
    parser.add_argument("--n-small", type=int, default=300, help="размер малой выборки")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Загрузка Taiwan Credit Card Default...")
    df = load()
    df["y_score"] = make_scores(df)
    path = out_dir / "taiwan_out_of_fold.parquet"
    df.drop(columns=["y_score"]).assign(y_score=df["y_score"]).to_parquet(path, index=False)
    print(f"  строк: {len(df)}, доля дефолтов: {df['y_true'].mean():.3f}, сохранено: {path}")

    findings = []
    print("Прогон валидатора на полном наборе (без дат)...")
    full = run(df, "full", out_dir)
    findings.append(full)
    print(f"  n={full['n']}, AUC={full['auc']:.4f}, CI_width={full['ci_width_auc']:.4f}, "
          f"stability={full['cards']['stability']['status']}")

    print(f"Малая выборка ({args.n_small} строк)...")
    small = df.sample(n=args.n_small, random_state=SEED).reset_index(drop=True)
    s = run(small, "small", out_dir)
    findings.append(s)
    print(f"  n={s['n']}, AUC={s['auc']:.4f}, CI_width={s['ci_width_auc']:.4f}")
    if s["sufficiency"]:
        print(f"  sufficiency: {s['sufficiency'][:120]}")

    (out_dir / "findings.json").write_text(
        json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nГотово. Отчёты: {out_dir}/")


if __name__ == "__main__":
    main()