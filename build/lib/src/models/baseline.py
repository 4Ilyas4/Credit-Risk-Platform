"""Эталонная модель: Logistic Regression с временной валидацией.

Модель нужна как baseline для сравнения и как источник out-of-time предсказаний для валидатора (A3).
Метрики здесь из scikit-learn: собственные реализации появятся в пакете валидации.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

from src.models.features import add_derived, build_preprocessor, feature_lists
from src.models.split import SplitConfig, assign_split, expanding_window_folds

SEED = 42


def make_model(numeric: list[str], categorical: list[str], C: float = 1.0) -> Pipeline:
    """Без class_weight: иначе вероятности смещаются и калибровку нельзя оценивать честно."""
    return Pipeline([
        ("prep", build_preprocessor(numeric, categorical)),
        ("clf", LogisticRegression(C=C, max_iter=500, solver="lbfgs")),
    ])


def score_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """AUC, Gini, KS, Brier, LogLoss. Если в y_true один класс, ранговые метрики не считаются."""
    y_true = np.asarray(y_true)
    out = {"n": int(len(y_true)), "default_rate": float(y_true.mean()),
           "brier": float(brier_score_loss(y_true, y_score)),
           "log_loss": float(log_loss(y_true, np.clip(y_score, 1e-15, 1 - 1e-15), labels=[0, 1]))}
    if len(np.unique(y_true)) < 2:
        out.update(auc=float("nan"), gini=float("nan"), ks=float("nan"))
        return out
    auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    out.update(auc=auc, gini=2 * auc - 1, ks=float(np.max(tpr - fpr)))
    return out


def expanding_window_cv(X: pd.DataFrame, y: pd.Series, dates: pd.Series, numeric, categorical,
                        C: float, valid_years: list[int]) -> pd.DataFrame:
    """Расширяющееся окно внутри переданных данных. Возвращает метрики по каждому году проверки."""
    rows = []
    for (train, valid), year in zip(expanding_window_folds(dates, valid_years), sorted(valid_years)):
        model = make_model(numeric, categorical, C)
        model.fit(X[train], y[train])
        pred = model.predict_proba(X[valid])[:, 1]
        rows.append({"valid_year": year, "n_train": int(train.sum()), **score_metrics(y[valid], pred)})
    return pd.DataFrame(rows)


def prior_rate_metrics(y_train: pd.Series, y_eval: pd.Series) -> dict[str, float]:
    """Тривиальный ориентир: всем одна и та же вероятность (доля дефолтов в train)."""
    return score_metrics(y_eval.to_numpy(), np.full(len(y_eval), float(y_train.mean())))


def _variant(name: str, exclude: bool, X_all: pd.DataFrame, y: pd.Series, dates: pd.Series,
             masks: dict[str, np.ndarray], c_grid: tuple[float, ...], cv_years: list[int]):
    """Один вариант набора признаков. Возвращает (метрики, предсказания по valid/test/late)."""
    numeric, categorical = feature_lists(exclude_lc_score=exclude)
    Xv = X_all[numeric + categorical]
    fit, valid, test, late = masks["train"], masks["valid"], masks["test"], masks["late"]
    t0 = time.time()
    # 1. Выбор C: expanding-window только внутри train, valid и test не участвуют.
    cv = {C: expanding_window_cv(Xv[fit], y[fit], dates[fit], numeric, categorical, C, cv_years)
          for C in c_grid}
    mean_auc = {C: float(f["auc"].mean()) for C, f in cv.items()}
    best_c = max(mean_auc, key=mean_auc.get)
    # 2. Обучение на train, проверка на valid.
    model = make_model(numeric, categorical, best_c).fit(Xv[fit], y[fit])
    preds = {"valid": model.predict_proba(Xv[valid])[:, 1]}
    # 3. Итоговая модель на train + valid. Test используется один раз.
    final = make_model(numeric, categorical, best_c).fit(Xv[fit | valid], y[fit | valid])
    preds["test"] = final.predict_proba(Xv[test])[:, 1]
    preds["late"] = final.predict_proba(Xv[late])[:, 1]
    info = {
        "features": numeric + categorical, "best_C": best_c,
        "cv_auc_by_C": {str(k): round(v, 4) for k, v in mean_auc.items()},
        "cv_folds_best_C": cv[best_c].round(4).to_dict("records"),
        "valid": score_metrics(y[valid], preds["valid"]),
        "test": score_metrics(y[test], preds["test"]),
        "late_not_for_quality": score_metrics(y[late], preds["late"]),
        "prior_rate_test": prior_rate_metrics(y[fit], y[test]),
        "seconds": round(time.time() - t0, 1)}
    return info, preds


def save_predictions(df: pd.DataFrame, y: pd.Series, masks: dict[str, np.ndarray],
                     preds: dict[str, np.ndarray], path: str | Path) -> None:
    parts = []
    for label in ("valid", "test", "late"):
        m = masks[label]
        parts.append(pd.DataFrame({
            "id": df.loc[m, "id"].to_numpy(), "issue_d": df.loc[m, "issue_d"].to_numpy(),
            "split": label, "y_true": y[m].to_numpy(), "y_score": preds[label]}))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(path, index=False)


def run(processed: str | Path, out_dir: str | Path, pred_dir: str | Path,
        cfg: SplitConfig = SplitConfig(), c_grid: tuple[float, ...] = (0.01, 0.1, 1.0),
        cv_years: list[int] | None = None) -> dict:
    df = pd.read_parquet(processed)
    split = assign_split(df["issue_d"], cfg)
    X_all = add_derived(df)
    y = df["target"].astype(int)
    masks = {k: (split == k).to_numpy() for k in ("train", "valid", "test", "late")}
    result: dict = {"split_config": cfg.__dict__,
                    "sizes": {k: int(v) for k, v in split.value_counts().items()},
                    "default_rate_by_split": {k: round(float(v), 4) for k, v in y.groupby(split).mean().items()},
                    "variants": {}}
    for name, exclude in (("full", False), ("no_lc_score", True)):
        info, preds = _variant(name, exclude, X_all, y, df["issue_d"], masks, c_grid,
                               cv_years or [2012, 2013])
        result["variants"][name] = info
        save_predictions(df, y, masks, preds, Path(pred_dir) / f"predictions_{name}.parquet")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline: Logistic Regression + временная валидация")
    parser.add_argument("--processed", default="data/processed/loans_clean.parquet")
    parser.add_argument("--out", default="experiments")
    parser.add_argument("--pred", default="data/processed")
    parser.add_argument("--run-name", default=None, help="имя запуска в MLflow")
    parser.add_argument("--no-track", action="store_true", help="не записывать запуск в MLflow")
    args = parser.parse_args()
    res = run(args.processed, args.out, args.pred)
    if not args.no_track:
        from src.models.tracking import log_run
        run_id = log_run(res, args.processed, run_name=args.run_name,
                         artifacts=[Path(args.out) / "baseline_metrics.json"])
        print(f"MLflow run_id: {run_id}")
    print("sizes", res["sizes"], "\ndefault_rate", res["default_rate_by_split"])
    for name, v in res["variants"].items():
        print(f"\n[{name}] C={v['best_C']} cv_auc={v['cv_auc_by_C']} ({v['seconds']} c)")
        for part in ("valid", "test", "late_not_for_quality", "prior_rate_test"):
            m = v[part]
            print(f"  {part:22s} n={m['n']:>7d} AUC={m['auc']:.4f} Gini={m['gini']:.4f} "
                  f"KS={m['ks']:.4f} Brier={m['brier']:.4f} LogLoss={m['log_loss']:.4f}")


if __name__ == "__main__":
    main()

