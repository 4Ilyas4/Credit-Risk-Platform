"""SHAP. Доступен только при наличии проверяемой модели или совместимого интерфейса предсказаний."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.result import MetricResult, NOT_COMPUTABLE, OK

NO_MODEL_TEXT = ("Модель не передана. По одним y_true, y_score и признакам настоящие SHAP-значения неизвестной "
                 "модели получить нельзя. Доступны статистика признаков и анализ ошибок, но не объяснение "
                 "внутренней логики модели.")


def shap_results(model=None, X: pd.DataFrame | None = None, max_rows: int = 500, seed: int = 42) -> list[MetricResult]:
    """Средний |SHAP| по признакам. model - объект с методом predict_proba (класс 1 = дефолт)."""
    if model is None or X is None:
        return [MetricResult("SHAP", None, NOT_COMPUTABLE, "Вклад признаков в предсказания модели.",
                             NO_MODEL_TEXT, "Требуется доступ к модели.")]
    if not hasattr(model, "predict_proba"):
        return [MetricResult("SHAP", None, NOT_COMPUTABLE, "Вклад признаков в предсказания модели.",
                             "У объекта модели нет метода predict_proba.", "")]
    try:
        import shap
    except ImportError:
        return [MetricResult("SHAP", None, NOT_COMPUTABLE, "Вклад признаков в предсказания модели.",
                             "Пакет shap не установлен.", "")]
    sample = X.sample(min(max_rows, len(X)), random_state=seed) if len(X) > max_rows else X
    background = sample.head(100)
    explainer = shap.Explainer(lambda d: model.predict_proba(pd.DataFrame(d, columns=X.columns))[:, 1], background)
    values = np.abs(explainer(sample.head(200)).values).mean(axis=0)
    table = pd.DataFrame({"feature": X.columns, "mean_abs_shap": values}).sort_values(
        "mean_abs_shap", ascending=False).reset_index(drop=True)
    return [MetricResult("SHAP", float(table["mean_abs_shap"].max()), OK,
                         "Средний абсолютный вклад признака в вероятность дефолта по выборке.",
                         f"Самый влиятельный признак: {table['feature'].iloc[0]}.",
                         "Считается на небольшой выборке (до 200 строк) и зависит от фонового набора. "
                         "SHAP объясняет модель, а не причинность в данных.", {"table": table.head(20)})]
