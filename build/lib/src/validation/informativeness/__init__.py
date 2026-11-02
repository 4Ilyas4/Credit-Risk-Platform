"""Оценка информативности признаков: IV, Mutual Information, univariate AUC, корреляции, утечка."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.informativeness.correlation import high_correlations, variance_inflation
from src.validation.informativeness.iv_tables import iv_by_period, iv_table
from src.validation.informativeness.mutual_information import mutual_information
from src.validation.informativeness.univariate_auc import LEAK_AUC, leakage_table
from src.validation.result import MetricNotComputable, MetricResult, NOT_COMPUTABLE, OK, WARNING, ValidationReport

SECTION = "Data Informativeness"


def _not_computable(name: str, text: str) -> MetricResult:
    return MetricResult(name, None, NOT_COMPUTABLE, "", text, "")


def _prepare(X, y):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X должен быть pandas.DataFrame")
    y_arr = pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce")
    if len(y_arr) != len(X):
        raise ValueError(f"Длина y ({len(y_arr)}) не совпадает с числом строк X ({len(X)})")
    keep = y_arr.notna().to_numpy()
    return X.reset_index(drop=True).loc[keep].reset_index(drop=True), y_arr[keep].to_numpy(float), keep


def _iv_result(iv: pd.DataFrame, n_bins: int, top_k: int) -> MetricResult:
    susp = iv[iv["level"] == "suspicious"]["feature"].tolist() if len(iv) else []
    useless = int((iv["level"] == "useless").sum()) if len(iv) else 0
    return MetricResult(
        "Information Value", float(iv["iv"].max()) if len(iv) else None, WARNING if susp else OK,
        "IV измеряет, насколько признак разделяет дефолты и недефолты (по корзинам). Значение - максимальный IV.",
        f"Признаков без информативности (IV < 0.02): {useless} из {len(iv)}. "
        + (f"IV ≥ 0.5 у {susp[:5]}: слишком сильная связь, проверьте утечку." if susp else
           "Подозрительно сильных признаков нет."),
        "Ориентиры IV (0.02 / 0.1 / 0.3 / 0.5) общепринятые, но условные. IV зависит от числа корзин "
        f"({n_bins}) и растёт у признаков с большим числом категорий. Корзины построены по всем переданным данным.",
        {"table": iv.head(top_k)})


def _leak_results(X: pd.DataFrame, yv: np.ndarray, threshold: float, top_k: int) -> list[MetricResult]:
    leak = leakage_table(X, yv, threshold)
    flagged = leak[leak["suspicious"]]["feature"].tolist() if len(leak) else []
    return [
        MetricResult(
            "Univariate ROC-AUC", float(leak["strength"].max()) if len(leak) else None, OK,
            "Ранжирующая способность каждого числового признака по отдельности. Значение - максимум max(AUC, 1-AUC).",
            "Очень большая сила признака требует проверки его происхождения.",
            "Только числовые признаки. Показывает одномерную связь и не учитывает взаимодействия.",
            {"table": leak.head(top_k)}),
        MetricResult(
            "Проверка на утечку", float(len(flagged)), WARNING if flagged else OK,
            f"Числовые признаки, у которых одномерная сила не ниже {threshold}.",
            (f"Подозрительные признаки: {flagged[:10]}. Проверьте, доступны ли они на момент принятия решения."
             if flagged else "Признаков с подозрительно сильной одномерной связью не найдено."),
            "Это сигнал для ручной проверки, а не доказательство. Слабая утечка (ниже порога) этим способом "
            "не находится, поэтому отсутствие сигнала не означает отсутствие утечки.", {"flagged": flagged}),
    ]


def _collinearity_results(X: pd.DataFrame, top_k: int) -> list[MetricResult]:
    corr = high_correlations(X)
    out = [MetricResult(
        "Корреляции", float(len(corr)), WARNING if len(corr) else OK,
        "Число пар числовых признаков с |корреляцией Спирмена| ≥ 0.9.",
        f"Сильно связанных пар: {len(corr)}." if len(corr) else "Сильно связанных пар не найдено.",
        "Спирмен ловит только монотонную связь. Для больших данных используется выборка.",
        {"table": corr.head(top_k)})]
    vif = variance_inflation(X)
    if len(vif):
        high = vif[vif["vif"] > 10]
        finite = vif["vif"][np.isfinite(vif["vif"])]
        out.append(MetricResult(
            "Мультиколлинеарность (VIF)", float(finite.max()) if len(finite) == len(vif) else float("inf"),
            WARNING if len(high) else OK,
            "Variance Inflation Factor: во сколько раз дисперсия коэффициента растёт из-за связи признаков.",
            f"Признаков с VIF > 10: {len(high)}." if len(high) else "VIF > 10 нет.",
            "Считается по строкам без пропусков. inf означает точную линейную зависимость. Порог 10 условный. "
            "Важно для линейных моделей и слабо влияет на деревья.", {"table": vif.head(top_k)}))
    return out



def _period_result(X: pd.DataFrame, yv: np.ndarray, dates: pd.Series, iv: pd.DataFrame, n_bins: int) -> MetricResult:
    top = iv["feature"].head(8).tolist() if len(iv) else []
    per = iv_by_period(X, yv, dates, top, n_bins)
    if per["period"].nunique() < 2:
        return _not_computable("Стабильность IV по периодам",
                               "Недостаточно периодов с достаточным числом дефолтов и недефолтов.")
    piv = per.pivot(index="feature", columns="period", values="iv")
    rel = ((piv.max(axis=1) - piv.min(axis=1)) / piv.max(axis=1).replace(0, np.nan)).dropna()
    unstable = rel[rel > 0.5].index.tolist()
    return MetricResult(
        "Стабильность IV по периодам", float(rel.max()) if len(rel) else None, WARNING if unstable else OK,
        "Относительный размах IV (max-min)/max по годам для самых сильных признаков.",
        (f"IV сильно меняется у {unstable[:5]}: связь признака с target нестабильна." if unstable else
         "IV сильных признаков держится в разумных пределах по периодам."),
        "Учитываются годы с ≥ 50 дефолтами и недефолтами. Изменение IV может быть шумом малых выборок "
        "или следствием незрелости target в поздних периодах. Порог 0.5 условный.",
        {"table": piv.reset_index()})


def assess_informativeness(X: pd.DataFrame, y, dates=None, n_bins: int = 10, top_k: int = 20,
                           leak_threshold: float = LEAK_AUC) -> ValidationReport:
    """Информативность признаков относительно target.

    Границы корзин IV считаются по всем переданным данным. Для честной оценки на новых данных
    их нужно считать на обучающей части. Если dates не переданы, сравнение по периодам не выполняется.
    """
    X, yv, keep = _prepare(X, y)
    report = ValidationReport(meta={"n_rows": int(len(X)), "n_features": int(X.shape[1])})
    if X.shape[1] == 0:
        report.add(SECTION, _not_computable("Информативность", "В X нет столбцов."))
        return report
    if len(np.unique(yv)) < 2:
        report.add(SECTION, _not_computable(
            "Информативность", "В target один класс: показатели информативности не определены."))
        return report
    iv = iv_table(X, yv, n_bins)
    report.add(SECTION, _iv_result(iv, n_bins, top_k))
    try:
        mi = mutual_information(X, yv)
        report.add(SECTION, MetricResult(
            "Mutual Information", float(mi["mutual_information"].max()) if len(mi) else None, OK,
            "Нелинейная связь числового признака с target. Значение - максимум по признакам.",
            "Используйте для ранжирования признаков, а не для сравнения абсолютных значений.",
            "Оценка приближённая, считается на выборке до 50 000 строк, пропуски заполняются медианой. "
            "Категориальные признаки не учитываются.", {"table": mi.head(top_k)}))
    except MetricNotComputable as exc:
        report.add(SECTION, _not_computable("Mutual Information", str(exc)))
    report.add(SECTION, _leak_results(X, yv, leak_threshold, top_k))
    report.add(SECTION, _collinearity_results(X, top_k))
    if dates is None:
        report.meta["skipped_checks"] = "Не переданы dates: сравнение по периодам не выполнялось."
        return report
    d = pd.Series(dates).reset_index(drop=True)
    if len(d) != len(keep):
        raise ValueError("Длина dates не совпадает с y")
    report.add(SECTION, _period_result(X, yv, d.loc[keep].reset_index(drop=True), iv, n_bins))
    return report

