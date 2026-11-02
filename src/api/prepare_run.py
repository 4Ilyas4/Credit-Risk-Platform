"""Проверка запроса на запуск относительно набора данных и подготовка входов для валидатора.

Ошибки в запросе находятся сразу, при POST /validation-runs, а не в фоновой задаче.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.api.columns import binary_target, need_column, parse_dates, period_mask, probability
from src.api.errors import InvalidRequest
from src.api.schemas import ValidationRunRequest
from src.api.settings import Settings


@dataclass
class RunInputs:
    y_true: pd.Series
    y_score: pd.Series
    X: pd.DataFrame | None
    dates: pd.Series | None
    groups: pd.Series | None
    reference_data: pd.DataFrame | None
    notes: list[str]


def _check_mapping(df: pd.DataFrame, req: ValidationRunRequest, settings: Settings) -> None:
    m = req.mapping
    need_column(df, m.y_true, "y_true")
    need_column(df, m.y_score, "y_score")
    if m.y_true == m.y_score:
        raise InvalidRequest("y_true и y_score должны быть разными столбцами.")
    if len(m.features) > settings.max_features:
        raise InvalidRequest(f"Слишком много признаков: {len(m.features)}, максимум {settings.max_features}.")
    if len(set(m.features)) != len(m.features):
        raise InvalidRequest("В списке признаков есть повторы.")
    for f in m.features:
        need_column(df, f, "признак")
    reserved = [f for f in m.features if f in (m.y_true, m.y_score)]
    if reserved:
        raise InvalidRequest(f"Исход или скор нельзя использовать как признак: {reserved}.")
    if m.group:
        need_column(df, m.group, "group")


def _select_rows(req: ValidationRunRequest, dates: pd.Series | None, notes: list[str], index: pd.Index):
    """Возвращает (маска проверяемых строк, маска референсных строк или None)."""
    if (req.current_period or (req.reference and req.reference.period)) and dates is None:
        raise InvalidRequest("Периоды можно задавать только вместе с mapping.date.")
    ref_rows = None
    if req.reference is not None and req.reference.period is not None:
        ref_rows = period_mask(dates, req.reference.period)
        if int(ref_rows.sum()) == 0:
            raise InvalidRequest("В референсном периоде нет строк.")
    if req.current_period is not None:
        keep = period_mask(dates, req.current_period)
        if int(keep.sum()) == 0:
            raise InvalidRequest("В проверяемом периоде нет строк.")
        if ref_rows is not None and bool((keep & ref_rows).any()):
            raise InvalidRequest("Проверяемый и референсный периоды пересекаются.")
        return keep, ref_rows
    if ref_rows is not None:
        keep = ~ref_rows
        if int(keep.sum()) == 0:
            raise InvalidRequest("Вне референсного периода нет строк для проверки.")
        notes.append("current_period не задан: проверяются все строки вне референсного периода.")
        return keep, ref_rows
    return pd.Series(True, index=index), None


def _reference_data(df, req, reference_df, ref_rows, score: pd.Series, notes) -> pd.DataFrame | None:
    ref, m = req.reference, req.mapping
    if ref is None:
        return None
    if ref.period is not None:
        cols = {"y_score": score[ref_rows].to_numpy()}
        cols.update({f: df.loc[ref_rows, f].to_numpy() for f in m.features})
        return pd.DataFrame(cols)
    score_col = ref.score_column or m.y_score
    if score_col not in reference_df.columns:
        raise InvalidRequest(f"В референсном наборе нет столбца скора {score_col!r}.")
    out = pd.DataFrame({"y_score": probability(reference_df[score_col], score_col).to_numpy()})
    missing = []
    for f in m.features:
        if f in reference_df.columns:
            out[f] = reference_df[f].to_numpy()
        else:
            missing.append(f)
    if missing:
        notes.append(f"В референсном наборе нет признаков {missing[:5]}: PSI для них не рассчитан.")
    return out


def build_inputs(df: pd.DataFrame, req: ValidationRunRequest, settings: Settings,
                 reference_df: pd.DataFrame | None) -> RunInputs:
    _check_mapping(df, req, settings)
    m, notes = req.mapping, []
    y = binary_target(df[m.y_true], m.y_true)
    s = probability(df[m.y_score], m.y_score)
    dates = None
    if m.date:
        need_column(df, m.date, "date")
        dates = parse_dates(df[m.date], m.date)
    keep, ref_rows = _select_rows(req, dates, notes, df.index)
    reference = _reference_data(df, req, reference_df, ref_rows, s, notes)
    sel = keep.to_numpy()

    def take(series_or_frame):
        return None if series_or_frame is None else series_or_frame[sel].reset_index(drop=True)

    return RunInputs(
        y_true=take(y), y_score=take(s), X=take(df[m.features] if m.features else None),
        dates=take(dates), groups=take(df[m.group] if m.group else None),
        reference_data=reference, notes=notes)
