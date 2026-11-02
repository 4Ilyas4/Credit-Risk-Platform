"""Bootstrap и доверительные интервалы. Стратифицированный и групповой (кластерный) варианты."""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.validation.result import MetricNotComputable

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Выборка с возвращением отдельно внутри каждого класса: доля дефолтов сохраняется."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def _cluster_indices(groups: np.ndarray, members: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Выборка целых групп с возвращением: наблюдения внутри группы считаются зависимыми."""
    chosen = rng.integers(0, len(members), len(members))
    return np.concatenate([members[i] for i in chosen])


def bootstrap_metric(y: np.ndarray, s: np.ndarray, metric: MetricFn, n_boot: int = 500,
                     alpha: float = 0.05, seed: int = 42, groups: np.ndarray | None = None) -> dict:
    """Bootstrap-распределение метрики.

    Без groups - стратифицированный bootstrap по классам. С groups - выборка целых групп.
    Реплики, на которых метрика не определена, пропускаются и считаются в n_failed.
    Возвращает значение на всей выборке, интервал (перцентили) и стандартную ошибку.
    """
    if n_boot < 20:
        raise ValueError("n_boot должен быть не меньше 20")
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    point = metric(y, s)  # MetricNotComputable пробрасывается: на исходных данных метрики нет
    rng = np.random.default_rng(seed)
    members = None
    if groups is not None:
        groups = np.asarray(groups)
        uniq, inv = np.unique(groups, return_inverse=True)
        inv = inv.ravel()
        members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
        if len(members) < 10:
            raise MetricNotComputable(
                f"Групповой bootstrap требует хотя бы 10 групп, найдено {len(members)}.")
    values = []
    failed = 0
    for _ in range(n_boot):
        idx = (_stratified_indices(y, rng) if members is None
               else _cluster_indices(groups, members, rng))
        try:
            values.append(metric(y[idx], s[idx]))
        except MetricNotComputable:
            failed += 1
    if len(values) < max(10, n_boot // 2):
        raise MetricNotComputable(
            f"Метрика не определена на {failed} из {n_boot} bootstrap-выборок, интервал не построен.")
    arr = np.asarray(values)
    lo, hi = np.quantile(arr, [alpha / 2, 1 - alpha / 2])
    return {"value": float(point), "ci_low": float(lo), "ci_high": float(hi), "std_error": float(arr.std(ddof=1)),
            "alpha": alpha, "n_boot": int(n_boot), "n_failed": int(failed),
            "method": "stratified" if members is None else f"cluster ({len(members)} групп)",
            "samples": arr}


def auto_n_boot(n: int, requested: int | None = None) -> int:
    """Число повторов: по умолчанию 500, на больших выборках меньше (иначе расчёт занимает минуты)."""
    if requested is not None:
        return requested
    return 500 if n <= 50_000 else 200


def bootstrap_metrics(y: np.ndarray, s: np.ndarray, metrics: dict[str, MetricFn], n_boot: int = 500,
                      alpha: float = 0.05, seed: int = 42, groups: np.ndarray | None = None) -> dict[str, dict]:
    """Bootstrap нескольких метрик на одних и тех же выборках (быстрее, чем по одной).

    Возвращает {имя: результат}. Для метрик, не определённых на исходных данных или на большинстве
    выборок, в словаре лежит MetricNotComputable вместо результата.
    """
    if n_boot < 20:
        raise ValueError("n_boot должен быть не меньше 20")
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    out: dict[str, dict | MetricNotComputable] = {}
    points = {}
    for name, fn in metrics.items():
        try:
            points[name] = fn(y, s)
        except MetricNotComputable as exc:
            out[name] = exc
    if not points:
        return out
    rng = np.random.default_rng(seed)
    members = None
    if groups is not None:
        uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
        inv = inv.ravel()
        members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
        if len(members) < 10:
            exc = MetricNotComputable(f"Групповой bootstrap требует хотя бы 10 групп, найдено {len(members)}.")
            return {**out, **{name: exc for name in points}}
    samples = {name: [] for name in points}
    failed = {name: 0 for name in points}
    for _ in range(n_boot):
        idx = _stratified_indices(y, rng) if members is None else _cluster_indices(groups, members, rng)
        yb, sb = y[idx], s[idx]
        for name in points:
            try:
                samples[name].append(metrics[name](yb, sb))
            except MetricNotComputable:
                failed[name] += 1
    method = "stratified" if members is None else f"cluster ({len(members)} групп)"
    for name, vals in samples.items():
        if len(vals) < max(10, n_boot // 2):
            out[name] = MetricNotComputable(
                f"Метрика не определена на {failed[name]} из {n_boot} bootstrap-выборок, интервал не построен.")
            continue
        arr = np.asarray(vals)
        lo, hi = np.quantile(arr, [alpha / 2, 1 - alpha / 2])
        out[name] = {"value": float(points[name]), "ci_low": float(lo), "ci_high": float(hi),
                     "std_error": float(arr.std(ddof=1)), "alpha": alpha, "n_boot": int(n_boot),
                     "n_failed": int(failed[name]), "method": method}
    return out


def minimum_positives_warning(n_pos: int, threshold: int = 100) -> str | None:
    """Мало дефолтов - оценки ранжирующих метрик неустойчивы. Порог - рабочий ориентир."""
    if n_pos < threshold:
        return (f"Дефолтов {n_pos} (меньше {threshold}): доверительные интервалы будут широкими, "
                "выводы о качестве ненадёжны.")
    return None
