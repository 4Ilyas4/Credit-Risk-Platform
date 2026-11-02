"""Единый формат результата проверки: значение, объяснение, интерпретация, ограничения."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

OK = "ok"
WARNING = "warning"
NOT_COMPUTABLE = "not_computable"


class MetricNotComputable(ValueError):
    """Метрику нельзя корректно рассчитать на этих данных (например, в выборке один класс)."""


def to_jsonable(obj):
    """Приводит numpy/pandas-объекты к JSON-совместимым. NaN и inf превращаются в None."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict("records"))
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (pd.Timestamp, pd.Period)):
        return str(obj)
    if obj is pd.NaT:
        return None
    return obj


@dataclass
class MetricResult:
    """Результат одной проверки.

    value=None при status="not_computable" означает, что метрика не определена.
    value=None при status="ok" означает табличный результат: данные лежат в details.
    """
    name: str
    value: float | None
    status: str
    explanation: str
    interpretation: str
    limitations: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return to_jsonable({
            "name": self.name, "value": self.value, "status": self.status,
            "explanation": self.explanation, "interpretation": self.interpretation,
            "limitations": self.limitations, "details": self.details})


def safe_result(name: str, compute: Callable[[], float], explanation: str,
                interpret: Callable[[float], tuple[str, str]], limitations: str,
                details: dict | None = None) -> MetricResult:
    """Считает метрику. Если она не определена, возвращает value=None, а не подставляет число."""
    try:
        value = float(compute())
    except MetricNotComputable as exc:
        return MetricResult(name, None, NOT_COMPUTABLE, explanation, str(exc), limitations, details or {})
    text, status = interpret(value)
    return MetricResult(name, value, status, explanation, text, limitations, details or {})


@dataclass
class ValidationReport:
    """Результат проверки: разделы с метриками, предупреждения и сведения о входных данных."""
    sections: dict[str, list[MetricResult]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict, repr=False)  # y, s для графиков; в JSON не попадает

    def save(self, path: str) -> str:
        """Сохраняет отчёт: .html (основной формат), .pdf (экспорт) или .json."""
        from src.validation.reporting.export import save_report
        return save_report(self, path)

    def add(self, section: str, results: MetricResult | list[MetricResult]) -> None:
        items = results if isinstance(results, list) else [results]
        self.sections.setdefault(section, []).extend(items)
        for r in items:
            if r.status == WARNING:
                self.warnings.append(f"[{section}] {r.name}: {r.interpretation}")
            elif r.status == NOT_COMPUTABLE:
                self.warnings.append(f"[{section}] {r.name}: не рассчитана. {r.interpretation}")

    def get(self, section: str, name: str) -> MetricResult:
        for r in self.sections.get(section, []):
            if r.name == name:
                return r
        raise KeyError(f"Нет результата {name!r} в разделе {section!r}")

    def merge(self, other: "ValidationReport") -> None:
        for section, items in other.sections.items():
            self.sections.setdefault(section, []).extend(items)
        self.warnings.extend(other.warnings)
        self.meta.update(other.meta)
        self.artifacts.update(other.artifacts)

    def to_dict(self) -> dict:
        return to_jsonable({
            "meta": self.meta, "warnings": self.warnings,
            "sections": {s: [r.to_dict() for r in items] for s, items in self.sections.items()}})
