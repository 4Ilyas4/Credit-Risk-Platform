"""Pydantic-схемы запросов и ответов. Лишние поля в запросах запрещены: опечатка не должна молча игнорироваться."""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Period(Strict):
    """Период по датам включительно. Можно задать только начало или только конец."""
    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def _check(self) -> "Period":
        if self.start is None and self.end is None:
            raise ValueError("Период: задайте start и/или end")
        if self.start and self.end and self.start > self.end:
            raise ValueError("Период: start не может быть позже end")
        return self


class ColumnMapping(Strict):
    """Сопоставление столбцов пользователя с переменными валидатора."""
    y_true: str = Field(min_length=1, description="Фактический исход: 0/1 или столбец со статусом кредита")
    y_score: str = Field(min_length=1, description="Предсказанный скор: вероятность или ранговый балл")
    score_kind: Literal["probability", "score"] = Field("probability", description="Тип скора: вероятность или ранговый балл")
    higher_is_riskier: bool = Field(True, description="Больше значение = выше риск. Если False, скор инвертируется")
    default_value: str | int | float | None = Field(
        None, description="Значение в y_true, которое считается дефолтом (1). "
                          "Если не задано, y_true должен быть числовым 0/1")
    date: str | None = Field(None, description="Дата выдачи/наблюдения (нужна для периодов и backtesting)")
    group: str | None = Field(None, description="Группа для группового bootstrap и ошибок по сегментам")
    features: list[str] = Field(default_factory=list, max_length=500,
                                description="Признаки модели. Без них проверки данных и информативности пропускаются")


class RunOptions(Strict):
    threshold: float | None = Field(None, gt=0, lt=1, description="Порог для Precision/Recall/F1. По умолчанию - доля дефолтов")
    n_boot: int | None = Field(None, ge=20, le=5000, description="Число bootstrap-повторов. По умолчанию 500 (200 при > 50 000 строк)")
    seed: int = Field(42, ge=0, le=2**31 - 1)


class ReferenceSpec(Strict):
    """Референсная выборка: период тех же данных ИЛИ другой загруженный набор. Ровно один из вариантов."""
    dataset_id: str | None = None
    score_column: str | None = Field(None, description="Столбец скора в референсном наборе (по умолчанию как в mapping.y_score)")
    period: Period | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "ReferenceSpec":
        if (self.dataset_id is None) == (self.period is None):
            raise ValueError("Референс: укажите ровно одно из dataset_id и period")
        if self.period is not None and self.score_column is not None:
            raise ValueError("Референс: score_column относится только к dataset_id")
        return self


class ValidationRunRequest(Strict):
    dataset_id: str
    mapping: ColumnMapping
    options: RunOptions = Field(default_factory=RunOptions)
    current_period: Period | None = Field(None, description="Ограничить проверяемые строки периодом")
    reference: ReferenceSpec | None = None


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    kind: Literal["numeric", "datetime", "boolean", "text"]
    missing_share: float
    n_unique: int | None = None
    date_min: str | None = None
    date_max: str | None = None
    looks_like_date: bool = False


class SuggestedMapping(BaseModel):
    y_true: str | None = None
    y_score: str | None = None
    date: str | None = None
    group: str | None = None


class MappingQuestion(BaseModel):
    """Вопрос пользователю по неоднозначному сопоставлению столбцов.

    field — какой параметр маппинга нужно уточнить.
    """
    field: Literal["score_kind", "default_value"]
    question: str


class DatasetInfo(BaseModel):
    dataset_id: str
    filename: str
    format: Literal["csv", "parquet"]
    size_bytes: int
    sha256: str
    n_rows: int
    n_columns: int
    columns: list[ColumnInfo]
    preview: list[dict[str, Any]]
    warnings: list[str]
    suggested_mapping: SuggestedMapping
    mapping_questions: list[MappingQuestion] = Field(
        default_factory=list, description="Вопросы по неоднозначным столбцам: тип скора, значение дефолта")
    parse_seconds: float
    upload_seconds: float | None = None
    created_at: str


class RunInfo(BaseModel):
    run_id: str
    dataset_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    request: dict[str, Any]
    timings: dict[str, float] | None = None
    summary: dict[str, Any] | None = None
    report_formats: list[str] = Field(default_factory=list)
    error: str | None = None


class RunDetail(RunInfo):
    result: dict[str, Any] | None = None
