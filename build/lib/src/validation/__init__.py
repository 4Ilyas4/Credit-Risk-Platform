"""Credit Scoring Model Validator: независимая проверка готовых кредитных моделей.

Точки входа: validate_dataset, assess_informativeness, evaluate_predictions, run_full_validation.
Модули независимы: каждую функцию можно вызвать отдельно.
"""
from src.validation.data_quality import validate_dataset
from src.validation.evaluate import evaluate_predictions
from src.validation.full import run_full_validation
from src.validation.informativeness import assess_informativeness
from src.validation.result import MetricNotComputable, MetricResult, ValidationReport

__all__ = ["validate_dataset", "assess_informativeness", "evaluate_predictions", "run_full_validation",
           "ValidationReport", "MetricResult", "MetricNotComputable"]
