"""Выполнение запуска проверки: расчёт, сохранение результата и отчётов, замер времени этапов."""
from __future__ import annotations

import time
import traceback
import logging

from src.api.prepare_run import RunInputs
from src.api.storage import Storage, now_iso
from src.validation import run_full_validation

log = logging.getLogger("validator.api")


def summarize(report) -> dict:
    """Короткая сводка для списка запусков и интерфейса."""
    def val(section, name):
        try:
            return report.get(section, name).value
        except KeyError:
            return None
    statuses = [r.status for items in report.sections.values() for r in items]
    return {"auc": val("Model Performance", "ROC-AUC"), "gini": val("Model Performance", "Gini"),
            "ks": val("Model Performance", "KS"), "brier": val("Model Performance", "Brier Score"),
            "n_used": report.meta.get("n_used"), "default_rate": report.meta.get("default_rate"),
            "n_warnings": len(report.warnings), "n_not_computable": statuses.count("not_computable"),
            "skipped_checks": report.meta.get("skipped_checks")}


def run_job(storage: Storage, run_id: str, inputs: RunInputs, options: dict) -> None:
    """Фоновая задача. Любая ошибка превращается в статус failed с безопасным текстом."""
    storage.update_run(run_id, status="running", started_at=now_iso())
    timings: dict[str, float] = {}
    try:
        t0 = time.perf_counter()
        report = run_full_validation(
            inputs.y_true, inputs.y_score, X=inputs.X, dates=inputs.dates, groups=inputs.groups,
            reference_data=inputs.reference_data, threshold=options.get("threshold"),
            n_boot=options.get("n_boot"), seed=options.get("seed", 42))
        timings["validation_seconds"] = round(time.perf_counter() - t0, 3)
        for note in inputs.notes:
            report.meta.setdefault("notes", []).append(note)

        t1 = time.perf_counter()
        formats = []
        for fmt in ("html", "json", "pdf"):
            try:
                report.save(str(storage.report_path(run_id, fmt)))
                formats.append(fmt)
            except Exception as exc:  # PDF зависит от шрифта в системе: его отсутствие не должно ломать запуск
                if fmt != "pdf":
                    raise
                log.warning("PDF не создан для %s: %s", run_id, type(exc).__name__)
                report.warnings.append("PDF-отчёт не создан: в системе нет шрифта с кириллицей. Доступен HTML.")
        timings["report_seconds"] = round(time.perf_counter() - t1, 3)
        result = report.to_dict()
        storage.save_result(run_id, result)
        storage.update_run(run_id, status="completed", finished_at=now_iso(), timings=timings,
                           summary=summarize(report), report_formats=formats)
    except Exception as exc:
        log.error("Запуск %s завершился ошибкой: %s", run_id, traceback.format_exc())
        storage.update_run(run_id, status="failed", finished_at=now_iso(), timings=timings,
                           error=f"Внутренняя ошибка при выполнении проверки ({type(exc).__name__}). "
                                 "Проверьте входные данные и параметры.")
