"""FastAPI-приложение. Принимает готовые скоринговые результаты и проводит их независимую проверку.
Сервис не выдаёт кредитные предсказания и не принимает решений о выдаче кредитов.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from src.api.errors import FileTooLarge, InvalidRequest
from src.api.execute import run_job
from src.api.handlers import error_response, install_handlers
from src.api.ingest import inspect_upload
from src.api.jobs import JobRunner
from src.api.prepare_run import build_inputs
from src.api.schemas import DatasetInfo, RunDetail, RunInfo, ValidationRunRequest
from src.api.settings import Settings
from src.api.storage import NotFound, Storage, now_iso

log = logging.getLogger("validator.api")
CHUNK = 1 << 20
MEDIA = {"html": "text/html; charset=utf-8", "pdf": "application/pdf", "json": "application/json"}


def _load(storage: Storage, dataset_id: str):
    try:
        return storage.load_dataset(dataset_id)
    except NotFound:
        raise InvalidRequest(f"Набор данных {dataset_id!r} не найден.") from None


def create_app(settings: Settings | None = None, inline_jobs: bool = False) -> FastAPI:
    settings = settings or Settings.from_env()
    storage = Storage(settings.storage_dir)
    runner = JobRunner(workers=settings.max_concurrent_runs, inline=inline_jobs)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        n = storage.fail_stale_runs()
        if n:
            log.warning("Помечено прерванных запусков: %d", n)
        yield
        runner.shutdown()

    app = FastAPI(title="Credit Scoring Model Validator", version="0.1.0", lifespan=lifespan,
                  description="Независимая проверка готовых кредитных моделей. "
                              "Исследовательский инструмент, не система одобрения кредитов.")
    app.state.storage, app.state.settings, app.state.runner = storage, settings, runner
    install_handlers(app)

    @app.get("/health")
    def health():
        return {"status": "ok", "pending_runs": runner.pending}

    @app.post("/datasets", response_model=DatasetInfo, status_code=201)
    async def upload_dataset(file: UploadFile = File(...)):
        """Загрузка CSV или Parquet с проверкой структуры. Возвращает описание столбцов и подсказку сопоставления."""
        t0 = time.perf_counter()
        tmp = storage.tmp_path()
        size = 0
        try:
            with open(tmp, "wb") as out:
                while chunk := await file.read(CHUNK):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise FileTooLarge(f"Файл больше {settings.max_upload_mb} МБ.")
                    out.write(chunk)
            df, meta = inspect_upload(tmp, file.filename or "upload", settings)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:  # файл ещё занят: его удалит fail_stale_runs при следующем запуске
                log.warning("Не удалось сразу удалить временный файл загрузки")
        meta["upload_seconds"] = round(time.perf_counter() - t0, 3)
        meta["created_at"] = now_iso()
        ident = storage.save_dataset(df, meta)
        return {**meta, "dataset_id": ident}

    @app.get("/datasets/{dataset_id}", response_model=DatasetInfo)
    def get_dataset(dataset_id: str):
        return storage.dataset_meta(dataset_id)

    @app.post("/validation-runs", response_model=RunInfo, status_code=202)
    def create_run(req: ValidationRunRequest):
        """Проверяет запрос, создаёт задачу и ставит её в очередь. Результат - GET /validation-runs/{id}."""
        df = _load(storage, req.dataset_id)
        ref_df = None
        if req.reference is not None and req.reference.dataset_id is not None:
            ref_df = _load(storage, req.reference.dataset_id)
        inputs = build_inputs(df, req, settings, ref_df)
        record = storage.create_run(req.dataset_id, req.model_dump(mode="json"))
        runner.submit(lambda: run_job(storage, record["run_id"], inputs, req.options.model_dump()))
        return storage.get_run(record["run_id"])

    @app.get("/validation-runs/{run_id}", response_model=RunDetail)
    def get_run(run_id: str):
        record = storage.get_run(run_id)
        result = storage.load_result(run_id) if record["status"] == "completed" else None
        return {**record, "result": result}

    @app.get("/reports/{run_id}")
    def get_report(run_id: str, format: str = "html"):
        """Скачивание отчёта: html (основной), pdf (экспорт) или json."""
        if format not in MEDIA:
            raise InvalidRequest("format должен быть html, pdf или json.")
        record = storage.get_run(run_id)
        if record["status"] in ("queued", "running"):
            return error_response(409, "Проверка ещё выполняется. Повторите запрос позже.")
        if record["status"] == "failed":
            return error_response(409, "Проверка завершилась ошибкой, отчёта нет.")
        path = storage.report_path(run_id, format)
        if format not in record["report_formats"] or not path.exists():
            raise NotFound(format)
        return FileResponse(path, media_type=MEDIA[format], filename=f"validation_report_{run_id[:8]}.{format}")

    return app


app = create_app()
