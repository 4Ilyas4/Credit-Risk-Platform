# Backend: классы и функции

## src/api/app.py
- `create_app(settings=None, inline_jobs=False) -> FastAPI` — создаёт приложение, lifespan, эндпойнты.
- `_load(storage, dataset_id)` — загрузка датасета или InvalidRequest.

### Эндпойнты
- `GET /health`
- `POST /datasets`
- `GET /datasets/{dataset_id}`
- `POST /validation-runs`
- `GET /validation-runs/{run_id}`
- `GET /reports/{run_id}`

## src/api/schemas.py
- `DatasetInfo`, `RunInfo`, `RunDetail`, `ValidationRunRequest`, `ColumnMapping`, `RunOptions`, `ReferenceSpec`, `PeriodSpec`.

## src/api/storage.py
- `class Storage` — save/load dataset, create/get run, load/save result, report_path, fail_stale_runs.
- `class NotFound` — исключение.

## src/api/ingest.py
- `inspect_upload(tmp, filename, settings)` — чтение CSV/Parquet, проверка, метаданные.

## src/api/columns.py
- Инференс типов, предупреждения о датах.

## src/api/prepare_run.py
- `build_inputs(df, req, settings, ref_df)` — сбор входов, фильтрация периодов, маппинг.

## src/api/execute.py
- `run_job(storage, run_id, inputs, options)` — выполнение валидации, сохранение результатов/отчётов.

## src/api/jobs.py
- `class JobRunner` — очередь задач, submit, shutdown, pending.

## src/api/settings.py
- `class Settings` — from_env, лимиты загрузки, storage_dir, max_concurrent_runs.

## src/api/errors.py
- `FileTooLarge`, `InvalidRequest`.

## src/api/handlers.py
- `install_handlers(app)`, `error_response`.

## src/api/aliases.py
- `ALIASES`, `DESCRIPTIONS`, `PRESETS`.
- `suggest_mapping(columns)` — точные/токенные/fuzzy автоподсказки.
- `_tokens`, `_token_overlap`.
- `describe(col)`.

## src/validation (пакет)
Точки входа:
- `validate_dataset`
- `assess_informativeness`
- `evaluate_predictions`
- `run_full_validation`

Подмодули:
- data_quality, informativeness, performance, calibration, stability, reliability, diagnostics, explainability, reporting.

Каждая функция возвращает структурированный результат с value/explanation/interpretation/limitations.
