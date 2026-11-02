# API валидатора кредитных моделей

Базовый URL по умолчанию: `http://localhost:8000`. OpenAPI: `/docs`, `/openapi.json`.

## Эндпойнты

### `POST /datasets`
Загрузка CSV/Parquet. Проверка структуры, сохранение датасета.
- Вход: multipart/form-data, поле `file`.
- Выход: `DatasetInfo` (dataset_id, columns, n_rows, n_columns, upload_seconds, created_at, …).

`DatasetInfo` дополнительно содержит:
- `suggested_mapping` — подсказка сопоставления: `{y_true, y_score, date, group}`.
- `mapping_questions` — вопросы по неоднозначным столбцам:
  - `score_kind` — скор вне [0,1]: вероятность или ранговый балл?
  - `default_value` — y_true не 0/1: какое значение считать дефолтом?
- `warnings` — замечания о структуре (пустые/константные столбцы, дубликаты и т.п.).

### `GET /datasets/{dataset_id}`
Метаданные датасета (та же `DatasetInfo`).

### `POST /validation-runs`
Создание задачи валидации.
- Вход: `ValidationRunRequest` (dataset_id, mapping, reference?, current_period?, options).
- `mapping`:
  - `y_true` — столбец исхода (0/1) или статус кредита.
  - `y_score` — столбец скора.
  - `score_kind` — `probability` (по умолчанию, скор в [0,1]) или `score` (ранговый балл).
    Для рангового балла Brier, LogLoss и калибровка получают `not_computable`.
  - `higher_is_riskier` — большее значение = выше риск (по умолчанию true). Если false,
    скор инвертируется и к референсу, и к текущим данным.
  - `default_value` — значение в `y_true`, которое считается дефолтом (для текстовых статусов).
    Без него `y_true` должен быть числовым 0/1.
  - `date`, `group`, `features` (опц.).
- `options`: `{threshold}`. Параметры `seed` (фиксированно 42) и `n_boot` (автоматически
  по размеру данных: 500 повторов при ≤ 50 000 строк, иначе 200) задаются на сервере
  и в интерфейсе не показываются. Их можно переопределить явно.
- Референс по умолчанию: если задан `mapping.date`, а `reference` не указан, ранние 70%
  строк по дате становятся референсом, проверяются поздние 30%. Отдельный датасет как
  референс — расширенный режим (`reference.dataset_id`).
- Выход: `RunInfo` (run_id, status=queued).

### `GET /validation-runs/{run_id}`
Статус и результат (если completed). `RunDetail` содержит `summary`: три карточки
(quality/stability/reliability), списки `attention` и `not_checked`, ключевые метрики.

### `GET /reports/{run_id}?format=html|pdf|json`
Скачивание отчёта. JSON-отчёт содержит `schema_version`.

### `GET /health`
Состояние сервиса.

## Ошибки

Единый формат `{"error": "понятная фраза с подсказкой", "detail": ...}` (422 для неверного
запроса). Ошибки не содержат путей и трассировок. Например, если столбец скора не найден,
в `error` перечисляются подходящие числовые столбцы.

## Схемы

- `mapping`: `{y_true, y_score, score_kind?, higher_is_riskier?, default_value?, date?, group?, features?}`.
- `options`: `{threshold?}` (+ необязательно `n_boot`, `seed`).
- `reference`: `{dataset_id?}` или `{period: {start?, end?}}`.
- `current_period`: `{start?, end?}`.

## Пример

```bash
# загрузка
curl -F "file=@data.csv" http://localhost:8000/datasets
# запуск: скор int_rate — ранговый балл, выше ставка = выше риск
curl -X POST http://localhost:8000/validation-runs -H "Content-Type: application/json" -d '{
  "dataset_id": "<id>",
  "mapping": {
    "y_true": "loan_status", "y_score": "int_rate",
    "default_value": "Charged Off", "score_kind": "score",
    "date": "issue_d", "group": "grade",
    "features": ["loan_amnt", "dti", "fico_range_low"]
  }
}'
```

