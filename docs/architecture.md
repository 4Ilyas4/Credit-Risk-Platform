# Архитектура

Поток данных:
1. Пользователь загружает файл → `POST /datasets` → сохранение.
2. Пользователь настраивает маппинг → `POST /validation-runs` → задача.
3. Worker выполняет `run_job` → результат сохраняется.
4. Пользователь получает статус и отчёт → `GET /validation-runs/{id}`, `GET /reports/{id}`.

Компоненты:
- FastAPI (API)
- JobRunner (асинхронные задачи)
- Storage (файлы)
- Validation package (метрики)
- Streamlit (UI)
- Docker/Compose (деплой)
- GitHub Actions (CI)
