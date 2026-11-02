# Data

Крупные датасеты в Git не хранятся.

## Источник

Основной датасет - Lending Club accepted loans (открытый набор данных, обычно скачивается с Kaggle: "Lending Club Loan Data").
Единица наблюдения - выданный кредит. Скоринг выполняется на момент выдачи.

## Структура

```
data/raw/        исходный файл без изменений (не меняется никогда)
data/processed/  подготовленные таблицы в Parquet
```

## Подготовка

```
python -m src.data.ingestion --csv data/raw/accepted_2007_to_2018Q4.csv
python -m src.data.prepare
```

1. `ingestion` сохраняет Parquet-копию исходных данных `data/raw/loans_raw.parquet` (около 2 минут). Исходный CSV не изменяется.
2. `prepare` читает только нужные столбцы, строит `target`, приводит числа к float32 и сохраняет `data/processed/loans_clean.parquet` (около 20 секунд, ~1 ГБ в памяти).

Если скачанный архив распаковался в папку с тем же именем, что и CSV, указывайте путь к вложенному файлу.
