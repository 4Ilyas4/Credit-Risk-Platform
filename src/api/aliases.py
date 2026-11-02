"""Алиасы колонок для кредитных датасетов и человекочитаемые описания.

Используется фронтендом для автоподсказок маппинга и для подписей в UI.
"""
from __future__ import annotations

ALIASES: dict[str, list[str]] = {
    "y_true": ["default", "target", "bad_loan", "is_default", "charged_off", "loan_status", "default_flag"],
    "y_score": ["score", "pd", "probability", "risk_score", "predicted_pd", "proba", "probability_default"],
    "date": ["issue_d", "loan_issue_date", "application_date", "observation_date", "date", "issue_date"],
    "group": ["grade", "sub_grade", "segment", "product", "channel", "region", "purpose"],
}

DESCRIPTIONS: dict[str, str] = {
    "int_rate": "Процентная ставка",
    "annual_inc": "Годовой доход",
    "dti": "Долг к доходу (DTI)",
    "fico_range_low": "Нижняя граница FICO",
    "fico_range_high": "Верхняя граница FICO",
    "loan_amnt": "Сумма кредита",
    "term": "Срок кредита",
    "emp_length": "Стаж работы",
    "home_ownership": "Тип жилья",
    "purpose": "Цель кредита",
    "grade": "Грейд",
    "sub_grade": "Подгрейд",
    "installment": "Ежемесячный платёж",
    "revol_bal": "Баланс револьверного кредита",
    "revol_util": "Использование револьверного лимита",
    "total_acc": "Количество открытых счетов",
    "delinq_2yrs": "Просрочки за 2 года",
    "inq_last_6mths": "Запросы в БК за 6 мес.",
    "open_acc": "Открытые счета",
    "pub_rec": "Публичные записи",
    "loan_status": "Статус кредита",
    "issue_d": "Дата выдачи",
}

PRESETS: dict[str, dict[str, str]] = {
    "LendingClub": {
        "y_true": "loan_status",
        "y_score": "score",
        "date": "issue_d",
        "group": "grade",
    },
    "Home Credit": {
        "y_true": "TARGET",
        "date": None,
        "group": None,
    },
    "German Credit": {
        "y_true": "class",
        "date": None,
        "group": None,
    },
    "Give Me Some Credit": {
        "y_true": "SeriousDlqin2yrs",
        "date": None,
        "group": None,
    },
}


import difflib
import re


def _tokens(s: str) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", s.lower())) - {""}


def _token_overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def suggest_mapping(columns: list[str]) -> dict[str, str | None]:
    """Возвращает автоподсказку маппинга: точные алиасы → токены → fuzzy."""
    lower = {c.lower(): c for c in columns}
    result: dict[str, str | None] = {}
    for field, aliases in ALIASES.items():
        found = None
        # 1. точное совпадение
        for a in aliases:
            if a.lower() in lower:
                found = lower[a.lower()]
                break
        # 2. токенное пересечение
        if found is None:
            best_score = 0.0
            for c in columns:
                for a in aliases:
                    score = _token_overlap(c, a)
                    if score > best_score and score >= 0.3:
                        best_score = score
                        found = c
        # 3. fuzzy
        if found is None:
            best_ratio = 0.0
            for c in columns:
                for a in aliases:
                    ratio = difflib.SequenceMatcher(None, c.lower(), a.lower()).ratio()
                    if ratio > best_ratio and ratio >= 0.6:
                        best_ratio = ratio
                        found = c
        result[field] = found
    return result


def describe(col: str) -> str:
    return DESCRIPTIONS.get(col, col)
