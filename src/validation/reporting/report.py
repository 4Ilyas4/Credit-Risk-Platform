"""HTML-отчёт: каждый результат содержит значение, объяснение, интерпретацию и ограничения."""
from __future__ import annotations

import base64
import html
import math

from src.validation.reporting.plots import charts_figures
from src.validation.result import (NOT_COMPUTABLE, OK, PILLAR_QUALITY, PILLAR_RELIABILITY,
                                   PILLAR_STABILITY, WARNING, ValidationReport)

STATUS_LABEL = {OK: "норма", WARNING: "внимание", NOT_COMPUTABLE: "не рассчитано"}
STATUS_COLOR = {OK: "#2e7d32", WARNING: "#e65100", NOT_COMPUTABLE: "#616161"}

CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}
h1{border-bottom:2px solid #333;padding-bottom:6px} h2{margin-top:36px;border-bottom:1px solid #ccc}
.card{border:1px solid #ddd;border-left-width:6px;border-radius:4px;padding:10px 14px;margin:12px 0}
.v{font-size:1.25em;font-weight:600} .badge{float:right;color:#fff;border-radius:3px;padding:1px 8px;font-size:.8em}
.lim{color:#555;font-size:.9em} .exp{color:#444} table{border-collapse:collapse;font-size:.85em;margin-top:6px}
td,th{border:1px solid #ccc;padding:3px 8px;text-align:right} th{background:#f2f2f2}
.warn li{margin:4px 0} .note{background:#fff8e1;padding:8px 12px;border-radius:4px}
img{max-width:100%} .charts{display:flex;flex-wrap:wrap;gap:12px}
"""


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "-" if not math.isfinite(v) else f"{v:.4g}"
    return html.escape(str(v))


def _table(rows: list[dict], max_rows: int = 15) -> str:
    if not rows:
        return ""
    cols = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{_fmt(r.get(c))}</td>" for c in cols) + "</tr>" for r in rows[:max_rows])
    more = f"<p class='lim'>Показано {max_rows} из {len(rows)} строк.</p>" if len(rows) > max_rows else ""
    return f"<table><tr>{head}</tr>{body}</table>{more}"


def _card(r: dict) -> str:
    color = STATUS_COLOR.get(r["status"], "#616161")
    table = r["details"].get("table") if isinstance(r["details"], dict) else None
    value = "" if r["value"] is None else f"<span class='v'>{_fmt(r['value'])}</span>"
    return (f"<div class='card' style='border-left-color:{color}'>"
            f"<span class='badge' style='background:{color}'>{STATUS_LABEL.get(r['status'], r['status'])}</span>"
            f"<b>{html.escape(r['name'])}</b> {value}"
            f"<div class='exp'><i>Что это:</i> {html.escape(r['explanation'])}</div>"
            f"<div><i>Интерпретация:</i> {html.escape(r['interpretation'])}</div>"
            f"<div class='lim'><i>Ограничения:</i> {html.escape(r['limitations'])}</div>"
            f"{_table(table) if isinstance(table, list) else ''}</div>")


def _png_tag(caption: str, png: bytes) -> str:
    b64 = base64.b64encode(png).decode()
    return f"<figure><img alt='{html.escape(caption)}' src='data:image/png;base64,{b64}'><figcaption>{html.escape(caption)}</figcaption></figure>"


def summarize(report: ValidationReport) -> dict:
    """Сводка по трём вопросам: качество, стабильность, надёжность.

    Возвращает:
      cards: {quality: {status, n_metrics, n_warning, n_not_computable}, stability: ..., reliability: ...}
        status карточки — худший статус внутри (not_computable > warning > ok);
        если в карточке нет ни одной метрики, status="not_computable" (не проверено).
      attention: список предупреждений (warning) с пояснением.
      not_checked: список метрик, которые не удалось рассчитать (not_computable).
    """
    data = report.to_dict()
    sections = data.get("sections", {})
    cards: dict[str, dict] = {}
    for pillar in (PILLAR_QUALITY, PILLAR_STABILITY, PILLAR_RELIABILITY):
        items = [r for sec in sections.values() for r in sec if r.get("pillar") == pillar]
        statuses = [i["status"] for i in items]
        if not items:
            status = NOT_COMPUTABLE  # не проверено: в карточке нет ни одной метрики
        elif "not_computable" in statuses:
            status = NOT_COMPUTABLE
        elif "warning" in statuses:
            status = WARNING
        else:
            status = OK
        cards[pillar] = {
            "status": status,
            "n_metrics": len(items),
            "n_warning": statuses.count(WARNING),
            "n_not_computable": statuses.count(NOT_COMPUTABLE),
        }
    attention = [w for w in data.get("warnings", []) if "не рассчитана" not in w]
    not_checked = [w for w in data.get("warnings", []) if "не рассчитана" in w]
    skipped = data.get("meta", {}).get("skipped_checks")
    if skipped:
        not_checked.append(f"[Пропущено] {skipped}")
    return {"cards": cards, "attention": attention, "not_checked": not_checked}
def render_html(report: ValidationReport, title: str = "Отчёт о валидации кредитной модели") -> str:
    data = report.to_dict()
    charts = charts_figures(report)
    parts = [f"<!doctype html><html lang='ru'><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
             f"<style>{CSS}</style></head><body><h1>{html.escape(title)}</h1>"]
    meta = data["meta"]
    parts.append("<p class='note'>Исследовательский инструмент: результаты не являются решением о выдаче кредита. "
                 f"Использовано строк: {_fmt(meta.get('n_used', meta.get('n_rows')))}. "
                 f"Доля дефолтов: {_fmt(meta.get('default_rate'))}.</p>")
    if meta.get("skipped_checks"):
        parts.append(f"<p class='note'><b>Часть проверок не выполнена.</b> {html.escape(str(meta['skipped_checks']))}</p>")
    parts.append("<h2>Итоговые предупреждения</h2>")
    if data["warnings"]:
        parts.append("<ul class='warn'>" + "".join(f"<li>{html.escape(w)}</li>" for w in data["warnings"]) + "</ul>")
    else:
        parts.append("<p>Предупреждений нет. Это не гарантия качества: см. ограничения каждого результата.</p>")
    plotly_inlined = False
    for section, items in data["sections"].items():
        parts.append(f"<h2>{html.escape(section)}</h2>")
        figs = charts.get(section)
        if figs:
            divs = []
            for caption, fig in figs:
                html_fig = fig.to_html(full_html=False, include_plotlyjs=not plotly_inlined)
                plotly_inlined = True
                divs.append(f"<figcaption>{html.escape(caption)}</figcaption>{html_fig}")
            parts.append("<div class='charts'>" + "".join(divs) + "</div>")
        parts.extend(_card(r) for r in items)
    parts.append("</body></html>")
    return "".join(parts)
