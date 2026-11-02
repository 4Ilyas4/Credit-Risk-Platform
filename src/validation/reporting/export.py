"""Сохранение отчёта: HTML (основной), PDF (экспорт, reportlab), JSON."""
from __future__ import annotations

import json
import math
from pathlib import Path

from src.validation.reporting.charts import charts_for
from src.validation.reporting.report import STATUS_LABEL, render_html
from src.validation.result import ValidationReport


def _fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.4g}" if isinstance(v, float) and math.isfinite(v) else str(v)


def _find_font() -> tuple[str, str] | None:
    """Шрифт с кириллицей. Без него PDF не покажет русский текст, поэтому экспорт в этом случае падает явно."""
    for reg, bold in (("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
                      ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")):
        if Path(reg).exists() and Path(bold).exists():
            return reg, bold
    return None


def save_pdf(report: ValidationReport, path: Path, title: str = "Отчёт о валидации кредитной модели") -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer
    import io
    from xml.sax.saxutils import escape

    fonts = _find_font()
    if fonts is None:
        raise RuntimeError("Не найден шрифт с кириллицей (Arial или DejaVu): PDF-экспорт невозможен.")
    pdfmetrics.registerFont(TTFont("Body", fonts[0]))
    pdfmetrics.registerFont(TTFont("Body-Bold", fonts[1]))
    base = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=base["BodyText"], fontName="Body", fontSize=9, leading=12)
    h1 = ParagraphStyle("h1", parent=base["Heading1"], fontName="Body-Bold")
    h2 = ParagraphStyle("h2", parent=base["Heading2"], fontName="Body-Bold")
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor="#555555")
    data = report.to_dict()
    charts = charts_for(report)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm, title=title)
    story = [Paragraph(escape(title), h1),
             Paragraph("Исследовательский инструмент: результаты не являются решением о выдаче кредита.", body)]
    if data["meta"].get("skipped_checks"):
        story.append(Paragraph("<b>Часть проверок не выполнена.</b> " + escape(str(data["meta"]["skipped_checks"])), body))
    story.append(Paragraph("Итоговые предупреждения", h2))
    for w in data["warnings"] or ["Предупреждений нет."]:
        story.append(Paragraph("• " + escape(w), body))
    for section, items in data["sections"].items():
        story.append(Paragraph(escape(section), h2))
        for caption, png in charts.get(section, []):
            story.append(Image(io.BytesIO(png), width=90 * mm, height=70 * mm, kind="proportional"))
        for r in items:
            story.append(Paragraph(f"<b>{escape(r['name'])}</b> {escape(_fmt(r['value']))} "
                                   f"[{STATUS_LABEL.get(r['status'], r['status'])}]", body))
            story.append(Paragraph("Что это: " + escape(r["explanation"]), body))
            story.append(Paragraph("Интерпретация: " + escape(r["interpretation"]), body))
            story.append(Paragraph("Ограничения: " + escape(r["limitations"]), small))
            story.append(Spacer(1, 4))
    doc.build(story)


def save_report(report: ValidationReport, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()
    if ext in (".html", ".htm"):
        p.write_text(render_html(report), encoding="utf-8")
    elif ext == ".pdf":
        save_pdf(report, p)
    elif ext == ".json":
        p.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Неизвестный формат {ext!r}: используйте .html, .pdf или .json")
    return str(p)
