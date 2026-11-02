"""Streamlit-интерфейс: экраны «Данные» и «Отчёт».

Тонкая оболочка над API. Путь новичка: файл (или «Демо») → подтверждение
столбцов → запуск → автопереход на «Отчёт». Без вопросов о seed и периодах.

Запуск:
    streamlit run frontend/app.py
Переменная окружения API_URL задаёт адрес FastAPI (по умолчанию http://localhost:8000).
"""
from __future__ import annotations

import hashlib
import os
import time

import plotly.io as pio
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

STATUS_COLOR = {"ok": "#2e7d32", "warning": "#e65100", "not_computable": "#9e9e9e"}
STATUS_LABEL = {"ok": "норма", "warning": "внимание", "not_computable": "не проверено"}

try:
    from src.api.aliases import describe, suggest_mapping
except Exception:
    describe = lambda x: x  # noqa: E731
    suggest_mapping = None

st.set_page_config(page_title="Credit Scoring Model Validator", layout="wide")
st.title("Credit Scoring Model Validator")
st.caption("Исследовательский инструмент: проверка качества, стабильности и надёжности. Не система одобрения кредитов.")


def api(method, path, **kw):
    try:
        r = requests.request(method, f"{API}{path}", **kw, timeout=120)
    except requests.ConnectionError:
        st.error(f"API недоступен по адресу {API}. Запустите бэкенд (uvicorn src.api.app:app).")
        return None
    if not r.ok:
        st.error(f"{r.status_code}: {r.json().get('error', r.text)}")
        return None
    return r.json()


# навигация; после запуска автоматически переходим на «Отчёт»
if st.session_state.pop("go_report", False):
    st.session_state["view"] = "Отчёт"
view = st.radio("Экран", ["Данные", "Отчёт"], key="view", horizontal=True, label_visibility="collapsed")


if view == "Данные":
    # ---------- загрузка ----------
    c1, c2 = st.columns([4, 1])
    file = c1.file_uploader("CSV или Parquet", type=["csv", "parquet"], label_visibility="collapsed")
    if c2.button("Демо", help="Загрузить синтетические результаты скоринга"):
        ds = api("POST", "/datasets/demo")
        if ds:
            st.session_state["dataset"] = ds
            st.session_state["dataset_name"] = "Демо-набор"
            for k in ("run_id", "map_ds", "uploaded_hash"):
                st.session_state.pop(k, None)
            st.rerun()

    elif file is not None:
        content = file.getvalue()
        h = hashlib.md5(content).hexdigest()
        if st.session_state.get("uploaded_hash") != h:
            for k in ("dataset", "run_id", "dataset_name", "map_ds"):
                st.session_state.pop(k, None)
            st.session_state["uploaded_hash"] = h
            resp = requests.post(f"{API}/datasets", files={"file": (file.name, content, "text/csv")}, timeout=120)
            if resp.ok:
                st.session_state["dataset"] = resp.json()
                st.session_state["dataset_name"] = file.name
                st.rerun()
            else:
                st.error(resp.json().get("error", resp.text))

    ds = st.session_state.get("dataset")
    if ds:
        st.write(f"**{st.session_state.get('dataset_name', '')}** — {ds['n_rows']} строк × {ds['n_columns']} столбцов")
        sm = ds.get("suggested_mapping", {})
        found = [f"{label}={sm.get(role) or '—'}" for role, label in
                 (("y_true", "исход"), ("y_score", "скор"), ("date", "дата"), ("group", "группа"))]
        st.success("Найдены столбцы: " + ", ".join(found))
        for q in ds.get("mapping_questions", []):
            st.warning(q["question"])

        cols = [c["name"] for c in ds["columns"]]

        def _idx(cols_all, name):
            return cols_all.index(name) if name in cols_all else 0

        # маппинг живёт в постоянном dict (не в ключах виджетов):
        # виджеты только внутри экрана «Данные» и их состояние стирается при
        # переключении на «Отчёт», а dict переживает смену экрана.
        sid = ds["dataset_id"]
        if st.session_state.get("map_ds") != sid:
            st.session_state["map_ds"] = sid
            st.session_state["map"] = {
                "y_true": sm.get("y_true") or (cols[0] if cols else ""),
                "y_score": sm.get("y_score") or (cols[1] if len(cols) > 1 else cols[0]),
                "date": sm.get("date") or "", "group": sm.get("group") or "",
                "dv": "", "score_kind": "probability", "higher_is_riskier": True,
                "feats": [c for c in cols
                          if c not in (sm.get("y_true"), sm.get("y_score"),
                                       sm.get("date"), sm.get("group"))][:4],
            }
        m = st.session_state["map"]

        with st.expander("Сопоставление столбцов (изменить при ошибке)"):
            # виджеты с отдельными ключами w_*; значения после рендера — в m
            st.selectbox("y_true (исход)", cols, index=_idx(cols, m["y_true"]),
                         format_func=describe, key="w_y_true")
            st.selectbox("y_score (скор)", cols, index=_idx(cols, m["y_score"]),
                         format_func=describe, key="w_y_score")
            st.radio("Тип скора", ["probability", "score"], horizontal=True,
                     index=0 if m["score_kind"] == "probability" else 1,
                     help="score — ранговый балл (int_rate, grade): Brier/LogLoss/калибровка не считаются",
                     key="w_score_kind")
            st.checkbox("Больше значение = выше риск", value=m["higher_is_riskier"], key="w_higher")
            st.text_input("default_value (опц.)", value=m["dv"], key="w_dv",
                          help="Значение в y_true = дефолт, например 'Charged Off'")
            opts_date = [""] + cols
            st.selectbox("date (опц.)", opts_date, index=_idx(opts_date, m["date"]),
                         format_func=lambda x: "—" if x == "" else describe(x), key="w_date")
            opts_grp = [""] + cols
            st.selectbox("group (опц.)", opts_grp, index=_idx(opts_grp, m["group"]),
                         format_func=lambda x: "—" if x == "" else describe(x), key="w_group")
            st.multiselect("features (опц.)", cols, default=m["feats"],
                           format_func=describe, key="w_feats")
            # синхронизация виджетов в постоянный dict
            m.update(y_true=st.session_state["w_y_true"], y_score=st.session_state["w_y_score"],
                     date=st.session_state["w_date"], group=st.session_state["w_group"],
                     dv=st.session_state["w_dv"], feats=st.session_state["w_feats"],
                     score_kind=st.session_state["w_score_kind"],
                     higher_is_riskier=st.session_state["w_higher"])

        mapping = {"y_true": m["y_true"], "y_score": m["y_score"],
                   "score_kind": m["score_kind"], "higher_is_riskier": m["higher_is_riskier"]}
        if m.get("dv"):
            mapping["default_value"] = m["dv"]
        if m.get("date"):
            mapping["date"] = m["date"]
        if m.get("group"):
            mapping["group"] = m["group"]
        if m.get("feats"):
            mapping["features"] = m["feats"]

        st.caption("Сервер сам выберет референс (ранние 70% по дате), число bootstrap-повторов и seed.")
        if st.button("Запустить проверку", type="primary"):
            body = {"dataset_id": ds["dataset_id"], "mapping": mapping}
            r = requests.post(f"{API}/validation-runs", json=body, timeout=60)
            if r.ok:
                st.session_state["run_id"] = r.json()["run_id"]
                st.session_state.pop("result", None)
                st.session_state.pop("summary", None)
                st.session_state["go_report"] = True  # авто-переход на «Отчёт»
                st.rerun()
            else:
                st.error(r.json().get("error", r.text))


else:
    run_id = st.session_state.get("run_id")
    if not run_id:
        st.info("Нажмите «Демо» или загрузите файл на экране «Данные», затем «Запустить проверку».")
    else:
        d = api("GET", f"/validation-runs/{run_id}")
        if d is None:
            st.stop()
        with st.spinner("Проверка выполняется..."):
            while d["status"] in ("queued", "running"):
                time.sleep(1)
                d = api("GET", f"/validation-runs/{run_id}")
                if d is None:
                    st.stop()
        if d["status"] == "failed":
            st.error(d.get("error", "Проверка завершилась ошибкой."))
            st.stop()
        summary = d.get("summary") or {}
        cards = summary.get("cards", {})

        # три карточки
        card_cols = st.columns(3)
        for col, (pillar, label) in zip(card_cols, [
                ("quality", "Качество"), ("stability", "Стабильность"),
                ("reliability", "Надёжность")]):
            card = cards.get(pillar, {})
            status = card.get("status", "not_computable")
            color = STATUS_COLOR.get(status, "#9e9e9e")
            n_warn = card.get("n_warning", 0)
            n_nc = card.get("n_not_computable", 0)
            n_all = card.get("n_metrics", 0)
            col.markdown(
                f"<div style='border:1px solid #ddd;border-left:6px solid {color};"
                f"border-radius:4px;padding:10px 14px'><b>{label}</b> — "
                f"<span style='color:{color};font-weight:600'>{STATUS_LABEL.get(status, status)}</span>"
                f"<br><small>{n_all} метрик · предупреждений {n_warn} · не проверено {n_nc}</small></div>",
                unsafe_allow_html=True)

        st.subheader("Требует внимания")
        att = summary.get("attention", [])
        st.markdown("\n".join(f"- {a}" for a in att) if att else "Предупреждений нет.")

        st.subheader("Не проверено")
        nc = summary.get("not_checked", [])
        st.markdown("\n".join(f"- {n}" for n in nc) if nc else "Все применимые проверки выполнены.")

        charts = d.get("result", {}).get("charts", {})
        if charts:
            st.subheader("Графики")
            for section, items in charts.items():
                with st.expander(f"Графики: {section}", expanded=True):
                    cols2 = st.columns(2)
                    for i, item in enumerate(items):
                        try:
                            fig = pio.from_json(item["fig_json"])
                            (cols2[i % 2] if len(items) > 1 else st).plotly_chart(fig, use_container_width=True)
                        except Exception:
                            st.write(item["caption"])

        st.subheader("Полный отчёт")
        try:
            html = requests.get(f"{API}/reports/{run_id}?format=html", timeout=60)
            if html.ok:
                st.components.v1.html(html.text, height=500, scrolling=True)
        except requests.ConnectionError:
            st.warning("HTML-отчёт недоступен.")
        fmt_btn = st.columns(3)
        for i, fmt in enumerate(d.get("report_formats", ["html"])):
            content = requests.get(f"{API}/reports/{run_id}?format={fmt}", timeout=120).content
            fmt_btn[i].download_button(f"Отчёт {fmt.upper()}", data=content,
                                       file_name=f"validation_report_{run_id[:8]}.{fmt}",
                                       key=f"dl_{fmt}", use_container_width=True)