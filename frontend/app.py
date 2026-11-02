"""Streamlit-интерфейс к API валидатора кредитных моделей.

Запуск:
    streamlit run frontend/app.py
Переменная окружения API_URL задаёт адрес FastAPI (по умолчанию http://localhost:8000).
"""
from __future__ import annotations

import hashlib
import io
import os
import time

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

try:
    from src.api.aliases import suggest_mapping, describe, PRESETS
except Exception:
    suggest_mapping = None
    describe = lambda x: x
    PRESETS = {}

st.set_page_config(page_title="Credit Scoring Model Validator", layout="wide")
st.title("Credit Scoring Model Validator")
st.caption("Исследовательский инструмент. Не система одобрения кредитов.")


def api(method, path, **kw):
    r = requests.request(method, f"{API}{path}", **kw, timeout=60)
    if not r.ok:
        st.error(f"{r.status_code}: {r.text}")
        return None
    return r.json() if r.content else None


# ---------- сайдбар: состояние ----------
with st.sidebar:
    st.header("Состояние")
    if st.button("Сбросить всё и начать заново", type="secondary"):
        for k in ("dataset", "run", "uploaded_name", "uploaded_hash", "mapping", "result"):
            st.session_state.pop(k, None)
        st.rerun()
    ds_sb = st.session_state.get("dataset")
    if ds_sb:
        st.write("Активный датасет:")
        st.code(ds_sb["dataset_id"])
        st.write("Файл:", st.session_state.get("uploaded_name", "—"))

# ---------- 1. Загрузка ----------
st.header("1. Загрузка данных")
file = st.file_uploader("CSV или Parquet", type=["csv", "parquet"])
if file is not None:
    content = file.getvalue()
    h = hashlib.md5(content).hexdigest()
    if st.session_state.get("uploaded_hash") != h:
        for k in ("dataset", "run", "mapping", "result"):
            st.session_state.pop(k, None)
        st.session_state["uploaded_name"] = file.name
        st.session_state["uploaded_hash"] = h
        resp = requests.post(f"{API}/datasets", files={"file": (file.name, content, "text/csv")}, timeout=120)
        if resp.ok:
            st.session_state["dataset"] = resp.json()
            st.success(f"Загружен: {resp.json()['dataset_id']}")
        else:
            st.error(resp.text)

ds = st.session_state.get("dataset")
if ds:
    st.write("Столбцы:", pd.DataFrame(ds["columns"]))
    st.info("Требования: y_true — 0/1 (числовой); y_score — вероятность в [0,1]. "
            "Признаки не должны совпадать со скором/таргетом.")

    st.subheader("Авто-подготовка (опционально)")
    auto = st.checkbox("Авто-подготовка: бинаризовать цель и нормализовать скор", value=False)
    df_preview = pd.read_csv(io.BytesIO(content)) if "content" in locals() else None


# ---------- 2. Сопоставление ----------
if ds:
    st.header("2. Сопоставление столбцов")
    cols = [c["name"] for c in ds["columns"]]
    labels = [f"{c} — {describe(c)}" for c in cols]
    if suggest_mapping:
        sug = suggest_mapping(cols)
    else:
        sug = {}
    preset = st.selectbox("Пресет датасета (опц.)", ["Свой"] + list(PRESETS.keys()))
    if preset != "Свой" and PRESETS.get(preset):
        for k, v in PRESETS[preset].items():
            if v and v in cols:
                sug[k] = v

    def idx(colname):
        return cols.index(colname) if colname in cols else 0

    y_true = st.selectbox("y_true (целевая)", cols, index=idx(sug.get("y_true")), format_func=describe)
    y_score = st.selectbox("y_score (скор)", cols, index=idx(sug.get("y_score")), format_func=describe)
    date_col = st.selectbox("date (опц.)", ["—"] + cols, index=0 if sug.get("date") is None else cols.index(sug["date"]) + 1, format_func=lambda x: "—" if x == "—" else describe(x))
    group_col = st.selectbox("group (опц.)", ["—"] + cols, index=0 if sug.get("group") is None else cols.index(sug["group"]) + 1, format_func=lambda x: "—" if x == "—" else describe(x))
    feat_cols = st.multiselect("features (опц.)", cols, format_func=describe)

    mapping = {"y_true": y_true, "y_score": y_score}
    if date_col != "—":
        mapping["date"] = date_col
    if group_col != "—":
        mapping["group"] = group_col
    if feat_cols:
        mapping["features"] = feat_cols

    # ---------- 3. Параметры ----------
    st.header("3. Параметры анализа")
    c1, c2 = st.columns(2)
    n_boot = c1.number_input("n_boot", 10, 2000, 200)
    seed = c2.number_input("seed", 0, 10_000, 42)
    ref_mode = st.radio("Референс", ["нет", "по периоду", "другой датасет"], horizontal=True)
    reference = {}
    current_period = {}
    if ref_mode == "по периоду" and "date" in mapping:
        current_period = {"start": st.text_input("current start", "2014-07-01")}
        reference = {"period": {"end": st.text_input("reference end", "2014-06-30")}}
    elif ref_mode == "другой датасет":
        ref_id = st.text_input("reference dataset_id")
        if ref_id:
            reference = {"dataset_id": ref_id}

    if st.button("Запустить проверку", type="primary"):
        ok = True
        if df_preview is not None:
            if auto:
                # бинаризация цели
                if y_true in df_preview.columns:
                    vals = set(pd.unique(df_preview[y_true].dropna()))
                    if not vals.issubset({0, 1, 0.0, 1.0, True, False}):
                        default_statuses = [v for v in vals if str(v).lower() in {"charged off", "default", "bad", "1"}]
                        if default_statuses:
                            df_preview["default"] = df_preview[y_true].isin(default_statuses).astype(int)
                        else:
                            st.warning(f"Выбранная колонка '{y_true}' не похожа на цель (нет значений Charged Off/Default). Выберите колонку со статусом кредита.")
                # нормализация скорa
                if y_score in df_preview.columns:
                    s = pd.to_numeric(df_preview[y_score], errors="coerce")
                    if not (s.min() >= 0 and s.max() <= 1):
                        if s.max() <= 100:
                            df_preview["score"] = (s / 100).clip(0.001, 0.999)
                        else:
                            st.warning(f"Скор '{y_score}' вне [0,1] и не похож на проценты. Преобразуйте вручную или выберите другую колонку.")
            if y_true in df_preview.columns:
                vals = set(pd.unique(df_preview[y_true].dropna()))
                if not vals.issubset({0, 1, 0.0, 1.0, True, False}):
                    st.error(f"Столбец '{y_true}' должен содержать только 0 и 1. Вы выбрали колонку, которая не является целью. Выберите колонку со статусом кредита (например loan_status → default).")
                    ok = False
            if y_score in df_preview.columns:
                s = pd.to_numeric(df_preview[y_score], errors="coerce")
                if s.isna().all() or s.min() < 0 or s.max() > 1:
                    st.error(f"Столбец '{y_score}' должен быть числовым и в [0,1]. Вы выбрали не скор. Выберите колонку с вероятностью (или score/int_rate).")
                    ok = False
        if ok:
            if auto and df_preview is not None:
                buf = io.StringIO()
                df_preview.to_csv(buf, index=False)
                resp = requests.post(f"{API}/datasets", files={"file": ("prepared.csv", buf.getvalue().encode(), "text/csv")}, timeout=120)
                if not resp.ok:
                    st.error(resp.text)
                    st.stop()
                dataset_id = resp.json()["dataset_id"]
            else:
                dataset_id = ds["dataset_id"]

            body = {"dataset_id": dataset_id, "mapping": mapping,
                    "options": {"n_boot": int(n_boot), "seed": int(seed)}}
            if reference:
                body["reference"] = reference
            if current_period:
                body["current_period"] = current_period
            r = requests.post(f"{API}/validation-runs", json=body, timeout=60)
            if r.ok:
                st.session_state["run"] = r.json()
                st.success("Задача создана")
            else:
                st.error(r.text)

# ---------- 4. Статус ----------
run = st.session_state.get("run")
if run:
    st.header("4. Результат")
    rid = run["run_id"]
    placeholder = st.empty()
    while True:
        d = requests.get(f"{API}/validation-runs/{rid}", timeout=60).json()
        placeholder.write(f"Статус: **{d['status']}**")
        if d["status"] in ("completed", "failed"):
            break
        time.sleep(1)
    if d["status"] == "failed":
        st.error(d["error"])
    else:
        st.subheader("Сводка")
        st.json(d["summary"])
        st.subheader("Метрики")
        for section, items in d["result"]["sections"].items():
            with st.expander(section, expanded=True):
                st.dataframe(pd.DataFrame(items))
        if d["result"].get("warnings"):
            st.warning("\n".join(d["result"]["warnings"]))
        st.subheader("Скачать отчёт")
        for fmt in ("html", "pdf", "json"):
            url = f"{API}/reports/{rid}?format={fmt}"
            st.markdown(f"[{fmt.upper()}]({url})")
