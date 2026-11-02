"""AppTest: путь новичка «Демо → отчёт» через Streamlit-интерфейс (Этап C, Шаг 7).

Запускает FastAPI в фоновом потоке (uvicorn), ставит API_URL и проигрывает
взаимодействие: нажать «Демо» → подтвердить найденные столбцы → запустить
проверку → дождаться отчёта с тремя карточками.
"""
import os
import threading
import time

import pytest

import uvicorn


@pytest.fixture(scope="module")
def api_url(tmp_path_factory):
    from src.api.app import create_app
    from src.api.settings import Settings

    storage = tmp_path_factory.mktemp("api_storage")
    app = create_app(Settings(storage_dir=storage, max_upload_mb=5, max_rows=100_000,
                               max_columns=50, max_features=20, max_cells=2_000_000))
    port = 8788
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    import requests
    url = f"http://127.0.0.1:{port}"
    for _ in range(80):
        try:
            if requests.get(f"{url}/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError("API не поднялся в AppTest")
    yield url
    server.should_exit = True
    thread.join(timeout=5)


def test_demo_to_report(api_url, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import requests
    monkeypatch.setenv("API_URL", api_url)
    at = AppTest.from_file("frontend/app.py")
    at.run(timeout=120)

    # 1. клик по «Демо»
    demo = next(b for b in at.button if b.label == "Демо")
    demo.click()
    at.run(timeout=120)
    assert any("Найдены столбцы" in m.value for m in at.success), at.exception

    # 2. запуск проверки: подсказка, без раскрытия спойлера → авто-переход на «Отчёт»
    run_btn = next(b for b in at.button if b.label == "Запустить проверку")
    run_btn.click()
    at.run(timeout=120)
    assert at.exception == []

    # 3. экран «Отчёт» с тремя карточками (включает ожидание завершения проверки)
    assert any("Качество" in m.value for m in at.markdown), [e.value for e in at.error]

    # 4. демо-ран использовал даты: проверяются поздние ~30% строк, а не все 3000
    run_id = at.session_state["run_id"]
    d = requests.get(f"{api_url}/validation-runs/{run_id}", timeout=60).json()
    assert d["status"] == "completed", d.get("error")
    assert 300 < d["summary"]["n_used"] < 1500, d["summary"].get("n_used")

    # 5. переключение обратно на «Данные» не должно падать (была баг: очистка ключей виджетов)
    at.session_state["view"] = "Данные"
    at.run(timeout=120)
    assert at.exception == [], at.exception
    assert any("Найдены столбцы" in m.value for m in at.success)