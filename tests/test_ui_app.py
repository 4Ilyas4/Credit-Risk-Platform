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
    monkeypatch.setenv("API_URL", api_url)
    at = AppTest.from_file("frontend/app.py")
    at.run(timeout=120)

    # 1. клик по «Демо»
    demo = next(b for b in at.button if b.label == "Демо")
    demo.click()
    at.run(timeout=120)
    assert any("Найдены столбцы" in m.value for m in at.success), at.exception

    # 2. запуск проверки (без раскрытия спойлера — используется подсказка)
    run_btn = next(b for b in at.button if b.label == "Запустить проверку")
    run_btn.click()
    at.run(timeout=120)
    assert at.exception == []
    assert any("Задача создана" in m.value for m in at.success), [e.value for e in at.error]

    # 3. экран «Отчёт»: передаём датасет и run_id в новое состояние, ждём завершение
    at2 = AppTest.from_file("frontend/app.py")
    at2.session_state["dataset"] = at.session_state["dataset"]
    at2.session_state["run_id"] = at.session_state["run_id"]
    at2.run(timeout=120)
    assert at2.exception == []
    # три карточки отрисованы (метка «Качество» появляется в HTML-маркдауне карточки)
    assert any("Качество" in m.value for m in at2.markdown), [e.value for e in at2.error]