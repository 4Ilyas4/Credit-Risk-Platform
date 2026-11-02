"""Обработчики ошибок: единый формат {"error": ..., "detail": ...}, без трассировок и путей в ответе."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.errors import ApiError
from src.api.storage import NotFound

log = logging.getLogger("validator.api")


def error_response(status: int, message: str, detail=None) -> JSONResponse:
    body = {"error": message}
    if detail is not None:
        body["detail"] = detail
    return JSONResponse(status_code=status, content=body)


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return error_response(exc.status_code, exc.message)

    @app.exception_handler(NotFound)
    async def _not_found(_: Request, exc: NotFound):
        return error_response(404, "Объект не найден.")

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        items = [{"field": ".".join(str(p) for p in e["loc"][1:]) or str(e["loc"][0]),
                  "message": e["msg"], "type": e["type"]} for e in exc.errors()]
        return error_response(422, "Запрос не прошёл проверку.", items)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("Необработанная ошибка")
        return error_response(500, "Внутренняя ошибка сервера.")
