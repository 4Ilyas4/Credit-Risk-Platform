"""Ошибки API с HTTP-кодами. Тексты безопасны для показа пользователю: без путей и трассировок."""
from __future__ import annotations


class ApiError(Exception):
    status_code = 400

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class UnsupportedFile(ApiError):
    status_code = 415


class FileTooLarge(ApiError):
    status_code = 413


class InvalidDataset(ApiError):
    status_code = 422


class InvalidRequest(ApiError):
    status_code = 422


class Conflict(ApiError):
    status_code = 409
