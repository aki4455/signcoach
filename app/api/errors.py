"""User-facing API errors: every failure is an i18n key the website can show, never a stack trace."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, error_key: str, status_code: int = 400, **params):
        super().__init__(error_key)
        self.error_key, self.status_code, self.params = error_key, status_code, params


async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse({"error_key": exc.error_key, "params": exc.params}, status_code=exc.status_code)
