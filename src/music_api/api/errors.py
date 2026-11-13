from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from music_api.errors import ApplicationError

logger = logging.getLogger(__name__)


def problem_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    title: str,
    detail: str,
    errors: list[dict[str, str]] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    body: dict[str, Any] = {
        "type": "about:blank",
        "title": title,
        "status": status_code,
        "detail": detail,
        "instance": f"urn:request:{request_id}",
        "code": code,
        "trace_id": request_id,
    }
    if errors is not None:
        body["errors"] = errors
    return JSONResponse(
        status_code=status_code,
        content=body,
        headers=dict(headers or {}),
        media_type="application/problem+json",
    )


async def application_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    if not isinstance(exc, ApplicationError):
        raise TypeError("Application error handler received an unexpected exception.")
    return problem_response(
        request,
        status_code=exc.status_code,
        code=exc.code,
        title="Request failed",
        detail=exc.detail,
    )


async def validation_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        raise TypeError("Validation error handler received an unexpected exception.")
    errors = [
        {
            "location": ".".join(str(part) for part in error["loc"]),
            "code": str(error["type"]),
            "message": "Invalid value.",
        }
        for error in exc.errors()
    ]
    return problem_response(
        request,
        status_code=422,
        code="request_validation_failed",
        title="Request validation failed",
        detail="One or more request fields are invalid.",
        errors=errors,
    )


async def http_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    if not isinstance(exc, StarletteHTTPException):
        raise TypeError("HTTP error handler received an unexpected exception.")
    titles = {
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        429: "Too Many Requests",
    }
    details = {
        400: "The request could not be completed.",
        401: "Authentication is required.",
        403: "The operation is not permitted.",
        404: "The requested resource was not found.",
        405: "The requested method is not supported.",
        429: "Too many requests.",
    }
    return problem_response(
        request,
        status_code=exc.status_code,
        code=f"http_{exc.status_code}",
        title=titles.get(exc.status_code, "Request failed"),
        detail=details.get(exc.status_code, "The request could not be completed."),
        headers=exc.headers,
    )


