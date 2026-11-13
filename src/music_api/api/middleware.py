from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("music_api.requests")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class RequestContextMiddleware:
    """Adds a safe request ID and emits one structured access log per HTTP request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = self._request_id(scope)
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        started = time.perf_counter()
        status_code = 500
        response_started = False

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = int(message["status"])
                headers = list(message.get("headers", []))
                headers = [
                    (name, value) for name, value in headers if name.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception:
            logger.exception(
                "Unhandled request exception",
                extra={
                    "request_id": request_id,
                    "method": scope.get("method"),
                    "path": _route_path(scope),
                },
            )
            if not response_started:
                response = JSONResponse(
                    status_code=500,
                    media_type="application/problem+json",
                    content={
                        "type": "about:blank",
                        "title": "Internal server error",
                        "status": 500,
                        "detail": "An unexpected error occurred.",
                        "instance": f"urn:request:{request_id}",
                        "code": "internal_error",
                        "trace_id": request_id,
                    },
                    headers={"X-Request-ID": request_id},
                )
                await response(scope, receive, send)
                status_code = 500
        finally:
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "method": scope.get("method"),
                    "path": _route_path(scope),
                    "status_code": status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                },
            )

    @staticmethod
    def _request_id(scope: Scope) -> str:
        headers: dict[bytes, bytes] = dict(scope.get("headers", []))
        supplied = headers.get(b"x-request-id", b"").decode("latin-1")
        if _REQUEST_ID_PATTERN.fullmatch(supplied):
            return supplied
        return str(uuid.uuid4())


def _route_path(scope: Scope) -> str:
    route = scope.get("route")
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str):
        return route_path
    return "unmatched"
