from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from music_api.api.errors import (
    application_error_handler,
    http_error_handler,
    validation_error_handler,
)
from music_api.api.middleware import RequestContextMiddleware
from music_api.api.v1.router import router as v1_router
from music_api.config import Settings, get_settings
from music_api.errors import ApplicationError
from music_api.observability import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings)
    docs_enabled = resolved_settings.should_enable_docs
    app_base_path = os.environ.get("APP_BASE_PATH", "").rstrip("/")
    app = FastAPI(
        title=resolved_settings.app_name,
        description=(
            "Tools for music theory and rhythm. Use the Music theory endpoints to work "
            "with notes, intervals, scales, chords, keys, progressions, transposition, "
            "and capo positions. Use Rhythm to generate a metronome schedule."
        ),
        version="1.0.0",
        debug=resolved_settings.debug,
        servers=[{"url": app_base_path}] if app_base_path else None,
        openapi_url=f"{resolved_settings.api_prefix}/openapi.json" if docs_enabled else None,
        docs_url=f"{resolved_settings.api_prefix}/docs" if docs_enabled else None,
        redoc_url=f"{resolved_settings.api_prefix}/redoc" if docs_enabled else None,
        swagger_ui_parameters={"url": "./openapi.json"},
        openapi_tags=[
            {
                "name": "Music theory",
                "description": (
                    "Parse and calculate notes, intervals, scales, chords, keys, "
                    "progressions, transposition, and capo positions."
                ),
            },
            {
                "name": "Rhythm",
                "description": "Generate beat and subdivision events for a metronome.",
            },
        ],
    )
    app.state.settings = resolved_settings

    @app.get("/", name="service_root", include_in_schema=False)
    async def service_root() -> dict[str, str]:
        return {
            "message": f"{resolved_settings.app_name} is running.",
            "api": resolved_settings.api_prefix,
            "docs": f"{resolved_settings.api_prefix}/docs" if docs_enabled else "disabled",
        }

    if resolved_settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved_settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "If-Match", "Idempotency-Key"],
            expose_headers=["ETag", "X-Request-ID"],
        )
    app.add_middleware(RequestContextMiddleware)

    app.add_exception_handler(ApplicationError, application_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.include_router(v1_router, prefix=resolved_settings.api_prefix)
    return app


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "music_api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        proxy_headers=True,
    )


app = create_app()


if __name__ == "__main__":
    run()
