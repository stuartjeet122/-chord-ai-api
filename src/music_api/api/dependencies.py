from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from music_api.resources import RuntimeResources
from music_api.storage.service import StorageService


def get_resources(request: Request) -> RuntimeResources:
    resources = getattr(request.app.state, "resources", None)
    if not isinstance(resources, RuntimeResources):
        raise RuntimeError("Application resources are not available.")
    return resources


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    resources = get_resources(request)
    async for session in resources.database.session():
        yield session


async def get_storage_service(request: Request) -> StorageService:
    return get_resources(request).storage


async def get_owner_id(
    request: Request,
    development_owner_id: Annotated[str | None, Header(alias="X-Dev-User-ID")] = None,
) -> uuid.UUID:
    settings = request.app.state.settings
    if settings.is_production or not settings.storage_dev_auth_enabled:
        raise HTTPException(status_code=401, detail="Authentication is required.")
    if development_owner_id is None:
        raise HTTPException(
            status_code=401,
            detail="Provide X-Dev-User-ID while development authentication is enabled.",
        )
    try:
        return uuid.UUID(development_owner_id)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="X-Dev-User-ID must be a UUID.") from exc
