from __future__ import annotations

from redis.asyncio import Redis

from music_api.ai.runtime import AIInferenceRuntime
from music_api.config import Settings
from music_api.db.session import Database
from music_api.storage.providers import build_storage_provider
from music_api.storage.service import StorageService


class RuntimeResources:
    """Application-scoped external clients, created and disposed with lifespan."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database = Database(settings.database_url)
        self.ai = AIInferenceRuntime(settings)
        self.redis: Redis | None = None
        if settings.redis_enabled:
            self.redis = Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                health_check_interval=30,
            )
        self.storage_provider = build_storage_provider(settings)
        self.storage = StorageService(settings, self.storage_provider)

    async def close(self) -> None:
        try:
            if self.redis is not None:
                await self.redis.aclose()
        finally:
            try:
                await self.storage_provider.close()
            finally:
                await self.database.close()
