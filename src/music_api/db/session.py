from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


class Database:
    """Owns the async SQLAlchemy engine and request-scoped session factory."""

    def __init__(self, database_url: str) -> None:
        engine_options: dict[str, Any] = {"pool_pre_ping": True}
        if not database_url.startswith("sqlite+"):
            engine_options.update(pool_size=5, max_overflow=10, pool_recycle=1800)
        self.engine: AsyncEngine = create_async_engine(database_url, **engine_options)
        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )

    async def ping(self) -> None:
        async with self.engine.connect() as connection:
            await connection.exec_driver_sql("SELECT 1")

    async def close(self) -> None:
        await self.engine.dispose()

    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session
