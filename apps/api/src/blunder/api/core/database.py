import json
from collections.abc import AsyncGenerator
from datetime import datetime
from functools import cache
from importlib import import_module

from sqlalchemy import MetaData, func
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from blunder.api.core.config import settings


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_N_label)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


@cache
def engine() -> AsyncEngine:
    return create_async_engine(
        settings().postgres.url,
        json_serializer=lambda value: json.dumps(value, default=lambda value: value.model_dump(mode="json")),
    )


async def get_session() -> AsyncGenerator[AsyncSession]:
    async with AsyncSession(engine(), expire_on_commit=False) as session:
        yield session


def load_models():
    modules = ["analysis", "game"]
    for module in modules:
        import_module(f"blunder.api.{module}.models")
