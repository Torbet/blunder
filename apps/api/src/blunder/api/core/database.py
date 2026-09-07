import json
from collections.abc import AsyncGenerator
from functools import cache
from importlib import import_module

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from blunder.api.core.config import settings


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


@cache
def engine() -> AsyncEngine:
    return create_async_engine(
        settings().postgres.url,
        json_serializer=lambda value: json.dumps(
            value, default=lambda value: value.model_dump(mode="json")
        ),
    )


@cache
def factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine(), expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession]:
    async with factory()() as session:
        yield session


def load_models() -> None:
    for module in ["game"]:
        import_module(f"blunder.api.{module}.models")
