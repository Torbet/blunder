import json
from collections.abc import AsyncGenerator
from importlib import import_module

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
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


engine = create_async_engine(
    settings().postgres.url,
    json_serializer=lambda value: json.dumps(
        value, default=lambda value: value.model_dump(mode="json")
    ),
)
factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession]:
    async with factory() as session:
        yield session


def load_models() -> None:
    for module in ["game"]:
        import_module(f"blunder.api.{module}.models")
