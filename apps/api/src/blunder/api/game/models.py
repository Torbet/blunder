from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from blunder.api.core.database import Base
from blunder.shared.models.game import Elos, Result


class Move(Base):
    __tablename__ = "moves"

    game_id: Mapped[UUID] = mapped_column(ForeignKey("games.id"), primary_key=True)
    index: Mapped[int] = mapped_column(primary_key=True)

    uci: Mapped[str]
    evaluation: Mapped[int | None]
    time: Mapped[float | None]

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    game: Mapped[Game] = relationship(back_populates="moves")


class Game(Base):
    __tablename__ = "games"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)

    elos: Mapped[Elos] = mapped_column(JSONB)
    result: Mapped[Result]

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    moves: Mapped[list[Move]] = relationship(
        back_populates="game", lazy="selectin", cascade="all, delete-orphan", order_by=Move.index
    )
