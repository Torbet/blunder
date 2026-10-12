from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, ForeignKeyConstraint, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from blunder.api.core.database import Base
from blunder.shared.models.game import Control, Elos, Players, Result


class Game(Base):
    __tablename__ = "games"
    __table_args__ = (UniqueConstraint("players", "played"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)

    result: Mapped[Result]
    players: Mapped[Players] = mapped_column(JSONB)
    elos: Mapped[Elos] = mapped_column(JSONB)
    control: Mapped[Control] = mapped_column(JSONB)

    played: Mapped[datetime]

    moves: Mapped[list[Move]] = relationship(
        back_populates="game",
        lazy="selectin",
        cascade="all, delete-orphan",
        order_by="Move.ply",
    )


class Move(Base):
    __tablename__ = "moves"

    game_id: Mapped[UUID] = mapped_column(ForeignKey("games.id"), primary_key=True)
    ply: Mapped[int] = mapped_column(primary_key=True)

    uci: Mapped[str]
    time: Mapped[float | None]

    game: Mapped[Game] = relationship(back_populates="moves")


class Feature(Base):
    __tablename__ = "features"
    __table_args__ = (ForeignKeyConstraint(["game_id", "ply"], ["moves.game_id", "moves.ply"], ondelete="CASCADE"),)

    game_id: Mapped[UUID] = mapped_column(ForeignKey("games.id"), primary_key=True)
    analysis_id: Mapped[UUID] = mapped_column(ForeignKey("analyses.id"), primary_key=True)
    ply: Mapped[int] = mapped_column(primary_key=True)

    evaluation: Mapped[int | None]
    best: Mapped[str | None]

    probability: Mapped[float | None]
    entropy: Mapped[float | None]
    ponder: Mapped[float | None]

    win: Mapped[float | None]
    draw: Mapped[float | None]
    loss: Mapped[float | None]
