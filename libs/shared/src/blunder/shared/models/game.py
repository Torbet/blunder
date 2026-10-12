from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from blunder.shared.models import Identifiable, Timestamped

type Color = Literal["white", "black"]
type Result = Literal["white", "black", "draw"]
type Source = Literal["fics", "synthetic", "chesscom", "lichess"]


class Players(BaseModel):
    white: str
    black: str


class Elos(BaseModel):
    white: int
    black: int


class Control(BaseModel):
    base: float
    increment: float


class GameBase(BaseModel):
    source: Source
    key: str

    result: Result
    players: Players
    elos: Elos
    control: Control
    played: datetime


class Game(Identifiable, GameBase): ...


class MoveBase(BaseModel):
    ply: int
    uci: str
    time: float | None = None


class Move(Timestamped, MoveBase):
    game_id: UUID


class FeatureBase(BaseModel):
    analysis_id: UUID
    ply: int

    evaluation: int | None = None
    best: str | None = None

    probability: float | None = None
    entropy: float | None = None
    ponder: float | None = None

    win: float | None = None
    draw: float | None = None
    loss: float | None = None


class Feature(Timestamped, FeatureBase):
    game_id: UUID


class LabelBase(BaseModel):
    color: Color
    ply: int | None = None
