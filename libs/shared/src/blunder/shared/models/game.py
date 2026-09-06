from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from blunder.shared.models import Identifiable

type Result = Literal["white", "black", "draw"]


class Elos(BaseModel):
    white: int
    black: int


class GameBase(BaseModel):
    result: Result
    elos: Elos


class Game(Identifiable, GameBase): ...


class MoveBase(BaseModel):
    index: int
    uci: str
    evaluation: float
    time: float


class Move(Identifiable, MoveBase):
    game_id: UUID
