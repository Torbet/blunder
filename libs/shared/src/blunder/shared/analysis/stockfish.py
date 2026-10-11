from typing import Literal

from pydantic import BaseModel


class StockfishConfig(BaseModel):
    type: Literal["stockfish"] = "stockfish"

    depth: int
