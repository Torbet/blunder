from collections.abc import Iterator

import chess
from pydantic import BaseModel, ConfigDict

from blunder.shared.core.records import GameRecord
from blunder.shared.models.game import Control, Elos


class Position(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    board: chess.Board
    elos: Elos
    control: Control
    clock: float

    @classmethod
    def replay(cls, game: GameRecord) -> Iterator[Position]:
        clocks = dict.fromkeys(chess.COLORS, game.control.base)

        for board, move in zip(game.positions(), game.moves):
            yield cls(board=board, elos=game.elos, control=game.control, clock=clocks[board.turn])
            clocks[board.turn] += game.control.increment - (move.time or 0)
