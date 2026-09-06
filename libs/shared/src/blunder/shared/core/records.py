from collections.abc import Iterable, Iterator
from pathlib import Path

import chess

from blunder.shared.models.game import GameBase, MoveBase


class GameRecord(GameBase):
    moves: list[MoveBase]

    @classmethod
    def save(cls, path: Path, records: Iterable[GameRecord]) -> None:
        with path.open("w") as f:
            f.writelines(record.model_dump_json() + "\n" for record in records)

    @classmethod
    def load(cls, path: Path) -> Iterator[GameRecord]:
        with path.open() as f:
            for line in f:
                yield cls.model_validate_json(line)

    def board(self, index: int) -> chess.Board:
        board = chess.Board()
        for move in self.moves[:index]:
            board.push_uci(move.uci)
        return board
