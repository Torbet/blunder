from collections.abc import Iterator
from pathlib import Path

import chess
import chess.pgn

from blunder.shared.core.records import GameRecord
from blunder.shared.models.game import Elos, MoveBase, Result

RESULTS: dict[str, Result] = {
    "1-0": "white",
    "0-1": "black",
    "1/2-1/2": "draw",
}


class Converter:
    def convert(self, path: Path) -> Iterator[GameRecord]:
        with path.open() as f:
            while game := chess.pgn.read_game(f):
                initial, _, increment = game.headers["TimeControl"].partition("+")
                increment = float(increment or 0)
                clocks = dict.fromkeys(chess.COLORS, float(initial))
                turn = game.turn()

                moves: list[MoveBase] = []

                for index, node in enumerate(game.mainline()):
                    clock, time = node.clock(), None

                    if clock is not None:
                        time = round(clocks[turn] + increment - clock, 1)
                        clocks[turn] = clock

                    moves.append(
                        MoveBase(
                            index=index,
                            uci=node.move.uci(),
                            evaluation=None,
                            time=time,
                        )
                    )

                    turn = not turn

                yield GameRecord(
                    result=RESULTS[game.headers["Result"]],
                    elos=Elos(
                        white=int(game.headers["WhiteElo"]),
                        black=int(game.headers["BlackElo"]),
                    ),
                    moves=moves,
                )
