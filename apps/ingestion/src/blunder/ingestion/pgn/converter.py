from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn

from blunder.shared.core.records import GameRecord
from blunder.shared.models.game import Control, Elos, MoveBase, Players, Result

RESULTS: dict[str, Result] = {
    "1-0": "white",
    "0-1": "black",
    "1/2-1/2": "draw",
}


class Converter:
    def convert(self, path: Path) -> Iterator[GameRecord]:
        with path.open() as f:
            while game := chess.pgn.read_game(f):
                base, _, increment = game.headers["TimeControl"].partition("+")
                control = Control(base=float(base), increment=float(increment or 0))
                clocks = dict.fromkeys(chess.COLORS, control.base)
                turn = game.turn()

                moves = []

                for ply, node in enumerate(game.mainline()):
                    clock, time = node.clock(), node.emt()

                    if clock is not None:
                        time = round(clocks[turn] + control.increment - clock, 1)
                        clocks[turn] = clock

                    moves.append(
                        MoveBase(
                            ply=ply,
                            uci=node.move.uci(),
                            time=time,
                        )
                    )

                    turn = not turn

                date = game.headers.get("UTCDate") or game.headers["Date"]
                hour = game.headers.get("UTCTime") or game.headers["Time"]

                yield GameRecord(
                    players=Players(
                        white=game.headers["White"],
                        black=game.headers["Black"],
                    ),
                    result=RESULTS[game.headers["Result"]],
                    elos=Elos(
                        white=int(game.headers["WhiteElo"]),
                        black=int(game.headers["BlackElo"]),
                    ),
                    control=control,
                    played=datetime.fromisoformat(f"{date.replace('.', '-')} {hour}"),
                    moves=moves,
                )
