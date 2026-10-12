from collections.abc import AsyncGenerator
from datetime import datetime
from itertools import islice
from pathlib import Path

import chess
import chess.pgn
from rich import progress

from blunder.shared.core.records import GameRecord
from blunder.shared.models.game import Color, Control, Elos, LabelBase, MoveBase, Players, Result

RESULTS: dict[str, Result] = {
    "1-0": "white",
    "0-1": "black",
    "1/2-1/2": "draw",
}

ENGINES: dict[Color, str] = {
    "white": "WhiteIsComp",
    "black": "BlackIsComp",
}


class Converter:
    async def convert(self, path: Path, limit: int | None = None) -> AsyncGenerator[GameRecord]:
        with path.open() as f:
            games = islice(iter(lambda: chess.pgn.read_game(f), None), limit)

            for game in progress.track(games, total=limit, description="Converting"):
                base, _, increment = game.headers["TimeControl"].partition("+")
                control = Control(base=float(base), increment=float(increment or 0))
                clocks = dict.fromkeys(chess.COLORS, control.base)
                engines = [color for color, key in ENGINES.items() if game.headers.get(key) == "Yes"]
                turn = game.turn()

                moves, labels = [], [LabelBase(color=color) for color in engines]

                for ply, node in enumerate(game.mainline()):
                    color = "white" if turn else "black"
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

                    if color in engines:
                        labels.append(LabelBase(color=color, ply=ply))

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
                    labels=labels,
                )
