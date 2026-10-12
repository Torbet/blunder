import re
from collections.abc import AsyncGenerator
from datetime import datetime
from itertools import islice
from pathlib import Path

import chess.pgn
from rich import progress

from blunder.shared.core.records import GameRecord
from blunder.shared.models.game import Color, Control, Elos, LabelBase, MoveBase, Players, Result, Source

RESULTS: dict[str, Result] = {
    "1-0": "white",
    "0-1": "black",
    "1/2-1/2": "draw",
}

ENGINES: dict[Color, str] = {
    "white": "WhiteIsComp",
    "black": "BlackIsComp",
}

SOURCES: dict[Source, tuple[str, str]] = {
    "fics": ("FICSGamesDBGameNo", r"(\d+)"),
    "lichess": ("Site", r"lichess\.org/(\w{8})"),
    "chesscom": ("Link", r"/game/(?:live|daily)/(\d+)"),
}

CONTROL = re.compile(r"\d+(\+\d+)?")


class Converter:
    async def convert(self, path: Path, limit: int | None = None) -> AsyncGenerator[GameRecord]:
        with path.open() as f:
            games = islice(iter(lambda: chess.pgn.read_game(f), None), limit)

            for game in progress.track(games, total=limit, description="Converting"):
                if not game.errors and self.supported(game.headers):
                    yield self.record(game)

    @staticmethod
    def supported(headers: chess.pgn.Headers) -> bool:
        return (
            CONTROL.fullmatch(headers.get("TimeControl", "")) is not None
            and "FEN" not in headers
            and headers.get("Variant", "Standard") == "Standard"
        )

    @staticmethod
    def identify(headers: chess.pgn.Headers) -> tuple[Source, str]:
        for source, (header, pattern) in SOURCES.items():
            if match := re.search(pattern, headers.get(header, "")):
                return source, match[1]

        raise ValueError(f"Unrecognised PGN source: {headers.get('Site')}")

    @staticmethod
    def control(headers: chess.pgn.Headers) -> Control:
        base, _, increment = headers["TimeControl"].partition("+")
        return Control(base=float(base), increment=float(increment or 0))

    @staticmethod
    def played(headers: chess.pgn.Headers) -> datetime:
        date = headers["Date"].replace(".", "-")
        time = headers.get("UTCTime") or headers["Time"]
        return datetime.fromisoformat(f"{date} {time}")

    @classmethod
    def record(cls, game: chess.pgn.Game) -> GameRecord:
        headers = game.headers
        source, key = cls.identify(headers)
        control = cls.control(headers)
        clocks = {"white": control.base, "black": control.base}
        engines = [color for color, header in ENGINES.items() if headers.get(header) == "Yes"]

        moves, labels = [], [LabelBase(color=color) for color in engines]

        for ply, node in enumerate(game.mainline()):
            color: Color = "black" if ply % 2 else "white"
            clock, time = node.clock(), node.emt()

            if clock is not None:
                time = round(clocks[color] + control.increment - clock, 1)
                clocks[color] = clock

            moves.append(MoveBase(ply=ply, uci=node.move.uci(), time=time))

            if color in engines:
                labels.append(LabelBase(color=color, ply=ply))

        return GameRecord(
            source=source,
            key=key,
            players=Players(white=headers["White"], black=headers["Black"]),
            result=RESULTS[headers["Result"]],
            elos=Elos(white=int(headers["WhiteElo"]), black=int(headers["BlackElo"])),
            control=control,
            played=cls.played(headers),
            moves=moves,
            labels=labels,
        )
