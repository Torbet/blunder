from collections.abc import Iterable

from blunder.client import Client
from blunder.client.api.games import create_game
from blunder.client.models import Elos, GameCreate, GameRead, MoveCreate, Result
from blunder.ingestion.core.config import settings
from blunder.shared.core.records import GameRecord


class Ingestor:
    def __init__(self):
        self.client = Client(base_url=settings().api.url)

    async def ingest(self, records: Iterable[GameRecord]) -> None:
        for record in records:
            game = await create_game.asyncio(
                client=self.client,
                body=GameCreate(
                    result=Result(record.result),
                    elos=Elos(white=record.elos.white, black=record.elos.black),
                    moves=[
                        MoveCreate(
                            index=move.index,
                            uci=move.uci,
                            evaluation=move.evaluation,
                            time=move.time,
                        )
                        for move in record.moves
                    ],
                ),
            )
            assert isinstance(game, GameRead)
