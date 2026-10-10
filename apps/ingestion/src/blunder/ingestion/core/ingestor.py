from collections.abc import Iterable

from blunder.client import Client
from blunder.client.api.games import create_game
from blunder.client.models import Elos, GameCreate, GameRead, MoveCreate, Result
from blunder.ingestion.core.config import settings
from blunder.shared.core.records import GameRecord


class Ingestor:
    async def ingest(self, records: Iterable[GameRecord]) -> None:
        async with Client(base_url=settings.api.url) as client:
            for record in records:
                game = await create_game.asyncio(
                    client=client,
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
                if not isinstance(game, GameRead):
                    raise TypeError("Failed to create game")
