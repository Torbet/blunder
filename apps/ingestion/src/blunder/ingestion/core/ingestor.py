from collections.abc import Iterable

from blunder.client import Client
from blunder.client.api.games import create_game
from blunder.client.models import Control, Elos, GameCreate, GameRead, MoveCreate, Players, Result
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
                        players=Players(white=record.players.white, black=record.players.black),
                        elos=Elos(white=record.elos.white, black=record.elos.black),
                        control=Control(base=record.control.base, increment=record.control.increment),
                        played=record.played,
                        moves=[
                            MoveCreate(ply=move.ply, uci=move.uci, evaluation=move.evaluation, time=move.time)
                            for move in record.moves
                        ],
                    ),
                )
                if not isinstance(game, GameRead):
                    raise TypeError("Failed to create game")
