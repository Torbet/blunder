from collections.abc import Iterable
from uuid import UUID

from blunder.client import Client
from blunder.client.api.analyses import create_analysis
from blunder.client.api.games import create_game
from blunder.client.models import (
    AnalysisCreate,
    AnalysisRead,
    Control,
    Elos,
    EvaluationConfig,
    GameCreate,
    GameRead,
    MoveCreate,
    PipelineConfig,
    Players,
    Result,
)
from blunder.ingestion.core.config import settings
from blunder.shared.core.records import AnalysisRecord, GameRecord, Record


class Ingestor:
    def __init__(self) -> None:
        self.client = Client(base_url=settings.api.url)
        self.ids: dict[UUID, UUID] = {}

    async def ingest(self, records: Iterable[Record]) -> None:
        async with self.client:
            for record in records:
                match record.type:
                    case "analysis":
                        await self._ingest_analysis(record)
                    case "game":
                        await self._ingest_game(record)

    async def _ingest_analysis(self, record: AnalysisRecord) -> None:
        analysis = await create_analysis.asyncio(client=self.client, body=AnalysisCreate(config=self._config(record)))
        if not isinstance(analysis, AnalysisRead):
            raise TypeError("Failed to create analysis")
        self.ids[record.id] = analysis.id

    async def _ingest_game(self, record: GameRecord) -> None:
        game = await create_game.asyncio(
            client=self.client,
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
        self.ids[record.id] = game.id

    @staticmethod
    def _config(record: AnalysisRecord) -> PipelineConfig:
        steps = []
        for step in record.config.steps:
            match step.type:
                case "evaluation":
                    steps.append(EvaluationConfig(depth=step.depth))
        return PipelineConfig(steps=steps)
