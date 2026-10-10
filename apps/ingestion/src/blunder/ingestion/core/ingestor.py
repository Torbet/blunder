import asyncio
from collections.abc import Iterable
from itertools import batched, groupby
from uuid import UUID

from blunder.client import Client
from blunder.client.api.analyses import create_analyses
from blunder.client.api.games import add_features, create_games
from blunder.client.models import (
    AnalysisCreate,
    Control,
    Elos,
    EvaluationConfig,
    FeatureCreate,
    GameCreate,
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
            for kind, group in groupby(records, lambda r: r.type):
                batches = batched(group, settings.ingestion.batch_size)
                for window in batched(batches, settings.ingestion.concurrency):
                    await asyncio.gather(*(self._ingest(kind, batch) for batch in window))

    async def _ingest(self, kind: str, batch: tuple[Record, ...]) -> None:
        match kind:
            case "analysis":
                await self._ingest_analysis([r for r in batch if r.type == "analysis"])
            case "game":
                await self._ingest_game([r for r in batch if r.type == "game"])

    async def _ingest_analysis(self, records: list[AnalysisRecord]) -> None:
        analyses = await create_analyses.asyncio(
            client=self.client, body=[AnalysisCreate(config=self._config(record)) for record in records]
        )
        if not isinstance(analyses, list):
            raise TypeError("Failed to create analysis")

        for record, analysis in zip(records, analyses, strict=True):
            self.ids[record.id] = analysis.id

    async def _ingest_game(self, records: list[GameRecord]) -> None:
        games = await create_games.asyncio(
            client=self.client,
            body=[
                GameCreate(
                    result=Result(record.result),
                    players=Players(white=record.players.white, black=record.players.black),
                    elos=Elos(white=record.elos.white, black=record.elos.black),
                    control=Control(base=record.control.base, increment=record.control.increment),
                    played=record.played,
                    moves=[MoveCreate(ply=move.ply, uci=move.uci, time=move.time) for move in record.moves],
                )
                for record in records
            ],
        )
        if not isinstance(games, list):
            raise TypeError("Failed to create game")

        for record, game in zip(records, games, strict=True):
            if not record.features:
                continue

            features = await add_features.asyncio(
                client=self.client,
                game_id=game.id,
                body=[
                    FeatureCreate(
                        analysis_id=self.ids[feature.analysis_id],
                        ply=feature.ply,
                        evaluation=feature.evaluation,
                        best=feature.best,
                    )
                    for feature in record.features
                ],
            )

            if not isinstance(features, list):
                raise TypeError("Failed to add features")

    @staticmethod
    def _config(record: AnalysisRecord) -> PipelineConfig:
        steps = []
        for step in record.config.steps:
            match step.type:
                case "evaluation":
                    steps.append(EvaluationConfig(depth=step.depth))
        return PipelineConfig(steps=steps)
