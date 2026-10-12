from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable
from contextlib import AsyncExitStack, asynccontextmanager
from itertools import batched

from rich import progress

from blunder.shared.analysis import PipelineConfig
from blunder.shared.core.records import AnalysisRecord, GameRecord, Record
from blunder.shared.models.game import FeatureBase


class PipelineContext:
    def __init__(self, analysis: AnalysisRecord, records: Iterable[Record]) -> None:
        self.analysis = analysis
        self.records = list(records)

    @property
    def games(self) -> list[GameRecord]:
        return [r for r in self.records if r.type == "game"]

    def feature(self, game: GameRecord, ply: int) -> FeatureBase:
        for feature in game.features:
            if feature.analysis_id == self.analysis.id and feature.ply == ply:
                return feature

        feature = FeatureBase(analysis_id=self.analysis.id, ply=ply)
        game.features.append(feature)
        return feature


class PipelineStep(ABC):
    @asynccontextmanager
    async def open(self) -> AsyncGenerator[None]:
        yield

    @abstractmethod
    async def run(self, ctx: PipelineContext) -> None: ...


class Pipeline:
    def __init__(self, config: PipelineConfig, steps: list[PipelineStep]) -> None:
        self.config = config
        self.steps = steps

    async def run(self, records: Iterable[Record], total: int | None = None) -> AsyncGenerator[Record]:
        analysis = AnalysisRecord(config=self.config)
        yield analysis

        async with AsyncExitStack() as stack:
            bars = stack.enter_context(progress.Progress())
            tasks = [bars.add_task(config.type.capitalize(), total=total) for config in self.config.steps]

            for step in self.steps:
                await stack.enter_async_context(step.open())

            for chunk in batched(records, self.config.chunk):
                ctx = PipelineContext(analysis, chunk)
                for task, step in zip(tasks, self.steps, strict=True):
                    await step.run(ctx)
                    bars.advance(task, len(chunk))
                for record in ctx.records:
                    yield record

    @classmethod
    def build(cls, config: PipelineConfig) -> Pipeline:
        from blunder.analysis.maia.step import MaiaStep
        from blunder.analysis.stockfish.step import StockfishStep

        steps: list[PipelineStep] = []

        for step in config.steps:
            match step.type:
                case "stockfish":
                    steps.append(StockfishStep(step))
                case "maia":
                    steps.append(MaiaStep(step))

        return cls(config, steps)
