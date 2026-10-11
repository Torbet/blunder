from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable
from contextlib import AsyncExitStack, asynccontextmanager
from itertools import batched

from blunder.shared.analysis import PipelineConfig
from blunder.shared.core.records import AnalysisRecord, GameRecord, Record


class PipelineContext:
    def __init__(self, analysis: AnalysisRecord, records: Iterable[Record]) -> None:
        self.analysis = analysis
        self.records = list(records)

    @property
    def games(self) -> list[GameRecord]:
        return [r for r in self.records if r.type == "game"]


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

    async def run(self, records: Iterable[Record]) -> AsyncGenerator[Record]:
        analysis = AnalysisRecord(config=self.config)
        yield analysis

        async with AsyncExitStack() as stack:
            for step in self.steps:
                await stack.enter_async_context(step.open())

            for chunk in batched(records, self.config.chunk):
                ctx = PipelineContext(analysis, chunk)
                for step in self.steps:
                    await step.run(ctx)
                for record in ctx.records:
                    yield record

    @classmethod
    def build(cls, config: PipelineConfig) -> Pipeline:
        from blunder.analysis.evaluation.step import EvaluationStep

        steps: list[PipelineStep] = []

        for step in config.steps:
            match step.type:
                case "evaluation":
                    steps.append(EvaluationStep(step))

        return cls(config, steps)
