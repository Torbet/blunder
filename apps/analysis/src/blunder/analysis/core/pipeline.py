from abc import ABC, abstractmethod
from collections.abc import Iterable

from blunder.shared.analysis import PipelineConfig
from blunder.shared.core.records import AnalysisRecord, GameRecord, Record


class PipelineContext:
    def __init__(self, analysis: AnalysisRecord, records: Iterable[Record]) -> None:
        self.analysis = analysis
        self.records = [analysis, *records]

    @property
    def games(self) -> list[GameRecord]:
        return [r for r in self.records if r.type == "game"]


class PipelineStep(ABC):
    @abstractmethod
    async def run(self, ctx: PipelineContext) -> None: ...


class Pipeline:
    def __init__(self, config: PipelineConfig, steps: list[PipelineStep]) -> None:
        self.config = config
        self.steps = steps

    async def run(self, records: Iterable[Record]) -> list[Record]:
        ctx = PipelineContext(AnalysisRecord(config=self.config), records)
        for step in self.steps:
            await step.run(ctx)
        return ctx.records

    @classmethod
    def build(cls, config: PipelineConfig) -> Pipeline:
        from blunder.analysis.evaluation.step import EvaluationStep

        steps: list[PipelineStep] = []

        for step in config.steps:
            match step.type:
                case "evaluation":
                    steps.append(EvaluationStep(step))

        return cls(config, steps)
