from abc import ABC, abstractmethod

from blunder.shared.analysis import PipelineConfig
from blunder.shared.core.records import AnalysisRecord, Record


class PipelineStep(ABC):
    @abstractmethod
    async def run(self, records: list[Record]) -> None: ...


class Pipeline:
    def __init__(self, config: PipelineConfig, steps: list[PipelineStep]) -> None:
        self.config = config
        self.steps = steps

    async def run(self, records: list[Record]) -> None:
        records.insert(0, AnalysisRecord(config=self.config))
        for step in self.steps:
            await step.run(records)

    @classmethod
    def build(cls, config: PipelineConfig) -> Pipeline:
        from blunder.analysis.evaluation.step import EvaluationStep

        steps: list[PipelineStep] = []

        for step in config.steps:
            match step.type:
                case "evaluation":
                    steps.append(EvaluationStep(step))

        return cls(config, steps)
