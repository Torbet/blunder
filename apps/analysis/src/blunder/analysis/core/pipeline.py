from abc import ABC, abstractmethod
from pathlib import Path

import yaml
from pydantic import BaseModel

from blunder.shared.analysis.evaluation import EvaluationConfig
from blunder.shared.core.records import GameRecord
lazy from blunder.analysis.evaluation.step import EvaluationStep


class PipelineConfig(BaseModel):
    steps: list[EvaluationConfig]

    @classmethod
    def load(cls, path: Path) -> PipelineConfig:
        with path.open() as file:
            return cls.model_validate(yaml.safe_load(file))


class PipelineStep(ABC):
    @abstractmethod
    async def run(self, games: list[GameRecord]) -> None: ...


class Pipeline:
    def __init__(self, steps: list[PipelineStep]) -> None:
        self.steps = steps

    async def run(self, games: list[GameRecord]) -> None:
        for step in self.steps:
            await step.run(games)

    @classmethod
    def build(cls, config: PipelineConfig) -> Pipeline:
        steps: list[PipelineStep] = []

        for step in config.steps:
            match step:
                case EvaluationConfig():
                    steps.append(EvaluationStep(step))

        return cls(steps)
