from pathlib import Path

import yaml
from pydantic import BaseModel

from blunder.shared.analysis.evaluation import EvaluationConfig


class PipelineConfig(BaseModel):
    steps: list[EvaluationConfig]

    @classmethod
    def load(cls, path: Path) -> PipelineConfig:
        with path.open() as file:
            return cls.model_validate(yaml.safe_load(file))
