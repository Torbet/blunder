from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field

from blunder.shared.analysis.evaluation import EvaluationConfig
from blunder.shared.analysis.maia import MaiaConfig


class PipelineConfig(BaseModel):
    chunk: int
    steps: list[Annotated[EvaluationConfig | MaiaConfig, Field(discriminator="type")]]

    @classmethod
    def load(cls, path: Path) -> PipelineConfig:
        with path.open() as file:
            return cls.model_validate(yaml.safe_load(file))
