from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field

from blunder.shared.analysis.maia import MaiaConfig
from blunder.shared.analysis.stockfish import StockfishConfig


class PipelineConfig(BaseModel):
    chunk: int
    steps: list[Annotated[StockfishConfig | MaiaConfig, Field(discriminator="type")]]

    @classmethod
    def load(cls, path: Path) -> PipelineConfig:
        with path.open() as file:
            return cls.model_validate(yaml.safe_load(file))
