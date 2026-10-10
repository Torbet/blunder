from pydantic import BaseModel

from blunder.shared.analysis import PipelineConfig
from blunder.shared.models import Identifiable


class AnalysisBase(BaseModel):
    config: PipelineConfig


class Analysis(Identifiable, AnalysisBase): ...
