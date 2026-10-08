from typing import Literal

from pydantic import BaseModel


class EvaluationConfig(BaseModel):
    type: Literal["evaluation"] = "evaluation"

    depth: int
