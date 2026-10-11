import math

from pydantic import BaseModel


class Wdl(BaseModel):
    win: float
    draw: float
    loss: float


class Prediction(BaseModel):
    policy: dict[str, float]
    wdl: Wdl
    ponder: float | None

    @property
    def entropy(self) -> float:
        return -sum(p * math.log(p) for p in self.policy.values() if p)
