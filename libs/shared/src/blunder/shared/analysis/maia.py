from typing import Literal

from pydantic import BaseModel


class MaiaConfig(BaseModel):
    type: Literal["maia"] = "maia"

    model: str
    batch: int
