from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, TypeAdapter

from blunder.shared.models.analysis import AnalysisBase
from blunder.shared.models.game import FeatureBase, GameBase, MoveBase


class RecordBase(BaseModel):
    id: UUID = Field(default_factory=uuid4)


class AnalysisRecord(RecordBase, AnalysisBase):
    type: Literal["analysis"] = "analysis"


class GameRecord(RecordBase, GameBase):
    type: Literal["game"] = "game"
    moves: list[MoveBase]
    features: list[FeatureBase] = Field(default_factory=list)


type Record = Annotated[GameRecord | AnalysisRecord, Field(discriminator="type")]


class Records:
    adapter = TypeAdapter(Record)

    @classmethod
    def load(cls, path: Path) -> Iterator[Record]:
        with path.open() as f:
            for line in f:
                yield cls.adapter.validate_json(line)

    @staticmethod
    def save(records: Iterable[Record], path: Path) -> None:
        with path.open("w") as f:
            f.writelines(record.model_dump_json() + "\n" for record in records)
