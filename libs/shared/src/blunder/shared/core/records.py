from collections.abc import AsyncIterable, Iterator
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID, uuid4

import chess
from pydantic import BaseModel, Field, TypeAdapter
from rich import progress

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

    def positions(self) -> Iterator[chess.Board]:
        board = chess.Board()
        yield board.copy()

        for move in self.moves:
            board.push_uci(move.uci)
            yield board.copy()

    def feature(self, analysis: UUID, ply: int) -> FeatureBase:
        for feature in self.features:
            if feature.analysis_id == analysis and feature.ply == ply:
                return feature

        feature = FeatureBase(analysis_id=analysis, ply=ply)
        self.features.append(feature)
        return feature


type Record = Annotated[GameRecord | AnalysisRecord, Field(discriminator="type")]


class Records:
    adapter = TypeAdapter(Record)

    @classmethod
    def load(cls, path: Path) -> Iterator[Record]:
        with progress.open(path, "r", description="Loading") as f:
            for line in f:
                yield cls.adapter.validate_json(line)

    @staticmethod
    async def save(records: AsyncIterable[Record], path: Path) -> None:
        with path.open("w") as f:
            async for record in records:
                f.write(record.model_dump_json() + "\n")
