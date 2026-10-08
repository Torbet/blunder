import asyncio
from pathlib import Path

import typer

from blunder.analysis.core.pipeline import Pipeline, PipelineConfig
from blunder.shared.core.records import GameRecord


def analyse() -> None:
    def _command(config: Path, input: Path, output: Path) -> None:
        pipeline = Pipeline.build(PipelineConfig.load(config))
        games = list(GameRecord.load(input))
        asyncio.run(pipeline.run(games))
        GameRecord.save(output, games)

    typer.run(_command)
