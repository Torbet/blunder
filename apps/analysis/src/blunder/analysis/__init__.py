import asyncio
from pathlib import Path

import typer

from blunder.analysis.core.pipeline import Pipeline
from blunder.shared.analysis import PipelineConfig
from blunder.shared.core.records import Records


def analyse() -> None:
    def _command(config: Path, input: Path, output: Path) -> None:
        pipeline = Pipeline.build(PipelineConfig.load(config))
        records = pipeline.run(Records.load(input), total=Records.count(input))
        asyncio.run(Records.save(records, output))

    typer.run(_command)
