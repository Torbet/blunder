import asyncio
from pathlib import Path

import typer

from blunder.analysis.core.pipeline import Pipeline
from blunder.shared.analysis import PipelineConfig
from blunder.shared.core.records import Records


def analyse() -> None:
    def _command(config: Path, input: Path, output: Path) -> None:
        pipeline = Pipeline.build(PipelineConfig.load(config))

        async def run() -> None:
            with output.open("w") as f:
                async for record in pipeline.run(Records.load(input)):
                    f.write(Records.dump(record))

        asyncio.run(run())

    typer.run(_command)
