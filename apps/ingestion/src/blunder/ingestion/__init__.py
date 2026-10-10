import asyncio
from pathlib import Path

import typer

from blunder.ingestion.core.ingestor import Ingestor
from blunder.ingestion.pgn.converter import Converter
from blunder.shared.core.records import Records


def ingest() -> None:
    def _command(input: Path) -> None:
        asyncio.run(Ingestor().ingest(Records.load(input)))

    typer.run(_command)


def convert() -> None:
    def _command(input: Path, output: Path) -> None:
        Records.save(Converter().convert(input), output)

    typer.run(_command)
