from pathlib import Path

import typer

from blunder.ingestion.pgn.converter import Converter
from blunder.shared.core.records import GameRecord


def ingest() -> None:
    def _command() -> None:
        print("Running ingestion...")

    typer.run(_command)


def convert() -> None:
    def _command(input: Path, output: Path) -> None:
        records = Converter().convert(input)
        GameRecord.save(output, records)

    typer.run(_command)
