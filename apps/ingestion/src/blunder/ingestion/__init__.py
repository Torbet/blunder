import typer


def ingest() -> None:
    def _command() -> None:
        print("Running ingestion...")

    typer.run(_command)


def convert() -> None:
    def _command() -> None:
        print("Running conversion...")

    typer.run(_command)
