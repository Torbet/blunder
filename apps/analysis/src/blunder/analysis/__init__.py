import typer


def analyse() -> None:
    def _command() -> None:
        print("Running analysis...")

    typer.run(_command)
