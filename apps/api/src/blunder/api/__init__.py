import typer


def api() -> None:
    def _command() -> None:
        print("Running api...")

    typer.run(_command)
