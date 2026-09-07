import typer
import uvicorn


def api() -> None:
    def _command(dev: bool = False) -> None:
        uvicorn.run("blunder.api.core.app:app", host="0.0.0.0", port=8000, reload=dev)

    typer.run(_command)
