import typer
import uvicorn

from blunder.api.core.config import settings


def api() -> None:
    def _command(dev: bool = False) -> None:
        uvicorn.run(
            "blunder.api.core.app:app", host="0.0.0.0", port=settings().api.port, reload=dev
        )

    typer.run(_command)
