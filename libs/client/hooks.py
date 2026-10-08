import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class Hook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        from blunder.api.core.app import app

        output = Path(self.root) / "src" / "blunder" / "client"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
            json.dump(app.openapi(), file)
            file.flush()

            subprocess.run(
                [
                    "openapi-python-client",
                    "generate",
                    "--path",
                    file.name,
                    "--output-path",
                    str(output),
                    "--meta",
                    "none",
                    "--overwrite",
                ],
                check=True,
            )
