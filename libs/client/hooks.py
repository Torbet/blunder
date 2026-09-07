import json
import subprocess
import tempfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

from blunder.api.core.app import app


class Hook(BuildHookInterface):
    def initialize(self, version, build_data):
        src = Path(self.root) / "src" / "blunder" / "client"
        config = Path(self.root) / "config.yml"

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as openapi:
            json.dump(app.openapi(), openapi)
            openapi.flush()

            subprocess.run(
                [
                    "openapi-python-client",
                    "generate",
                    "--path",
                    openapi.name,
                    "--output-path",
                    str(src),
                    "--config",
                    str(config),
                    "--meta",
                    "none",
                    "--overwrite",
                ],
                check=True,
            )
