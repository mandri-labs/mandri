import hashlib
import json
import os
import tempfile
from pathlib import Path

from mandri.core.codex_catalog import CATALOG_ENV


def materialize_catalog(
    argv: list[str], env: dict[str, str], directory: Path, exposed: str | None = None
) -> None:
    content = env.pop(CATALOG_ENV, None)
    if content is None:
        return
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = ".mandri-codex-model-" + hashlib.sha256(content.encode()).hexdigest() + ".json"
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as target:
        target.write(content)
        temporary = target.name
    try:
        os.replace(temporary, directory / name)
    finally:
        Path(temporary).unlink(missing_ok=True)
    path = str(Path(exposed or directory) / name)
    argv.extend(("-c", "model_catalog_json=" + json.dumps(path)))
