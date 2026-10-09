import re
import tomllib
from pathlib import Path

from mandri.core.codex_versions import CODEX_FORK_VERSIONS


def test_daemon_and_worker_codex_pins_support_native_forks():
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    dependency = next(
        value
        for value in project["project"]["dependencies"]
        if value.startswith("openai-codex-cli-bin==")
    )
    version = dependency.split("==")[1]
    worker = (root / "mandri_runtime/src/mandri/runtime/worker/Dockerfile").read_text()
    match = re.search(r"@openai/codex@(\d+\.\d+\.\d+)", worker)
    assert match is not None
    assert match[1] == version
    assert version in CODEX_FORK_VERSIONS
