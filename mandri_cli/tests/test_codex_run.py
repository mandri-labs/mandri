import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from mandri.cli.run import RunCommand
from mandri.cli.types import RunSpec
from mandri.core.codex_catalog import CATALOG_ENV
from mandri.core.ids import HarnessKind
from mandri.core.model_metadata import ModelMetadata


def test_direct_codex_run_materializes_the_same_catalog_as_runtime(monkeypatch, tmp_path):
    command = RunCommand(
        RunSpec(harness=HarnessKind.CODEX, model_arg="fixture/model", base_dir=tmp_path)
    )
    command._ensure_running = Mock(return_value=SimpleNamespace(host="127.0.0.1", port=8000))
    command._create_route = Mock(return_value=("route", "scoped-key"))
    command._fetch_route_metadata = Mock(
        return_value=ModelMetadata(262144, 65536, reasoning_efforts=("low", "high"))
    )
    try:
        command._plan = command._build_run_plan()
        command._binary = Path("codex")
        args = command._harness_command()
        value = next(arg.split("=", 1)[1] for arg in args if arg.startswith("model_catalog_json="))
        entry = json.loads(Path(json.loads(value)).read_text(encoding="utf-8"))["models"][0]
        assert entry["slug"] == "fixture/model"
        assert entry["context_window"] == 262144
        assert entry["auto_compact_token_limit"] is None
        assert CATALOG_ENV not in command._plan.env
        assert "model_context_window=262144" in args
    finally:
        if command._catalogs is not None:
            command._catalogs.cleanup()
