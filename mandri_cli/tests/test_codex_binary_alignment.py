from pathlib import Path
from unittest.mock import Mock

import pytest
from mandri.cli import run
from mandri.cli.run_errors import HarnessBinaryNotFoundError
from mandri.cli.types import RunSpec
from mandri.core.ids import HarnessKind


def test_cli_codex_uses_pinned_binary_without_path_lookup(monkeypatch):
    pinned = Path("/pinned/codex")
    monkeypatch.setattr(run, "bundled_codex_path", lambda: pinned)
    which = Mock(return_value="/older/codex")
    monkeypatch.setattr(run.shutil, "which", which)
    command = run.RunCommand(RunSpec(HarnessKind.CODEX, "synthetic/model", Path("/data")))
    assert command._resolve_binary() == pinned
    which.assert_not_called()


def test_cli_missing_pinned_codex_does_not_fall_back_to_unverified_binary(monkeypatch):
    monkeypatch.setattr(run, "bundled_codex_path", Mock(side_effect=FileNotFoundError("missing")))
    command = run.RunCommand(RunSpec(HarnessKind.CODEX, "synthetic/model", Path("/data")))
    with pytest.raises(HarnessBinaryNotFoundError, match="Pinned Codex"):
        command._resolve_binary()
