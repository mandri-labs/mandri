from pathlib import Path

import pytest
from mandri.runtime.model_selection import ModelSelectionService
from mandri.runtime.service import RuntimeService
from mandri.runtime.session_state import RuntimeStates


def test_default_storage_uses_user_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    runtime = RuntimeService({})
    models = ModelSelectionService({}, RuntimeStates(), None)
    assert runtime.attachments.root == (tmp_path / ".mandri" / "attachments").resolve()
    assert runtime._agy_profiles == tmp_path / ".mandri" / "agy-profiles"
    assert models._agy_profiles == runtime._agy_profiles


def test_explicit_storage_paths_are_preserved(tmp_path: Path) -> None:
    attachments = tmp_path / "uploads"
    profiles = tmp_path / "profiles"
    runtime = RuntimeService({}, attachments_dir=attachments, agy_profiles_dir=profiles)
    models = ModelSelectionService({}, RuntimeStates(), None, agy_profiles_dir=profiles)
    assert runtime.attachments.root == attachments.resolve()
    assert runtime._agy_profiles == profiles
    assert models._agy_profiles == profiles
