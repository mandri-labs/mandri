from pathlib import Path

import pytest
from mandri.config.types import _windows_opencode_db_path


def test_windows_prefers_native_store_and_honors_xdg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "legacy"))
    canonical = tmp_path / ".local/share/opencode/opencode.db"
    legacy = tmp_path / "legacy/opencode/opencode.db"
    assert _windows_opencode_db_path() == canonical
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    assert _windows_opencode_db_path() == legacy
    canonical.parent.mkdir(parents=True)
    canonical.touch()
    assert _windows_opencode_db_path() == canonical
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert _windows_opencode_db_path() == tmp_path / "xdg/opencode/opencode.db"
