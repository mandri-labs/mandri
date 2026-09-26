"""Tests for harness executable resolution and spawn-path validation."""

import sys
from pathlib import Path

import pytest
from mandri.runtime.errors import ProcessSpawnError
from mandri.runtime.executable import require_spawn_executable, resolve_executable
from mandri.runtime.service import RuntimeService

FAKE_NAME = "mandri_fake_harness"


def _fake_executable(directory: Path) -> Path:
    name = f"{FAKE_NAME}.EXE" if sys.platform == "win32" else FAKE_NAME
    path = directory / name
    path.write_text("", encoding="utf-8")
    if sys.platform != "win32":
        path.chmod(0o755)
    return path


def test_resolve_executable_finds_fake_binary_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_executable(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert resolve_executable(FAKE_NAME) == fake


def test_resolve_executable_returns_none_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    assert resolve_executable(FAKE_NAME) is None


def test_require_spawn_executable_resolves_bare_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_executable(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert require_spawn_executable(FAKE_NAME) == str(fake)


def test_require_spawn_executable_unresolvable_bare_name_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ProcessSpawnError) as exc_info:
        require_spawn_executable(FAKE_NAME)
    assert FAKE_NAME in str(exc_info.value)


def test_require_spawn_executable_accepts_existing_path(tmp_path: Path) -> None:
    target = tmp_path / "tool.exe"
    target.write_text("", encoding="utf-8")
    assert require_spawn_executable(str(target)) == str(target)


def test_require_spawn_executable_missing_path_raises_precise_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing-harness.exe"
    with pytest.raises(ProcessSpawnError) as exc_info:
        require_spawn_executable(str(missing))
    message = str(exc_info.value)
    assert str(missing) in message
    assert "not found" in message


def test_require_spawn_executable_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ProcessSpawnError) as exc_info:
        require_spawn_executable(str(tmp_path))
    assert "not a file" in str(exc_info.value)


async def test_start_session_missing_binary_raises_precise_spawn_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing-harness.exe"
    service = RuntimeService(harness_commands={"claude": [str(missing)]})
    with pytest.raises(ProcessSpawnError) as exc_info:
        await service.start_session("claude", "openrouter/test-model", tmp_path)
    assert str(missing) in str(exc_info.value)
