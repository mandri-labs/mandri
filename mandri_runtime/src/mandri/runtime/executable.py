"""Harness executable resolution shared by detection and spawn."""

import shutil
from pathlib import Path

from mandri.runtime.errors import ProcessSpawnError
from mandri.sessions.agy_binary import find_agy_binary


def resolve_executable(name: str) -> Path | None:
    if name in ("agy", "agy.exe"):
        return find_agy_binary()
    found = shutil.which(name)
    if found is None:
        return None
    return Path(found)


def require_spawn_executable(argv0: str) -> str:
    if _path_like(argv0):
        return _require_existing_file(argv0)
    resolved = resolve_executable(argv0)
    if resolved is None:
        raise ProcessSpawnError(f"executable not found on PATH: {argv0!r}")
    return str(resolved)


def _path_like(value: str) -> bool:
    return "/" in value or "\\" in value


def _require_existing_file(value: str) -> str:
    path = Path(value)
    if not path.exists():
        raise ProcessSpawnError(f"executable not found: {path}")
    if not path.is_file():
        raise ProcessSpawnError(f"executable is not a file: {path}")
    return str(path)
