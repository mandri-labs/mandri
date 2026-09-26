import json
import os
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4


def read_pi_path_index(
    root: Path, validator: Callable[[Path], Path] | None = None
) -> dict[Path, str]:
    if validator is not None:
        root = validator(root)
    paths = {}
    for entry in root.glob("*.json"):
        if validator is not None:
            entry = validator(entry)
        try:
            with entry.open("rb") as handle:
                record = json.loads(handle.read(65536))
            if not isinstance(record, dict) or record.get("native_id") != entry.stem:
                continue
            path = record.get("path")
            if isinstance(path, str) and Path(path).is_absolute():
                paths[Path(path)] = entry.stem
        except (OSError, ValueError, RecursionError):
            continue
    return paths


def write_pi_path_index(root: Path, native_id: str, path: Path) -> None:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = root / f"{native_id}.json"
    temporary = root / f".{native_id}-{uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"native_id": native_id, "path": str(path)}, handle)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def pi_session_header_id(path: Path) -> str | None:
    with path.open("rb") as handle:
        remaining = 1024 * 1024
        while remaining:
            raw = handle.readline(remaining)
            if not raw:
                return None
            remaining -= len(raw)
            try:
                entry = json.loads(raw)
            except (ValueError, RecursionError):
                continue
            if not isinstance(entry, dict):
                continue
            if entry.get("type") != "session":
                return None
            native_id = entry.get("id")
            return native_id if isinstance(native_id, str) else None
    return None
