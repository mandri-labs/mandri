import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, BinaryIO

from mandri.core.codex_versions import CODEX_FORK_VERSIONS
from mandri.core.types.execution import ProtectionError
from mandri.sessions.ownership.file_lock import try_lock, unlock

MAX_BYTES = 256 * 1024 * 1024
MAX_LINE = 16 * 1024 * 1024
MAX_DEPTH = 32
MAX_LOOKUP_ENTRIES = 100_000
MAX_ORDINAL = (1 << 64) - 1
_NATIVE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_ROLLOUT_NAME = re.compile(r"rollout-(\d{4})-(\d{2})-(\d{2})T.+\.jsonl")


def incompatible() -> ProtectionError:
    return ProtectionError(
        "session_transition_unsupported", "The selected native history cannot be forked"
    )


def identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns


def validate_path(home: Path, path: Path) -> None:
    try:
        relative = path.relative_to(home)
        if not relative.parts or relative.parts[0] not in {"sessions", "archived_sessions"}:
            raise incompatible()
        current = home
        if current.is_symlink() or current.resolve(strict=True) != current:
            raise incompatible()
        for index, part in enumerate(relative.parts):
            current = current / part
            metadata = current.lstat()
            expected = stat.S_ISREG if index == len(relative.parts) - 1 else stat.S_ISDIR
            if stat.S_ISLNK(metadata.st_mode) or not expected(metadata.st_mode):
                raise incompatible()
    except (OSError, ValueError):
        raise incompatible() from None


def target_relative(path: Path) -> Path:
    match = _ROLLOUT_NAME.fullmatch(path.name)
    try:
        if match is None:
            raise ValueError
        year, month, day = match.groups()
        date(int(year), int(month), int(day))
        return Path("sessions", year, month, day, path.name)
    except ValueError:
        raise incompatible() from None


@dataclass(frozen=True)
class HistoryPosition:
    rollout_id: str
    end_ordinal: int
    end_bytes: int


def history_position(value: Any) -> HistoryPosition | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise incompatible()
    native_id = value.get("thread_id")
    ordinal = value.get("end_ordinal_exclusive")
    offset = value.get("end_byte_offset")
    if (
        not isinstance(native_id, str)
        or _NATIVE_ID.fullmatch(native_id) is None
        or type(ordinal) is not int
        or not 0 < ordinal <= MAX_ORDINAL
        or type(offset) is not int
        or not 0 < offset <= MAX_BYTES
    ):
        raise incompatible()
    return HistoryPosition(native_id, ordinal, offset)


def _resolve(home: Path, native_id: str) -> Path:
    candidates = []
    visited = 0
    for directory in ("sessions", "archived_sessions"):
        root = home / directory
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise incompatible()
        for parent, directories, files in os.walk(root, followlinks=False):
            visited += len(directories) + len(files) + 1
            if visited > MAX_LOOKUP_ENTRIES:
                raise incompatible()
            directories[:] = [
                name for name in directories if not (Path(parent) / name).is_symlink()
            ]
            for name in files:
                if name.startswith("rollout-") and name.endswith(f"-{native_id}.jsonl"):
                    path = Path(parent) / name
                    validate_path(home, path)
                    target_relative(path)
                    candidates.append(path)
    if len(candidates) != 1:
        raise incompatible()
    return candidates[0]


@contextmanager
def _writer_lease(home: Path, native_id: str) -> Iterator[None]:
    directory = home / "thread-writer-locks"
    if directory.is_symlink():
        raise incompatible()
    directory.mkdir(mode=0o700, exist_ok=True)
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise incompatible()
    path = directory / f"{native_id}.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "r+b") as lease:
        if not stat.S_ISREG(os.fstat(lease.fileno()).st_mode):
            raise incompatible()
        if not try_lock(lease):
            raise ProtectionError("session_writer_conflict", "The native history has a writer")
        try:
            yield
        finally:
            unlock(lease)


def _inspect_prefix(handle: BinaryIO, position: HistoryPosition) -> tuple[bytes, dict[str, Any]]:
    digest = hashlib.sha256()
    remaining = position.end_bytes
    expected = None
    payload = None
    while remaining:
        line = handle.readline(min(MAX_LINE + 1, remaining))
        if not line or len(line) > MAX_LINE or not line.endswith(b"\n"):
            raise incompatible()
        remaining -= len(line)
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError, RecursionError):
            raise incompatible() from None
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("type"), str)
            or type(row.get("ordinal")) is not int
            or not 0 <= row["ordinal"] < MAX_ORDINAL
            or (expected is not None and row["ordinal"] != expected)
        ):
            raise incompatible()
        if payload is None:
            payload = row.get("payload")
            if (
                row["type"] != "session_meta"
                or not isinstance(payload, dict)
                or not isinstance(payload.get("id"), str)
                or _NATIVE_ID.fullmatch(payload["id"]) is None
                or payload.get("cli_version") not in CODEX_FORK_VERSIONS
                or payload.get("history_mode") != "paginated"
            ):
                raise incompatible()
            base = history_position(payload.get("history_base"))
            if row["ordinal"] != (base.end_ordinal if base is not None else 0):
                raise incompatible()
        elif row["type"] == "session_meta":
            raise incompatible()
        expected = row["ordinal"] + 1
        digest.update(line)
    if expected != position.end_ordinal or payload is None:
        raise incompatible()
    return digest.digest(), payload


@dataclass(frozen=True)
class RolloutPrefix:
    path: Path
    relative: Path
    length: int
    _home: Path = field(repr=False)
    _handle: BinaryIO = field(repr=False)
    _signature: tuple[int, int, int, int] = field(repr=False)
    _digest: bytes = field(repr=False)

    def verify(self) -> None:
        if self._handle.closed:
            raise incompatible()
        validate_path(self._home, self.path)
        if (
            identity(self.path.lstat()) != self._signature
            or identity(os.fstat(self._handle.fileno())) != self._signature
        ):
            raise incompatible()

    def copy(self, output: BinaryIO) -> None:
        self.verify()
        digest = hashlib.sha256()
        self._handle.seek(0)
        remaining = self.length
        while remaining:
            chunk = self._handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise incompatible()
            digest.update(chunk)
            output.write(chunk)
            remaining -= len(chunk)
        if digest.digest() != self._digest:
            raise incompatible()
        self.verify()


def load_lineage(
    home: Path,
    selected: Path,
    payload: dict[str, Any],
    selected_bytes: int,
    stack: ExitStack,
) -> tuple[RolloutPrefix, ...]:
    position = history_position(payload.get("history_base"))
    if position is None:
        return ()
    if payload.get("history_mode") != "paginated":
        raise incompatible()
    selected_id = selected.name.removesuffix(".jsonl").rsplit("-", 5)
    if len(selected_id) != 6:
        raise incompatible()
    seen = {"-".join(selected_id[-5:])}
    prefixes: list[RolloutPrefix] = []
    leased = set()
    total = selected_bytes
    while position is not None:
        if position.rollout_id in seen or len(prefixes) >= MAX_DEPTH:
            raise incompatible()
        seen.add(position.rollout_id)
        total += position.end_bytes
        if total > MAX_BYTES:
            raise incompatible()
        if position.rollout_id not in leased:
            stack.enter_context(_writer_lease(home, position.rollout_id))
            leased.add(position.rollout_id)
        path = _resolve(home, position.rollout_id)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        handle = stack.enter_context(os.fdopen(descriptor, "rb"))
        signature = identity(os.fstat(handle.fileno()))
        if position.end_bytes > signature[2]:
            raise incompatible()
        digest, metadata = _inspect_prefix(handle, position)
        logical_id = metadata["id"]
        if logical_id not in leased:
            stack.enter_context(_writer_lease(home, logical_id))
            leased.add(logical_id)
        prefix = RolloutPrefix(
            path, target_relative(path), position.end_bytes, home, handle, signature, digest
        )
        prefix.verify()
        prefixes.append(prefix)
        position = history_position(metadata.get("history_base"))
    return tuple(prefixes)
