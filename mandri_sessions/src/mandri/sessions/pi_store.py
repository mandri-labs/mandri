import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import BinaryIO

from mandri.core.fs.paths import normalize_fs_path
from mandri.sessions.pi_leaf import read_pi_leaf
from mandri.sessions.pi_paths import (
    pi_session_header_id,
    read_pi_path_index,
    write_pi_path_index,
)
from mandri.sessions.pi_records import read_pi_entry_fields

MAX_METADATA_RECORD_BYTES = 1024 * 1024
logger = logging.getLogger(__name__)


def default_pi_agent_dir() -> Path:
    configured = os.environ.get("PI_CODING_AGENT_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".pi/agent"


def validate_pi_id(native_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", native_id):
        raise ValueError("Invalid Pi session id")
    return native_id


def _timestamp(value: object) -> int:
    if not isinstance(value, str):
        return 0
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    except (ValueError, OverflowError, OSError):
        return 0


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(block["text"])
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


@dataclass(frozen=True)
class PiSessionInfo:
    path: Path
    native_id: str
    cwd: str
    created_at: int
    updated_at: int
    parent_path: str | None = None
    name: str | None = None
    first_message: str | None = None
    model: str | None = None
    thinking_level: str | None = None
    leaf_id: str | None = None
    version: int = 1


@dataclass(frozen=True)
class _CachedSession:
    signature: tuple[int, int, int]
    info: PiSessionInfo | None
    anchor: bytes


class PiSessionStore:
    def __init__(
        self,
        sessions_root: Path | None = None,
        *,
        path_validator: Callable[[Path], Path] | None = None,
    ) -> None:
        self.root = sessions_root or default_pi_agent_dir() / "sessions"
        self.index_root = self.root.parent / ".mandri-session-paths"
        configured = (
            os.environ.get("PI_CODING_AGENT_SESSION_DIR") if sessions_root is None else None
        )
        self.extra_root = Path(configured).expanduser() if configured else None
        self._cache: dict[Path, _CachedSession] = {}
        self._leaf_cache: dict[Path, tuple[tuple[int, int, int], str | None, PiSessionInfo]] = {}
        self._paths: dict[str, Path] = {}
        self._lock = RLock()
        self._path_validator = path_validator

    def fetch(self) -> list[PiSessionInfo]:
        with self._lock:
            inventory = self._inventory()
            registered = {
                path: native_id
                for path, native_id in read_pi_path_index(
                    self.index_root, self._path_validator
                ).items()
                if path not in inventory
            }
            inventory.update(registered)
            self._cache = {path: row for path, row in self._cache.items() if path in inventory}
            self._leaf_cache = {
                path: row for path, row in self._leaf_cache.items() if path in inventory
            }
            rows: dict[str, PiSessionInfo] = {}
            for path in inventory:
                self._checked(path)
                row = self._fetch_path(path)
                if row is None or (path in registered and row.native_id != registered[path]):
                    continue
                previous = rows.get(row.native_id)
                if previous is None or row.updated_at > previous.updated_at:
                    rows[row.native_id] = row
            self._paths = {row.native_id: row.path for row in rows.values()}
            return sorted(
                rows.values(), key=lambda row: (row.updated_at, row.native_id), reverse=True
            )

    def _fetch_path(self, path: Path) -> PiSessionInfo | None:
        cached = self._cache.get(path)
        try:
            stat = path.stat()
            signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if cached is None or cached.signature != signature:
                with path.open("rb") as handle:
                    info, offset = None, 0
                    if cached is not None and self._can_extend(handle, cached, signature):
                        info, offset = cached.info, cached.signature[2]
                    handle.seek(offset)
                    info = self._read(handle, path, int(stat.st_mtime * 1000), info)
                    handle.seek(max(0, stat.st_size - 256))
                    anchor = handle.read(256)
                if info is None and cached is not None and cached.info is not None:
                    logger.debug(
                        "Retaining Pi session %s during native rewrite", cached.info.native_id
                    )
                    return self._known(path, cached)
                cached = _CachedSession(signature, info, anchor)
                self._cache[path] = cached
            return self._selected(cached.info, signature) if cached.info is not None else None
        except FileNotFoundError:
            self._cache.pop(path, None)
            return None
        except OSError:
            logger.debug("Pi session inventory read deferred for %s", path, exc_info=True)
            return self._known(path, cached)

    def _known(self, path: Path, cached: _CachedSession | None) -> PiSessionInfo | None:
        if cached is None:
            return None
        selected = self._leaf_cache.get(path)
        return selected[2] if selected and selected[0] == cached.signature else cached.info

    def _selected(self, info: PiSessionInfo, signature: tuple[int, int, int]) -> PiSessionInfo:
        selected = read_pi_leaf(info.path, info.native_id, signature)
        if selected is None or selected.leaf_id == info.leaf_id:
            self._leaf_cache[info.path] = (signature, info.leaf_id, info)
            return info
        cached = self._leaf_cache.get(info.path)
        if cached is not None and cached[:2] == (signature, selected.leaf_id):
            return cached[2]
        base = replace(info, first_message=None, model=None, thinking_level=None, leaf_id=None)
        result = base
        if selected.leaf_id is not None:
            entries: dict[str, PiSessionInfo] = {}
            with info.path.open("rb") as handle:
                while raw := self._record(handle):
                    try:
                        entry = json.loads(raw)
                    except (ValueError, RecursionError):
                        continue
                    if not isinstance(entry, dict) or entry.get("type") == "session":
                        continue
                    entry_id = entry.get("id")
                    if not isinstance(entry_id, str):
                        continue
                    parent = entry.get("parentId")
                    ancestor = entries.get(parent, base) if isinstance(parent, str) else base
                    entries[entry_id] = self._apply(ancestor, entry)
            result = entries.get(selected.leaf_id, info)
        result = replace(result, name=info.name)
        self._leaf_cache[info.path] = (signature, selected.leaf_id, result)
        return result

    def register(self, native_id: str, path: str | Path) -> None:
        validate_pi_id(native_id)
        candidate = Path(os.path.abspath(Path(path).expanduser()))
        self._checked(candidate)
        self._checked(self.index_root)
        if candidate.exists() and pi_session_header_id(candidate) != native_id:
            raise ValueError("Pi session path does not match its native identity")
        write_pi_path_index(self.index_root, native_id, candidate)

    def resolve(self, native_id: str, project_path: str | None = None) -> Path | None:
        validate_pi_id(native_id)
        with self._lock:
            cached = self._paths.get(native_id)
            if cached is not None and cached.is_file():
                self._checked(cached)
                stat = cached.stat()
                cached_row = self._cache[cached]
                info = cached_row.info
                if (
                    info is not None
                    and (
                        project_path is None
                        or normalize_fs_path(info.cwd) == normalize_fs_path(project_path)
                    )
                    and cached_row.signature == (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                ):
                    return cached
            rows = self.fetch()
            for row in rows:
                if row.native_id == native_id and (
                    project_path is None
                    or normalize_fs_path(row.cwd) == normalize_fs_path(project_path)
                ):
                    return row.path
        return None

    def _inventory(self) -> set[Path]:
        paths: set[Path] = set()
        for root in (self.root, self.extra_root):
            if root is not None:
                self._checked(root)
            if root is None or not root.is_dir():
                continue
            for pattern in ("*.jsonl", "*/*.jsonl"):
                paths.update(root.glob(pattern))
        return paths

    def _checked(self, path: Path) -> Path:
        return self._path_validator(path) if self._path_validator is not None else path

    @staticmethod
    def _record(handle: BinaryIO) -> bytes:
        start = handle.tell()
        raw = handle.readline(MAX_METADATA_RECORD_BYTES + 1)
        if len(raw) <= MAX_METADATA_RECORD_BYTES:
            return raw
        prefix = raw
        while raw and not raw.endswith(b"\n"):
            raw = handle.readline(MAX_METADATA_RECORD_BYTES + 1)
        fields = read_pi_entry_fields(handle, start, handle.tell(), prefix)
        return json.dumps(fields).encode("utf-8") + b"\n"

    @staticmethod
    def _can_extend(
        handle: BinaryIO, cached: _CachedSession, signature: tuple[int, int, int]
    ) -> bool:
        inode, _, size = cached.signature
        if (
            cached.info is None
            or signature[0] != inode
            or signature[2] <= size
            or not cached.anchor.endswith(b"\n")
        ):
            return False
        handle.seek(max(0, size - 256))
        return handle.read(min(size, 256)) == cached.anchor

    @classmethod
    def _read(
        cls, handle: BinaryIO, path: Path, modified: int, info: PiSessionInfo | None = None
    ) -> PiSessionInfo | None:
        if info is not None:
            info = replace(info, updated_at=max(modified, info.created_at))
        while raw := cls._record(handle):
            try:
                entry = json.loads(raw)
            except (ValueError, RecursionError):
                continue
            if not isinstance(entry, dict):
                continue
            if info is None:
                native_id = entry.get("id")
                if entry.get("type") != "session" or not isinstance(native_id, str):
                    return None
                try:
                    validate_pi_id(native_id)
                except ValueError:
                    return None
                created = _timestamp(entry.get("timestamp")) or modified
                cwd, parent, version = (
                    entry.get("cwd"),
                    entry.get("parentSession"),
                    entry.get("version"),
                )
                info = PiSessionInfo(
                    path=path,
                    native_id=native_id,
                    cwd=cwd if isinstance(cwd, str) else "",
                    created_at=created,
                    updated_at=max(created, modified),
                    parent_path=parent if isinstance(parent, str) else None,
                    version=version if isinstance(version, int) else 1,
                )
            else:
                info = cls._apply(info, entry)
        return info

    @staticmethod
    def _apply(info: PiSessionInfo, entry: dict[str, object]) -> PiSessionInfo:
        entry_id = entry.get("id")
        if isinstance(entry_id, str):
            info = replace(info, leaf_id=entry_id)
        kind = entry.get("type")
        if kind == "session_info":
            name = entry.get("name")
            info = replace(info, name=name.strip() or None if isinstance(name, str) else None)
        elif kind == "thinking_level_change" and isinstance(entry.get("thinkingLevel"), str):
            info = replace(info, thinking_level=str(entry["thinkingLevel"]))
        elif kind == "model_change" and isinstance(entry.get("modelId"), str):
            model = str(entry["modelId"])
            provider = entry.get("provider")
            info = replace(info, model=f"{provider}/{model}" if provider else model)
        elif kind == "message" and isinstance(message := entry.get("message"), dict):
            if message.get("role") == "user" and info.first_message is None:
                preview = " ".join(_text(message.get("content")).split())[:200]
                info = replace(info, first_message=preview or None)
            if message.get("role") == "assistant" and isinstance(message.get("model"), str):
                model, provider = message["model"], message.get("provider")
                info = replace(info, model=f"{provider}/{model}" if provider else model)
        return info
