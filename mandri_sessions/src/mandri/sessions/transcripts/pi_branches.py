import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import BinaryIO

from mandri.sessions.pi_leaf import read_pi_leaf
from mandri.sessions.pi_records import read_pi_entry_fields
from mandri.sessions.transcripts.records import RecordSpan

_PREFIX_BYTES = 65536


@dataclass(frozen=True)
class _Entry:
    span: RecordSpan
    parent_id: str | None


@dataclass
class _Index:
    signature: tuple[int, int, int]
    entries: dict[str, _Entry]
    header: RecordSpan | None
    native_id: str | None
    leaf: str | None
    legacy: list[RecordSpan]
    version: int
    offset: int
    anchor: bytes


class PiBranchIndex:
    def __init__(self) -> None:
        self._cache: dict[Path, _Index] = {}
        self._lock = RLock()

    def spans(self, path: Path) -> list[RecordSpan]:
        with self._lock, path.open("rb") as handle:
            stat = os.fstat(handle.fileno())
            signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            index = self._cache.get(path)
            if index is None or index.signature != signature:
                index = self._update(handle, index, signature)
                self._cache[path] = index
            if index.version < 2:
                return list(index.legacy)
            ancestors = []
            seen = set()
            current = index.leaf
            selected = read_pi_leaf(path, index.native_id, signature)
            if selected is not None and (
                selected.leaf_id is None or selected.leaf_id in index.entries
            ):
                current = selected.leaf_id
            while current is not None and current not in seen:
                seen.add(current)
                entry = index.entries.get(current)
                if entry is None:
                    break
                ancestors.append(entry.span)
                current = entry.parent_id
            return ([index.header] if index.header is not None else []) + list(reversed(ancestors))

    @staticmethod
    def _update(handle: BinaryIO, index: _Index | None, signature: tuple[int, int, int]) -> _Index:
        if index is not None:
            handle.seek(max(0, index.offset - 256))
            if (
                index.signature[0] != signature[0]
                or signature[2] <= index.signature[2]
                or handle.read(min(index.offset, 256)) != index.anchor
            ):
                index = None
        if index is None:
            index = _Index(signature, {}, None, None, None, [], 1, 0, b"")
        handle.seek(index.offset)
        while handle.tell() < signature[2]:
            start = handle.tell()
            prefix = handle.readline(_PREFIX_BYTES)
            chunk = prefix
            while chunk and not chunk.endswith(b"\n"):
                chunk = handle.readline(_PREFIX_BYTES)
            if not chunk:
                break
            span = RecordSpan(start, handle.tell())
            fields = read_pi_entry_fields(handle, start, span.end, prefix)
            if fields.get("type") == "session":
                index.header = span
                native_id = fields.get("id")
                index.native_id = native_id if isinstance(native_id, str) else None
                version = fields.get("version")
                index.version = version if isinstance(version, int) else 1
            elif isinstance(entry_id := fields.get("id"), str):
                parent = fields.get("parentId")
                index.entries[entry_id] = _Entry(span, parent if isinstance(parent, str) else None)
                index.leaf = entry_id
            if index.version < 2:
                index.legacy.append(span)
            index.offset = span.end
        index.signature = signature
        handle.seek(max(0, index.offset - 256))
        index.anchor = handle.read(min(index.offset, 256))
        return index
