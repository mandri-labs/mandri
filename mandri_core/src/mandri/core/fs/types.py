"""Read-only filesystem browsing domain types."""

from __future__ import annotations

import dataclasses

from mandri.core.ids import EpochMs, FsPath


@dataclasses.dataclass(frozen=True)
class FilesystemEntry:
    name: str
    path: FsPath
    is_dir: bool
    size: int | None
    modified_at: EpochMs | None


@dataclasses.dataclass(frozen=True)
class FsNode:
    entry: FilesystemEntry
    children: list[FsNode]
