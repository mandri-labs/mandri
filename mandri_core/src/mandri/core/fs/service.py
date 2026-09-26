"""Filesystem listing service: roots, directory listings, bounded trees."""

import asyncio
import os
import string
from pathlib import Path

from mandri.core.fs.errors import (
    FsNotADirectoryError,
    FsNotFoundError,
    FsReadError,
)
from mandri.core.fs.paths import normalize_fs_path
from mandri.core.fs.types import FilesystemEntry, FsNode
from mandri.core.ids import EpochMs, FsPath, ProjectPath
from mandri.core.ports.database import DatabasePort

MAX_TREE_DEPTH = 8

PROJECTS_SQL = (
    "SELECT project_path, MAX(updated_at) AS last_activity FROM session"
    " WHERE deleted = 0 AND project_path != ''"
    " GROUP BY project_path"
    " ORDER BY last_activity DESC, project_path ASC"
)


def _mtime_ms(raw: os.stat_result | None) -> EpochMs | None:
    if raw is None:
        return None
    return EpochMs(int(raw.st_mtime * 1000))


def _stat_entry(item: os.DirEntry[str]) -> os.stat_result | None:
    try:
        return item.stat(follow_symlinks=False)
    except OSError:
        return None


def _entry_from_dir_entry(item: os.DirEntry[str]) -> FilesystemEntry:
    is_dir = item.is_dir(follow_symlinks=True)
    info = _stat_entry(item)
    return FilesystemEntry(
        name=item.name,
        path=normalize_fs_path(item.path),
        is_dir=is_dir,
        size=None if is_dir or info is None else info.st_size,
        modified_at=_mtime_ms(info),
    )


def _sorted_entries(items: list[FilesystemEntry]) -> list[FilesystemEntry]:
    return sorted(items, key=lambda entry: entry.name.lower())


def _list_dir_sync(raw_path: str) -> list[FilesystemEntry]:
    try:
        with os.scandir(raw_path) as iterator:
            return _sorted_entries([_entry_from_dir_entry(item) for item in iterator])
    except FileNotFoundError as error:
        raise FsNotFoundError(f"path not found: {raw_path}") from error
    except NotADirectoryError as error:
        raise FsNotADirectoryError(f"not a directory: {raw_path}") from error
    except OSError as error:
        raise FsReadError(f"cannot list directory {raw_path}: {error}") from error


def _root_entry(raw_path: str, name: str) -> FilesystemEntry:
    normalized = normalize_fs_path(raw_path)
    info: os.stat_result | None = None
    try:
        info = os.stat(raw_path)
    except OSError:
        info = None
    return FilesystemEntry(
        name=name,
        path=normalized,
        is_dir=True,
        size=None,
        modified_at=_mtime_ms(info),
    )


def _roots_sync() -> list[FilesystemEntry]:
    entries: list[FilesystemEntry] = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if os.path.exists(root):
            entries.append(_root_entry(root, root))
    home = Path.home()
    entries.append(_root_entry(str(home), home.name))
    return entries


def _resolved(raw_path: str) -> str:
    return os.path.normcase(os.path.realpath(raw_path))


def _build_node(
    raw_path: str, entry: FilesystemEntry, depth: int, visited: frozenset[str]
) -> FsNode:
    resolved = _resolved(raw_path)
    if resolved in visited:
        return FsNode(entry=entry, children=[])
    next_visited = visited | {resolved}
    children: list[FsNode] = []
    if depth > 0:
        try:
            listed = _list_dir_sync(raw_path)
        except (FsNotFoundError, FsNotADirectoryError, FsReadError):
            listed = []
        children = [_build_node(str(item.path), item, depth - 1, next_visited) for item in listed]
    return FsNode(entry=entry, children=children)


def _tree_sync(raw_path: str, depth: int) -> FsNode:
    normalized = normalize_fs_path(raw_path)
    if not os.path.exists(normalized):
        raise FsNotFoundError(f"path not found: {normalized}")
    if not os.path.isdir(normalized):
        raise FsNotADirectoryError(f"not a directory: {normalized}")
    root_entry = _root_entry(normalized, os.path.basename(normalized) or normalized)
    return _build_node(normalized, root_entry, depth, frozenset())


async def list_project_paths(db: DatabasePort) -> list[ProjectPath]:
    rows = await db.fetch_all(PROJECTS_SQL)
    paths: list[ProjectPath] = []
    seen: set[str] = set()
    for row in rows:
        normalized = normalize_fs_path(str(row["project_path"]))
        if normalized in seen:
            continue
        seen.add(normalized)
        paths.append(ProjectPath(normalized))
    return paths


class FsService:
    async def list_roots(self) -> list[FilesystemEntry]:
        return await asyncio.to_thread(_roots_sync)

    async def list_dir(self, path: FsPath) -> list[FilesystemEntry]:
        return await asyncio.to_thread(_list_dir_sync, str(path))

    async def tree(self, path: FsPath, depth: int) -> FsNode:
        clamped = max(0, min(depth, MAX_TREE_DEPTH))
        return await asyncio.to_thread(_tree_sync, str(path), clamped)
