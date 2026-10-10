import os
import stat
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO

from mandri.sessions.fork_lineage import RolloutPrefix, incompatible


def _home(destination: Path) -> Path:
    for parent in destination.parents:
        if parent.name == ".codex":
            relative = destination.relative_to(parent)
            if relative.parts[0] != "sessions":
                break
            return parent
    raise incompatible()


def _parents(root: Path, path: Path, created: list[Path]) -> None:
    if root.resolve(strict=True) != root or not root.is_dir():
        raise incompatible()
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        try:
            current.mkdir(mode=0o700)
            created.append(current)
        except FileExistsError:
            pass
        if not stat.S_ISDIR(current.lstat().st_mode):
            raise incompatible()


def _rollback(files: list[tuple[Path, tuple[int, int]]], directories: list[Path]) -> None:
    for path, signature in reversed(files):
        try:
            metadata = path.lstat()
            if (metadata.st_dev, metadata.st_ino) == signature:
                path.unlink()
        except FileNotFoundError:
            pass
    for path in reversed(directories):
        with suppress(OSError):
            path.rmdir()


def copy_bundle(
    destination: Path,
    prefixes: tuple[RolloutPrefix, ...],
    selected_copy: Callable[[BinaryIO], None],
    verify: Callable[[], None],
    *,
    sandboxed: bool = False,
) -> None:
    verify()
    files = []
    directories: list[Path] = []
    copies = [(destination, selected_copy)]
    try:
        if prefixes:
            home = _home(destination)
            for prefix in prefixes:
                target = home / prefix.relative
                _parents(home, target.parent, directories)
                copies.append((target, prefix.copy))
        for path, copy in copies:
            if sandboxed and path.parent.resolve(strict=True) != path.parent:
                raise incompatible()
            descriptor = os.open(
                path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | (getattr(os, "O_NOFOLLOW", 0) if sandboxed else 0),
                0o600,
            )
            with os.fdopen(descriptor, "wb") as output:
                metadata = os.fstat(output.fileno())
                files.append((path, (metadata.st_dev, metadata.st_ino)))
                copy(output)
                output.flush()
                os.fsync(output.fileno())
        verify()
    except BaseException:
        _rollback(files, directories)
        raise
