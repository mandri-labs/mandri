import os
import stat
from pathlib import Path

from mandri.runtime.errors.docker import DockerExecutionError


def validate_git_directory(root: Path) -> None:
    git = root / ".git"
    try:
        metadata = git.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(metadata.st_mode):
        return
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 4096:
        raise DockerExecutionError("workspace_unavailable", "Invalid Git administrative entry")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            git, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise ValueError
        raw = os.read(descriptor, 4097)
        current = git.lstat()
        if len(raw) > 4096 or (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ) != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
            raise ValueError
        value = raw.decode("utf-8").strip()
    except (OSError, ValueError):
        raise DockerExecutionError(
            "workspace_unavailable", "Git administrative state changed"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if value.startswith("gitdir:"):
        target = (root / value.split(":", 1)[1].strip()).resolve()
        if not target.is_relative_to(root):
            raise DockerExecutionError(
                "workspace_unavailable", "Git administrative state lies outside the workspace"
            )
