import errno
import os
import sys
from pathlib import Path
from typing import BinaryIO

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


def try_lock(file: BinaryIO) -> bool:
    try:
        if sys.platform == "win32":
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as error:
        if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            return False
        raise


def unlock(file: BinaryIO) -> None:
    if sys.platform == "win32":
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def writer_locked(path: Path) -> bool | None:
    coordination = path.parent / ".coordination.lock"
    try:
        if not path.exists():
            return False
        if not coordination.exists():
            return _inspect_lock(path)
        with coordination.open("r+b") as guard:
            if not try_lock(guard):
                return None
            try:
                return _inspect_lock(path)
            finally:
                unlock(guard)
    except OSError:
        return None


def _inspect_lock(path: Path) -> bool:
    try:
        with path.open("r+b") as file:
            if not try_lock(file):
                return True
            unlock(file)
            return False
    except FileNotFoundError:
        return False


def linux_lock_pids(path: Path) -> set[int]:
    if sys.platform == "win32":
        raise OSError("Linux lock inspection is unavailable")
    stat = path.stat()
    expected = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
    owners = set()
    for line in Path("/proc/locks").read_text().splitlines():
        fields = line.split()
        if (
            len(fields) >= 6
            and fields[1] == "FLOCK"
            and fields[3] == "WRITE"
            and fields[5] == expected
            and fields[4].isdecimal()
        ):
            owners.add(int(fields[4]))
    return owners
