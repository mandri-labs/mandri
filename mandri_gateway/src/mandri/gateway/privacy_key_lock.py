import contextlib
import os
import stat
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from mandri.core.types.execution import ProtectionError

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


def _try_lock(handle: BinaryIO) -> bool:
    try:
        if sys.platform == "win32":
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(handle: BinaryIO) -> None:
    if sys.platform == "win32":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def key_creation_lock(installation: Path) -> Iterator[None]:
    try:
        installation.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            installation / ".privacy-key.lock",
            os.O_RDWR | os.O_CREAT,
            0o600,
        )
        with os.fdopen(descriptor, "r+b") as handle:
            metadata = os.fstat(handle.fileno())
            private_owner = True
            if sys.platform != "win32":
                private_owner = not metadata.st_mode & 0o077 and metadata.st_uid == os.geteuid()
            if not stat.S_ISREG(metadata.st_mode) or not private_owner:
                raise ProtectionError("privacy_key_unavailable", "Privacy key lock is unavailable")
            if metadata.st_size == 0:
                handle.write(b"\0")
                handle.flush()
            deadline = time.monotonic() + 5
            while not _try_lock(handle):
                if time.monotonic() >= deadline:
                    raise ProtectionError(
                        "privacy_key_unavailable", "Privacy key initialization is busy"
                    )
                time.sleep(0.02)
            try:
                yield
            finally:
                _unlock(handle)
    except OSError:
        raise ProtectionError(
            "privacy_key_unavailable", "Privacy key lock is unavailable"
        ) from None
