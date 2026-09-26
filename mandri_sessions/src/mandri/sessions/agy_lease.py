from pathlib import Path
from typing import BinaryIO

from mandri.sessions.agy_profiles import validate_agy_id
from mandri.sessions.errors import SessionRunningError
from mandri.sessions.ownership.file_lock import try_lock, unlock


class AgyConversationLease:
    def __init__(self, canonical_root: Path, native_id: str) -> None:
        self.path = (
            canonical_root / "antigravity-cli/.mandri-locks" / f"{validate_agy_id(native_id)}.lock"
        )
        self._file: BinaryIO | None = None

    def acquire(self) -> None:
        if self._file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            if not try_lock(handle):
                raise SessionRunningError("Antigravity conversation already has a Mandri writer")
        except BaseException:
            handle.close()
            raise
        self._file = handle

    def release(self) -> None:
        handle, self._file = self._file, None
        if handle is not None:
            try:
                unlock(handle)
            finally:
                handle.close()
