import os
import sys
from pathlib import Path
from typing import IO

from mandri.runtime.errors.docker import DockerExecutionError

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


class StateLease:
    def __init__(self, path: Path) -> None:
        self._file: IO[str] | None = None
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        stream = os.fdopen(fd, "r+", encoding="utf-8")
        try:
            if sys.platform == "win32":
                if os.fstat(fd).st_size == 0:
                    stream.write("0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise DockerExecutionError(
                "native_state_in_use", "Another execution owns this native state"
            ) from error
        self._file = stream

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
