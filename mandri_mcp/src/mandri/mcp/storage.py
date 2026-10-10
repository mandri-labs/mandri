import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet


def atomic_write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        if os.name != "nt":
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(content.encode("utf-8") if isinstance(content, str) else content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


class SecretStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._cipher: Fernet | None = None

    def _fernet(self) -> Fernet:
        if self._cipher is None:
            path = self.directory / "key"
            try:
                key = path.read_bytes()
            except FileNotFoundError:
                key = Fernet.generate_key()
                atomic_write(path, key)
            self._cipher = Fernet(key)
        return self._cipher

    def load(self, identity: str) -> dict[str, Any]:
        path = self.directory / f"{identity}.bin"
        if not path.exists():
            return {}
        value: dict[str, Any] = json.loads(self._fernet().decrypt(path.read_bytes()))
        return value

    def save(self, identity: str, value: dict[str, Any]) -> None:
        payload = self._fernet().encrypt(json.dumps(value).encode("utf-8"))
        atomic_write(self.directory / f"{identity}.bin", payload)

    def delete(self, identity: str) -> None:
        (self.directory / f"{identity}.bin").unlink(missing_ok=True)
