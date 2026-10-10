import base64
import hashlib
import os
import secrets
import stat
import sys
from pathlib import Path
from typing import cast

import keyring
from keyring.backend import KeyringBackend
from keyring.backends.chainer import ChainerBackend
from keyring.errors import KeyringError
from mandri.core.types.execution import ProtectionError
from mandri.gateway.privacy_key_lock import key_creation_lock

_OS_BACKENDS = frozenset(
    {
        "keyring.backends.SecretService",
        "keyring.backends.kwallet",
        "keyring.backends.macOS",
        "keyring.backends.Windows",
    }
)


class FilePrivacyKey:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self, *, create: bool = False) -> bytes:
        if sys.platform == "win32":
            raise ProtectionError(
                "privacy_key_unavailable", "Use the OS credential store on Windows"
            )
        else:
            if create:
                try:
                    self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    pass
                except OSError:
                    raise ProtectionError(
                        "privacy_key_unavailable", "Privacy key cannot be created"
                    ) from None
                else:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(secrets.token_bytes(32))
                        handle.flush()
                        os.fsync(handle.fileno())
            try:
                fd = os.open(self._path, os.O_RDONLY)
                with os.fdopen(fd, "rb") as handle:
                    metadata = os.fstat(handle.fileno())
                    if (
                        not stat.S_ISREG(metadata.st_mode)
                        or metadata.st_mode & 0o077
                        or metadata.st_uid != os.geteuid()
                    ):
                        raise ProtectionError(
                            "privacy_key_unavailable",
                            "Privacy key must be a private owner-only file",
                        )
                    key = handle.read(33)
            except OSError:
                raise ProtectionError(
                    "privacy_key_unavailable", "Privacy key is unavailable"
                ) from None
            if len(key) != 32:
                raise ProtectionError(
                    "privacy_key_unavailable", "Privacy key has an invalid length"
                )
            return key


class KeyringPrivacyKey:
    def __init__(self, installation: Path, service: str = "mandri-privacy") -> None:
        self._service = service
        self._installation = installation.resolve()
        self._account = hashlib.sha256(str(self._installation).encode()).hexdigest()

    def load(self, *, create: bool = False) -> bytes:
        if create:
            with key_creation_lock(self._installation):
                return self._load(create=True)
        return self._load(create=False)

    def _load(self, *, create: bool) -> bytes:
        try:
            backend: KeyringBackend | None = keyring.get_keyring()
            if type(backend).__module__ == "keyring.backends.chainer":
                backend = next(
                    (
                        candidate
                        for candidate in cast(ChainerBackend, backend).backends
                        if type(candidate).__module__ in _OS_BACKENDS
                    ),
                    None,
                )
            if backend is None or type(backend).__module__ not in _OS_BACKENDS:
                raise ProtectionError(
                    "privacy_key_unavailable", "A supported OS credential store is required"
                )
            encoded = backend.get_password(self._service, self._account)
            if encoded is None and create:
                encoded = base64.b64encode(secrets.token_bytes(32)).decode("ascii")
                backend.set_password(self._service, self._account, encoded)
            if encoded is None:
                raise ProtectionError("privacy_key_unavailable", "Privacy key is unavailable")
            key = base64.b64decode(encoded, validate=True)
        except (KeyringError, ValueError, OSError):
            raise ProtectionError(
                "privacy_key_unavailable", "The local credential store is unavailable or locked"
            ) from None
        if len(key) != 32:
            raise ProtectionError("privacy_key_unavailable", "Privacy key has an invalid length")
        return key
