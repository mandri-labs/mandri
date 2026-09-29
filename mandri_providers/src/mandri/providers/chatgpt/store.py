"""Owner-only token store for ChatGPT provider credentials."""

import contextlib
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from mandri.providers.chatgpt.oauth import TokenSet

_DIRECTORY = "chatgpt"
_MODE = 0o600


def token_path(base_dir: Path, provider_name: str) -> Path:
    identity = hashlib.sha256(provider_name.encode("utf-8")).hexdigest()
    return base_dir / _DIRECTORY / f"{identity}.json"


class ChatGptTokenStore:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    def migrate(self, names: list[str]) -> None:
        legacy: dict[Path, list[str]] = {}
        for name in names:
            slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "provider"
            path = self._base_dir / _DIRECTORY / f"{slug[:64]}.json"
            legacy.setdefault(path, []).append(name)
        for source, owners in legacy.items():
            if len(owners) != 1 or not source.is_file():
                continue
            target = token_path(self._base_dir, owners[0])
            if not target.exists():
                source.rename(target)

    def load(self, provider_name: str) -> TokenSet | None:
        path = token_path(self._base_dir, provider_name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        access = payload.get("access_token")
        refresh = payload.get("refresh_token")
        expires = payload.get("expires_at_ms")
        if not isinstance(access, str) or not access:
            return None
        return TokenSet(
            access_token=access,
            refresh_token=refresh if isinstance(refresh, str) else "",
            expires_at_ms=expires if type(expires) is int else 0,
        )

    def save(self, provider_name: str, tokens: TokenSet) -> None:
        path = token_path(self._base_dir, provider_name)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        document = json.dumps(
            {
                "access_token": tokens.access_token,
                "refresh_token": tokens.refresh_token,
                "expires_at_ms": tokens.expires_at_ms,
            },
            indent=2,
            sort_keys=True,
        )
        handle_fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            if os.name != "nt":
                os.fchmod(handle_fd, _MODE)
            with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
                handle.write(document)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
        if os.name != "nt":
            os.chmod(path, _MODE)

    def clear(self, provider_name: str) -> None:
        with contextlib.suppress(OSError):
            token_path(self._base_dir, provider_name).unlink()
