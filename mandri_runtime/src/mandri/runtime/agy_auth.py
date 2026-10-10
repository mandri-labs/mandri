import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mandri.runtime import agy_credentials_linux, agy_credentials_macos, agy_credentials_windows
from mandri.sessions.agy_profiles import default_agy_root


def _stored_credential() -> str | None:
    if sys.platform == "win32":
        return agy_credentials_windows.read_credential()
    if sys.platform == "darwin":
        return agy_credentials_macos.read_credential()
    if sys.platform == "linux":
        return agy_credentials_linux.read_credential()
    raise OSError("Unsupported Antigravity credential store")


def _profile_root(command: Sequence[str], cwd: Path) -> Path:
    root = default_agy_root()
    for index, argument in enumerate(command):
        if argument == "--gemini_dir":
            root = Path(command[index + 1])
        elif argument.startswith("--gemini_dir="):
            root = Path(argument.partition("=")[2])
    return root if root.is_absolute() else cwd / root


def _authenticated(command: Sequence[str], cwd: Path) -> bool:
    try:
        store = _profile_root(command, cwd) / "antigravity-cli"
        raw = (
            None
            if (store / "cache/antigravity-keyring-unavailable").exists()
            else _stored_credential()
        )
        if raw is None:
            path = store / "antigravity-oauth-token"
            if path.stat().st_size > 65536:
                return False
            raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return False
        if "auth_method" in payload:
            if payload["auth_method"] != "consumer":
                return False
            token = payload.get("token")
        else:
            token = payload
        if not isinstance(token, dict):
            return False
        for field in ("access_token", "refresh_token"):
            value = token.get(field)
            if not isinstance(value, str) or not value.strip():
                return False
        if token.get("token_type") != "Bearer":
            return False
        expiry = token.get("expiry")
        if not isinstance(expiry, str):
            return False
        deadline = datetime.fromisoformat(expiry)
        return deadline.tzinfo is not None and deadline > datetime.now(UTC) + timedelta(seconds=60)
    except Exception:
        return False


async def require_agy_authentication(command: Sequence[str], cwd: str | Path) -> None:
    if not await asyncio.to_thread(_authenticated, command, Path(cwd)):
        raise PermissionError("Antigravity CLI authentication required")
