"""Daemon configuration vocabularies and default path resolution."""

import os
from pathlib import Path, PurePath, PurePosixPath

from mandri.core.ids import HarnessKind, ProviderKind

PROVIDER_KINDS: frozenset[str] = frozenset(k.value for k in ProviderKind)
LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama", "lm_studio", "custom"})
HARNESS_KINDS: frozenset[str] = frozenset(k.value for k in HarnessKind)

CLAUDE_MODES: frozenset[str] = frozenset(
    {"default", "manual", "acceptEdits", "plan", "bypassPermissions", "dontAsk", "auto"}
)
CODEX_APPROVAL_POLICIES: frozenset[str] = frozenset({"untrusted", "on-request", "never"})
CODEX_SANDBOX_MODES: frozenset[str] = frozenset(
    {"read-only", "workspace-write", "danger-full-access"}
)


def default_opencode_db_path() -> PurePath:
    if os.name == "nt":
        return _windows_opencode_db_path()
    return _posix_opencode_db_path()


def _windows_opencode_db_path() -> Path:
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "opencode" / "opencode.db"
    canonical = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
    local_app_data = os.environ.get("LOCALAPPDATA")
    legacy = (
        (Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local")
        / "opencode"
        / "opencode.db"
    )
    return legacy if not canonical.is_file() and legacy.is_file() else canonical


def _posix_opencode_db_path() -> PurePosixPath:
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return PurePosixPath(xdg_data_home) / "opencode" / "opencode.db"
    home = PurePosixPath(os.path.expanduser("~"))
    return home / ".local" / "share" / "opencode" / "opencode.db"
