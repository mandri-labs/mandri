"""Composition helpers assembling per-harness session backends from config."""

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import httpx
from mandri.core.ids import HarnessKind, SessionId, SessionTitle
from mandri.core.ports.sessions import (
    CheckSessionExistsPort,
    DeleteSessionPort,
    FetchSessionsPort,
    RenameSessionPort,
)
from mandri.core.types.config import SessionsConfig
from mandri.core.types.sessions import Session, SessionStateError
from mandri.sessions.adapters.agy_sessions import AgySessionsAdapter
from mandri.sessions.adapters.claude_fetch_sessions import ClaudeSdkFetchSessionsAdapter
from mandri.sessions.adapters.claude_mutations import (
    ClaudeSdkCheckSessionExistsAdapter,
    ClaudeSdkDeleteSessionAdapter,
    ClaudeSdkRenameSessionAdapter,
)
from mandri.sessions.adapters.codex_fetch_sessions import CodexFetchSessions
from mandri.sessions.adapters.codex_mutations import (
    CodexMutationsAdapter,
    connect_codex_app_server,
)
from mandri.sessions.adapters.opencode_fetch_sessions import OpencodeSqliteFetchSessionsAdapter
from mandri.sessions.adapters.opencode_mutations import (
    OpencodeRestCheckSessionExistsAdapter,
    OpencodeRestDeleteSessionAdapter,
    OpencodeRestRenameSessionAdapter,
)
from mandri.sessions.adapters.pi_sessions import PiSessionsAdapter
from mandri.sessions.agy_profiles import default_agy_root
from mandri.sessions.errors import SessionDeleteError, SessionRenameError

DEFAULT_CODEX_HOME = Path.home() / ".codex"
CODEX_STATE_DB_NAME = "state_5.sqlite"
CODEX_SESSIONS_DIR_NAME = "sessions"


class HarnessSessionsBackend:
    def __init__(
        self,
        harness: HarnessKind,
        fetch: FetchSessionsPort,
        rename: RenameSessionPort | None = None,
        delete: DeleteSessionPort | None = None,
        exists: CheckSessionExistsPort | None = None,
    ) -> None:
        self._harness = harness
        self._fetch = fetch
        self._rename = rename
        self._delete = delete
        self._exists = exists

    def fetch(self) -> list[Session]:
        return self._fetch.fetch()

    def rename(self, session_id: SessionId, title: SessionTitle) -> None:
        if self._rename is None:
            raise SessionRenameError(f"mutations disabled for {self._harness.value}")
        self._rename.rename(session_id, title)

    def delete(self, session_id: SessionId) -> None:
        if self._delete is None:
            raise SessionDeleteError(f"mutations disabled for {self._harness.value}")
        self._delete.delete(session_id)

    def exists(self, session_id: SessionId) -> bool:
        if self._exists is None:
            raise SessionStateError(f"mutations disabled for {self._harness.value}")
        return self._exists.exists(session_id)


@dataclasses.dataclass(frozen=True)
class HarnessBackendResult:
    harness: HarnessKind
    backend: HarnessSessionsBackend | None


def build_sessions_backends(
    config: SessionsConfig,
    default_opencode_db: Path,
) -> Mapping[HarnessKind, HarnessSessionsBackend | None]:
    return {
        HarnessKind.OPENCODE: _build_opencode_backend(config, default_opencode_db),
        HarnessKind.CODEX: _build_codex_backend(config),
        HarnessKind.CLAUDE: _build_claude_backend(config),
        HarnessKind.AGY: _build_agy_backend(config),
        HarnessKind.PI: _build_pi_backend(),
    }


def _build_pi_backend() -> HarnessSessionsBackend:
    adapter = PiSessionsAdapter()
    return HarnessSessionsBackend(HarnessKind.PI, adapter, adapter, adapter, adapter)


def _build_agy_backend(config: SessionsConfig) -> HarnessSessionsBackend:
    root = Path(config.agy_home) if config.agy_home else default_agy_root()
    profiles = Path(config.agy_profiles_dir) if config.agy_profiles_dir else None
    adapter = AgySessionsAdapter(root, profiles)
    return HarnessSessionsBackend(HarnessKind.AGY, adapter, delete=adapter, exists=adapter)


def _build_opencode_backend(
    sessions: SessionsConfig, default_db: Path
) -> HarnessSessionsBackend | None:
    db_path = Path(sessions.opencode_db_path) if sessions.opencode_db_path else default_db
    if not db_path.is_file():
        return None
    fetch: FetchSessionsPort = OpencodeSqliteFetchSessionsAdapter(db_path)
    rename: RenameSessionPort | None = None
    delete: DeleteSessionPort | None = None
    exists: CheckSessionExistsPort | None = None
    if sessions.opencode_base_url is not None:
        client = httpx.AsyncClient(base_url=sessions.opencode_base_url)
        rename = OpencodeRestRenameSessionAdapter(client)
        delete = OpencodeRestDeleteSessionAdapter(client)
        exists = OpencodeRestCheckSessionExistsAdapter(client)
    return HarnessSessionsBackend(HarnessKind.OPENCODE, fetch, rename, delete, exists)


def _build_codex_backend(sessions: SessionsConfig) -> HarnessSessionsBackend | None:
    codex_home = Path(sessions.codex_home) if sessions.codex_home else DEFAULT_CODEX_HOME
    if not codex_home.is_dir():
        return None
    fetch = CodexFetchSessions(
        codex_home / CODEX_STATE_DB_NAME,
        codex_home / CODEX_SESSIONS_DIR_NAME,
    )
    mutations = CodexMutationsAdapter(connect_codex_app_server)
    return HarnessSessionsBackend(HarnessKind.CODEX, fetch, mutations, mutations, mutations)


def _build_claude_backend(sessions: SessionsConfig) -> HarnessSessionsBackend:
    config_dir = sessions.claude_config_dir
    return HarnessSessionsBackend(
        HarnessKind.CLAUDE,
        ClaudeSdkFetchSessionsAdapter(config_dir),
        ClaudeSdkRenameSessionAdapter(config_dir),
        ClaudeSdkDeleteSessionAdapter(config_dir),
        ClaudeSdkCheckSessionExistsAdapter(config_dir),
    )
