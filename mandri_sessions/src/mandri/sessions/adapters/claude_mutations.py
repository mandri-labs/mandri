"""Claude session mutation adapters backed by claude-agent-sdk sync functions."""

from claude_agent_sdk import delete_session, get_session_info, rename_session
from mandri.core.ids import SessionId, SessionTitle
from mandri.core.ports.sessions import (
    CheckSessionExistsPort,
    DeleteSessionPort,
    RenameSessionPort,
)
from mandri.sessions.adapters.claude_fetch_sessions import apply_config_dir
from mandri.sessions.errors import (
    SessionDeleteError,
    SessionFetchError,
    SessionNotFoundError,
    SessionRenameError,
)


class ClaudeSdkRenameSessionAdapter(RenameSessionPort):
    def __init__(self, config_dir: str | None = None) -> None:
        apply_config_dir(config_dir)

    def rename(self, session_id: SessionId, title: SessionTitle) -> None:
        try:
            rename_session(session_id, title)
        except ValueError as error:
            raise SessionRenameError(f"{session_id}: {error}") from error
        except FileNotFoundError as error:
            raise SessionNotFoundError(f"{session_id}: {error}") from error
        except OSError as error:
            raise SessionRenameError(f"{session_id}: {error}") from error


class ClaudeSdkDeleteSessionAdapter(DeleteSessionPort):
    def __init__(self, config_dir: str | None = None) -> None:
        apply_config_dir(config_dir)

    def delete(self, session_id: SessionId) -> None:
        try:
            delete_session(session_id)
        except (FileNotFoundError, ValueError) as error:
            raise SessionDeleteError(f"{session_id}: {error}") from error
        except OSError as error:
            raise SessionDeleteError(f"{session_id}: {error}") from error


class ClaudeSdkCheckSessionExistsAdapter(CheckSessionExistsPort):
    def __init__(self, config_dir: str | None = None) -> None:
        apply_config_dir(config_dir)

    def exists(self, session_id: SessionId) -> bool:
        try:
            return get_session_info(session_id) is not None
        except OSError as error:
            raise SessionFetchError(f"{session_id}: {error}") from error
