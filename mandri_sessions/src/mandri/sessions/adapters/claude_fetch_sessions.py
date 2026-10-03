"""Claude session fetch adapter backed by claude-agent-sdk sync functions."""

import os
import time

from claude_agent_sdk import SDKSessionInfo, list_sessions
from mandri.core.ids import (
    EpochMs,
    HarnessKind,
    HarnessSessionId,
    ProjectPath,
    SessionId,
    SessionState,
    SessionTitle,
)
from mandri.core.ports.sessions import FetchSessionsPort
from mandri.core.types.sessions import Session
from mandri.sessions.errors import (
    AgentBinaryNotFoundError,
    SessionFetchError,
    SessionParseError,
)


def apply_config_dir(config_dir: str | None) -> None:
    """Point the SDK at an isolated config directory; None keeps the current env."""
    if config_dir is not None:
        os.environ["CLAUDE_CONFIG_DIR"] = config_dir


class ClaudeSdkFetchSessionsAdapter(FetchSessionsPort):
    def __init__(self, config_dir: str | None = None) -> None:
        apply_config_dir(config_dir)

    def fetch(self) -> list[Session]:
        try:
            infos = list_sessions()
        except FileNotFoundError as error:
            raise AgentBinaryNotFoundError(str(error)) from error
        except OSError as error:
            raise SessionFetchError(str(error)) from error
        except (KeyError, TypeError, ValueError) as error:
            raise SessionParseError(str(error)) from error
        synced_at = EpochMs(int(time.time() * 1000))
        sessions = [self._to_session(info, synced_at) for info in infos]
        sessions.sort(key=lambda session: session.updated_at, reverse=True)
        return sessions

    @staticmethod
    def _to_session(info: SDKSessionInfo, synced_at: EpochMs) -> Session:
        return Session(
            id=SessionId(info.session_id),
            harness=HarnessKind.CLAUDE,
            native_id=HarnessSessionId(info.session_id),
            native_title=SessionTitle(info.custom_title or info.first_prompt or "untitled"),
            title_overlay=None,
            project_path=ProjectPath(info.cwd or ""),
            created_at=EpochMs(
                info.created_at if info.created_at is not None else info.last_modified
            ),
            updated_at=EpochMs(info.last_modified),
            state=SessionState.DISCOVERED,
            model=None,
            gateway_route_id=None,
            deleted=False,
            last_synced_at=synced_at,
        )
