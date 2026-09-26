"""Native session identity capture from harness launch channels."""

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

import httpx
from mandri.core.ids import HarnessSessionId
from mandri.runtime.control.errors import ControlTransportError
from mandri.runtime.errors import OpencodeSessionMissingError

_OPENCODE_PROBE_TIMEOUT = httpx.Timeout(2.0, connect=1.0)
_OPENCODE_CREATE_TIMEOUT = httpx.Timeout(30.0, connect=1.0)
_OPENCODE_STARTUP_SECONDS = 60.0
_OPENCODE_LAUNCH_DELAY_S = 0.25


def claude_native_id(frame: Mapping[str, Any]) -> HarnessSessionId | None:
    if frame.get("type") != "system" or frame.get("subtype") != "init":
        return None
    session_id = frame.get("session_id")
    if not isinstance(session_id, str):
        return None
    return HarnessSessionId(session_id)


def codex_native_id(result: Mapping[str, Any]) -> HarnessSessionId | None:
    thread = result.get("thread")
    if not isinstance(thread, dict):
        return None
    thread_id = thread.get("id")
    if not isinstance(thread_id, str):
        return None
    return HarnessSessionId(thread_id)


async def capture_opencode_session_id(
    port: int, auth: tuple[str, str] | None = None
) -> HarnessSessionId | None:
    url = f"http://127.0.0.1:{port}/session"
    if await _list_opencode_session_ids(url, auth) is None:
        return None
    try:
        async with httpx.AsyncClient(
            timeout=_OPENCODE_CREATE_TIMEOUT, auth=auth, trust_env=False
        ) as client:
            response = await client.post(url, json={})
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return None
    except (httpx.HTTPError, OSError) as error:
        raise ControlTransportError("OpenCode session creation could not be confirmed") from error
    if response.status_code >= 400:
        raise ControlTransportError(f"OpenCode session creation rejected: {response.status_code}")
    try:
        payload = response.json()
    except ValueError:
        raise ControlTransportError("OpenCode session creation returned invalid data") from None
    if not isinstance(payload, dict):
        raise ControlTransportError("OpenCode session creation returned invalid data")
    session_id = payload.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ControlTransportError("OpenCode session creation returned no identity")
    return HarnessSessionId(session_id)


async def await_opencode_session_id(
    port: int, alive: Callable[[], bool] | None = None, auth: tuple[str, str] | None = None
) -> HarnessSessionId | None:
    deadline = asyncio.get_running_loop().time() + _OPENCODE_STARTUP_SECONDS
    while alive is None or alive():
        if asyncio.get_running_loop().time() >= deadline:
            raise ControlTransportError("OpenCode did not become ready before startup expired")
        session_id = (
            await capture_opencode_session_id(port, auth)
            if auth
            else await capture_opencode_session_id(port)
        )
        if session_id is not None:
            return session_id
        await asyncio.sleep(_OPENCODE_LAUNCH_DELAY_S)
    return None


async def verify_opencode_session_id(
    port: int,
    session_id: str,
    alive: Callable[[], bool] | None = None,
    auth: tuple[str, str] | None = None,
) -> HarnessSessionId:
    """Return the session id once the existing opencode conversation is listed by the server."""
    url = f"http://127.0.0.1:{port}/session"
    deadline = asyncio.get_running_loop().time() + _OPENCODE_STARTUP_SECONDS
    while alive is None or alive():
        if asyncio.get_running_loop().time() >= deadline:
            raise ControlTransportError("OpenCode did not become ready before startup expired")
        sessions = (
            await _list_opencode_session_ids(url, auth)
            if auth
            else await _list_opencode_session_ids(url)
        )
        if sessions is not None:
            if session_id in sessions:
                return HarnessSessionId(session_id)
            raise OpencodeSessionMissingError(session_id)
        await asyncio.sleep(_OPENCODE_LAUNCH_DELAY_S)
    raise ControlTransportError(
        f"opencode process exited before session {session_id} was available"
    )


async def _list_opencode_session_ids(
    url: str, auth: tuple[str, str] | None = None
) -> set[str] | None:
    try:
        async with httpx.AsyncClient(
            timeout=_OPENCODE_PROBE_TIMEOUT, auth=auth, trust_env=False
        ) as client:
            response = await client.get(url)
    except (httpx.HTTPError, OSError):
        return None
    if response.status_code >= 400:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, list):
        return None
    return {
        item["id"] for item in payload if isinstance(item, dict) and isinstance(item["id"], str)
    }
