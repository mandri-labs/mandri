import dataclasses
from collections.abc import Awaitable, Callable
from typing import Any

from mandri.runtime.control.errors import ControlTransportError

NativeCall = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
_CHILD_SOURCES = [
    "subAgent",
    "subAgentReview",
    "subAgentCompact",
    "subAgentThreadSpawn",
    "subAgentOther",
]


@dataclasses.dataclass(frozen=True)
class CodexLivenessSnapshot:
    status: dict[str, Any]
    queued: bool
    children: frozenset[str]
    approval_pending: bool


def _result(message: dict[str, Any]) -> dict[str, Any]:
    error = message.get("error")
    if error is not None:
        raise ControlTransportError(f"Codex liveness query failed: {error}")
    result = message.get("result")
    if not isinstance(result, dict):
        raise ControlTransportError("Codex liveness query has no result")
    return result


def _status(thread: dict[str, Any], *, root: bool = False) -> dict[str, Any]:
    status = thread.get("status")
    allowed = {"active", "idle"} if root else {"active", "idle", "notLoaded"}
    if not isinstance(status, dict) or status.get("type") not in allowed:
        raise ControlTransportError("Codex runtime status is unavailable")
    return status


def _approval_pending(status: dict[str, Any]) -> bool:
    flags = status.get("activeFlags")
    if status.get("type") == "active" and not isinstance(flags, list):
        raise ControlTransportError("Codex active status has no flags")
    return isinstance(flags, list) and bool(
        {"waitingOnApproval", "waitingOnUserInput"}.intersection(flags)
    )


async def read_queue(call: NativeCall, thread_id: str) -> bool:
    result = _result(await call("thread/queue/list", {"threadId": thread_id, "limit": 1}))
    entries = result.get("data")
    if not isinstance(entries, list):
        raise ControlTransportError("Codex queue state is unavailable")
    return bool(entries or result.get("nextCursor"))


async def read_snapshot(call: NativeCall, thread_id: str) -> CodexLivenessSnapshot:
    result = _result(await call("thread/read", {"threadId": thread_id, "includeTurns": False}))
    root = result.get("thread")
    if not isinstance(root, dict) or root.get("id") != thread_id:
        raise ControlTransportError("Codex liveness query returned a different thread")
    status = _status(root, root=True)
    queued = await read_queue(call, thread_id)
    pending = _approval_pending(status)
    children: set[str] = set()
    cursor = None
    seen: set[str] = set()
    while True:
        params: dict[str, Any] = {
            "ancestorThreadId": thread_id,
            "sourceKinds": _CHILD_SOURCES,
            "useStateDbOnly": True,
        }
        if cursor is not None:
            params["cursor"] = cursor
        page = _result(await call("thread/list", params))
        entries = page.get("data")
        if not isinstance(entries, list):
            raise ControlTransportError("Codex descendant state is unavailable")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                raise ControlTransportError("Codex descendant state is invalid")
            child_status = _status(entry)
            if child_status.get("type") == "active":
                children.add(entry["id"])
            pending |= _approval_pending(child_status)
        cursor = page.get("nextCursor")
        if cursor is None:
            return CodexLivenessSnapshot(status, queued, frozenset(children), pending)
        if not isinstance(cursor, str) or cursor in seen:
            raise ControlTransportError("Codex descendant cursor is invalid")
        seen.add(cursor)
