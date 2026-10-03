import json
from typing import Any

from mandri.core.ids import HarnessKind
from mandri.core.types.conversation_status import WorkObservation
from mandri.core.work_content import work_content_key
from mandri.core.work_outcomes import terminal_outcome, work_event_key
from mandri.sessions.native_activity import native_turn_busy


def native_work_observations(
    harness: HarnessKind, raw: dict[str, Any], native_id: str
) -> tuple[WorkObservation, ...]:
    if not _belongs(harness, raw, native_id):
        return ()
    key = work_event_key(raw)
    outcome = terminal_outcome(harness, raw)
    if outcome is not None:
        settled = _settles(harness, raw)
        return (
            WorkObservation(
                state="idle" if settled else "unknown",
                outcome=outcome,
                key=key,
                content_key=work_content_key(harness, raw),
            ),
        )
    if _settles(harness, raw):
        return (WorkObservation(state="idle", key=key),)
    busy = native_turn_busy(harness, [_encode(raw)])
    if busy is True or _progress(harness, raw):
        return (WorkObservation(state="working", progress=True, key=key),)
    return ()


def _belongs(harness: HarnessKind, raw: dict[str, Any], native_id: str) -> bool:
    if harness is HarnessKind.CLAUDE and raw.get("parent_tool_use_id"):
        return False
    params = raw.get("params")
    properties = raw.get("properties")
    data = raw.get("data")
    nested = params if isinstance(params, dict) else properties
    owner = None
    if isinstance(nested, dict):
        owner = nested.get("threadId", nested.get("sessionID"))
        info = nested.get("info", nested.get("part"))
        if owner is None and isinstance(info, dict):
            owner = info.get("sessionID")
    if harness is HarnessKind.AGY:
        owner = data.get("conversationId") if isinstance(data, dict) else raw.get("conversation_id")
        step = raw.get("step_update")
        if owner is None and isinstance(step, dict):
            owner = step.get("conversation_id")
    return owner is None or owner == native_id


def _settles(harness: HarnessKind, raw: dict[str, Any]) -> bool:
    if harness is HarnessKind.CODEX:
        return terminal_outcome(harness, raw) is not None
    if harness is HarnessKind.CLAUDE:
        queued = raw.get("queued_turn_count")
        return raw.get("type") == "result" and not (isinstance(queued, int) and queued > 0)
    if harness is HarnessKind.OPENCODE:
        properties = raw.get("properties")
        status = properties.get("status") if isinstance(properties, dict) else None
        return raw.get("type") in {"session.idle", "session.error"} or (
            raw.get("type") == "session.status"
            and isinstance(status, dict)
            and status.get("type") == "idle"
        )
    if harness is HarnessKind.PI:
        return raw.get("type") == "agent_settled"
    hook = raw.get("hook") if raw.get("event") == "hook" else raw.get("event")
    data = raw.get("data") if raw.get("event") == "hook" else raw
    return hook == "Stop" and isinstance(data, dict) and data.get("fullyIdle") is True


def _progress(harness: HarnessKind, raw: dict[str, Any]) -> bool:
    if harness is HarnessKind.OPENCODE:
        properties = raw.get("properties")
        status = properties.get("status") if isinstance(properties, dict) else None
        return (
            raw.get("type") == "session.status"
            and isinstance(status, dict)
            and status.get("type") in {"busy", "retry"}
        )
    if harness is HarnessKind.CODEX:
        return raw.get("method") == "turn/started"
    if harness is HarnessKind.PI:
        return raw.get("type") in {"agent_start", "message_update", "tool_execution_start"}
    if harness is HarnessKind.CLAUDE:
        return raw.get("type") in {"assistant", "stream_event"}
    if harness is HarnessKind.AGY:
        data = raw.get("data") if raw.get("event") == "hook" else raw
        hook = raw.get("hook") if raw.get("event") == "hook" else raw.get("event")
        step = raw.get("step_update", raw)
        return (hook in {"PreInvocation", "PreToolUse"} and isinstance(data, dict)) or (
            isinstance(step, dict)
            and step.get("state", step.get("status")) in {"ACTIVE", "RUNNING"}
        )
    return False


def _encode(raw: dict[str, Any]) -> str:
    return json.dumps(raw)
