import hashlib
import json
from typing import Any

from mandri.core.ids import HarnessKind
from mandri.core.work_outcomes import terminal_outcome


def work_content_key(harness: HarnessKind, raw: dict[str, Any]) -> str | None:
    identity: object = None
    content: object = None
    if harness is HarnessKind.CODEX:
        params = _record(raw.get("params"))
        turn = _record(params.get("turn"))
        payload = _record(raw.get("payload"))
        identity = (
            turn.get("id")
            if raw.get("method") == "turn/completed"
            else (payload.get("turn_id") if terminal_outcome(harness, raw) else None)
        )
    elif harness is HarnessKind.CLAUDE:
        message = _record(raw.get("message"))
        if raw.get("type") == "assistant":
            identity, content = message.get("id"), _text(message.get("content"))
    elif harness is HarnessKind.PI:
        message = _record(raw.get("message"))
        if message.get("role") == "assistant":
            identity, content = message.get("timestamp"), _text(message.get("content"))
    elif harness is HarnessKind.OPENCODE:
        properties = _record(raw.get("properties"))
        info = _record(properties.get("info"))
        if info.get("role") == "assistant" and terminal_outcome(harness, raw):
            identity = info.get("id")
            content = _record(info.get("time")).get("completed")
    elif harness is HarnessKind.AGY:
        step = _record(raw.get("step_update")) or raw
        if step.get("step_type", step.get("type")) in {
            "agent_response",
            "planner_response",
            "PLANNER_RESPONSE",
        } and step.get("state", step.get("status")) in {"DONE", "COMPLETED"}:
            identity = step.get("step_index", step.get("stepIdx"))
            response = _record(step.get("agent_response"))
            content = _text(response.get("text", step.get("content", step.get("text"))))
    if identity is None:
        return None
    encoded = json.dumps([harness.value, identity, content], separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def contains_completed_content(
    harness: HarnessKind, entries: list[str], content_key: str | None
) -> bool:
    if content_key is None:
        return False
    for entry in reversed(entries):
        try:
            raw = json.loads(entry)
        except (ValueError, TypeError, RecursionError):
            continue
        if not isinstance(raw, dict):
            continue
        if harness is not HarnessKind.AGY and terminal_outcome(harness, raw) is None:
            continue
        if work_content_key(harness, raw) == content_key:
            return True
    return False


def _record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(item.get("text", "")) for item in value if isinstance(item, dict))
    return ""
