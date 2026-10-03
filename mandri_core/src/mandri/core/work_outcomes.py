import hashlib
import json
from typing import Any

from mandri.core.ids import HarnessKind
from mandri.core.types.conversation_status import WorkOutcome


def work_event_key(raw: dict[str, Any], timestamp: object = None) -> str:
    value = {"raw": raw, "timestamp": timestamp}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def terminal_outcome(harness: HarnessKind, raw: dict[str, Any]) -> WorkOutcome | None:
    if harness is HarnessKind.CODEX:
        params = _record(raw.get("params"))
        turn = _record(params.get("turn"))
        payload = _record(raw.get("payload"))
        if raw.get("method") == "turn/completed":
            return _outcome(turn.get("status"))
        if raw.get("type") == "event_msg":
            event = payload.get("type")
            if event in {"task_aborted", "turn_aborted"}:
                return "interrupted"
            if event in {"task_complete", "task_completed", "turn_complete"}:
                return "failed" if payload.get("error") else "completed"
    elif harness is HarnessKind.CLAUDE:
        if raw.get("type") == "result":
            if raw.get("is_error") or str(raw.get("subtype", "")).startswith("error"):
                return "failed"
            return "completed"
        message = _record(raw.get("message"))
        if raw.get("type") == "assistant" and message.get("stop_reason") in {
            "end_turn",
            "stop_sequence",
            "max_tokens",
        }:
            return "completed"
    elif harness is HarnessKind.OPENCODE:
        properties = _record(raw.get("properties"))
        if raw.get("type") in {"session.idle", "session.status"}:
            status = _record(properties.get("status"))
            if raw.get("type") == "session.idle" or status.get("type") == "idle":
                return _outcome(properties.get("outcome")) or "completed"
        if raw.get("type") == "session.error":
            error = _record(properties.get("error"))
            return "interrupted" if error.get("name") == "MessageAbortedError" else "failed"
        info = _record(properties.get("info"))
        if info.get("role") == "assistant":
            error = _record(info.get("error"))
            if error:
                return "interrupted" if error.get("name") == "MessageAbortedError" else "failed"
            if (
                _record(info.get("time")).get("completed") is not None
                and info.get("finish") not in {None, "tool-calls", "unknown"}
                and not info.get("summary")
            ):
                return "completed"
    elif harness is HarnessKind.PI:
        message = _record(raw.get("message"))
        if message.get("role") == "assistant":
            reason = message.get("stopReason")
            if reason == "error":
                return "failed"
            if reason == "aborted":
                return "interrupted"
            if reason in {"stop", "length"}:
                return "completed"
    elif harness is HarnessKind.AGY:
        if raw.get("event") == "result":
            return _outcome(_record(raw.get("result")).get("status"))
        hook = raw.get("hook") if raw.get("event") == "hook" else raw.get("event")
        data = _record(raw.get("data")) if raw.get("event") == "hook" else raw
        if hook == "Stop" and data.get("fullyIdle") is True:
            if data.get("error"):
                return "failed"
            if data.get("killed") or data.get("interrupted"):
                return "interrupted"
            return "completed"
    return None


def _record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _outcome(value: Any) -> WorkOutcome | None:
    if not isinstance(value, str):
        return None
    if value in {"completed", "succeeded", "success", "SUCCESS"}:
        return "completed"
    if value in {"failed", "error", "ERROR"}:
        return "failed"
    if value in {"interrupted", "aborted", "cancelled", "killed", "stopped"}:
        return "interrupted"
    return None
