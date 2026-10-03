import json
from typing import Any


def codex_children(
    raw: dict[str, Any], children: set[str], calls: dict[str, Any]
) -> tuple[bool, bool]:
    before = set(children)
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        return False, False
    item = payload.get("item", payload)
    if not isinstance(item, dict):
        return False, False
    kind = str(item.get("type", "")).replace("_", "").lower()
    if kind == "subagentactivity":
        child = item.get("agent_thread_id", item.get("agentThreadId"))
        if isinstance(child, str):
            if item.get("kind") in {"started", "interacted"}:
                children.add(child)
            elif item.get("kind") in {"completed", "interrupted"}:
                children.discard(child)
    if kind == "collabagenttoolcall":
        states = item.get("agents_states", item.get("agentsStates"))
        if isinstance(states, dict):
            for child, state in states.items():
                status = state.get("status") if isinstance(state, dict) else state
                _state(children, str(child), status)
        elif item.get("tool") in {"spawn_agent", "send_message", "followup_task", "resume_agent"}:
            receivers = item.get("receiver_thread_ids", [])
            children.update(child for child in receivers if isinstance(child, str))
    if raw.get("type") == "response_item" and payload.get("type") == "function_call":
        name = str(payload.get("name", "")).split(".")[-1]
        reference = payload.get("call_id")
        if name in {
            "spawn_agent",
            "send_message",
            "send_input",
            "followup_task",
            "resume_agent",
            "close_agent",
            "wait",
            "wait_agent",
        } and isinstance(reference, str):
            arguments = _decode(payload.get("arguments"))
            calls[reference] = {"name": name, "arguments": arguments}
            if name == "spawn_agent":
                children.add(f"spawn:{reference}")
    if raw.get("type") == "response_item" and payload.get("type") == "function_call_output":
        call = calls.pop(str(payload.get("call_id")), None)
        if isinstance(call, dict):
            name, arguments = call["name"], call["arguments"]
            output = _decode(payload.get("output"))
            targets = arguments.get(
                "ids", arguments.get("targets", [arguments.get("id", arguments.get("target"))])
            )
            targets = targets if isinstance(targets, list) else [targets]
            if name == "spawn_agent":
                children.discard(f"spawn:{payload.get('call_id')}")
                child = output.get("agent_id", output.get("task_name"))
                if isinstance(child, str):
                    children.add(child)
                elif not output.get("error"):
                    return before != children, True
            elif name in {"close_agent"} and "previous_status" in output:
                for target in targets:
                    if isinstance(target, str):
                        children.discard(target)
            elif name in {
                "send_message",
                "send_input",
                "followup_task",
                "resume_agent",
            } and not output.get("error"):
                children.update(target for target in targets if isinstance(target, str))
            statuses = output.get("status")
            if isinstance(statuses, dict):
                for child, status in statuses.items():
                    _state(children, str(child), status)
    return before != children, False


def _state(children: set[str], child: str, status: Any) -> None:
    if isinstance(status, dict):
        status = next(iter(status), None)
    if status in {"pendingInit", "running", "pending_init"}:
        children.add(child)
    elif status in {"completed", "interrupted", "errored", "shutdown", "notFound", "not_found"}:
        children.discard(child)


def _decode(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            return {}
    return value if isinstance(value, dict) else {}
