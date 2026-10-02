from typing import Any


def part_id(message_id: str, kind: str, ordinal: int | str) -> str:
    return f"{message_id}:{kind}:{ordinal}"


def message_error(error: dict[str, Any]) -> dict[str, Any]:
    return {
        **error,
        "name": "MessageAbortedError"
        if error.get("type") == "aborted"
        else error.get("name", error.get("type", "UnknownError")),
    }


def message_record(message: dict[str, Any], session_id: str) -> dict[str, Any]:
    kind = message.get("type")
    info = {
        key: value
        for key, value in message.items()
        if key not in {"content", "type", "text", "files"}
    }
    info.update(role=kind, sessionID=session_id)
    if isinstance(info.get("error"), dict):
        info["error"] = message_error(info["error"])
    model = message.get("model")
    if isinstance(model, dict):
        info.update(providerID=model.get("providerID"), modelID=model.get("id"))
    parts: list[dict[str, Any]] = []
    if kind in {"user", "synthetic", "system", "skill"}:
        content = [{"type": "text", "text": message.get("text", "")}]
        content.extend(
            {
                "type": "file",
                "mime": item["mime"],
                "filename": item.get("name"),
                "url": f"data:{item['mime']};base64,{item['data']}",
            }
            for item in message.get("files", [])
        )
    else:
        content = message.get("content", [])
    ordinals: dict[str, int] = {}
    for item in content:
        item_kind = item["type"]
        ordinal = ordinals.get(item_kind, 0)
        ordinals[item_kind] = ordinal + 1
        part = {
            **item,
            "id": part_id(message["id"], item_kind, item.get("id", ordinal)),
            "messageID": message["id"],
            "sessionID": session_id,
        }
        timing = item.get("time", message.get("time", {}))
        part["time"] = {"start": timing.get("created")}
        if timing.get("completed") is not None:
            part["time"]["end"] = timing["completed"]
        if item_kind == "tool":
            state = dict(item["state"])
            state["time"] = part["time"]
            if state.get("status") == "streaming":
                state.update(status="pending", input={}, raw=state.get("input", ""))
            if "content" in state:
                state["output"] = "\n".join(
                    value["text"] for value in state["content"] if value.get("type") == "text"
                )
            if isinstance(state.get("error"), dict):
                state["error"] = state["error"].get("message", "Tool failed")
            part.update(tool=item["name"], callID=item["id"], state=state)
        parts.append(part)
    return {"message": info, "parts": parts}


def message_events(message: dict[str, Any], session_id: str) -> list[dict[str, Any]]:
    record = message_record(message, session_id)
    if message.get("type") == "idle":
        return [{"type": "session.idle", "properties": {"sessionID": session_id}}]
    return [
        {"type": "message.updated", "properties": {"info": record["message"]}},
        *[
            {"type": "message.part.updated", "properties": {"part": part}}
            for part in record["parts"]
        ],
    ]
