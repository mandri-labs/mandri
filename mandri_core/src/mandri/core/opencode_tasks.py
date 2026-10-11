from typing import Any


def task_content(message: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = dict(message.get("metadata") or {})
    source = "shell" if message.get("type") == "shell" else metadata.get("source")
    if not isinstance(source, str) or not source:
        return []
    status = metadata.get("state", message.get("status", "completed"))
    exit_code = message.get("exit", metadata.get("exit"))
    failed = (
        status in {"error", "timeout"}
        or metadata.get("timeout") is True
        or (isinstance(exit_code, int) and exit_code != 0)
    )
    state = (
        "cancelled"
        if status in {"cancelled", "killed"}
        else "error"
        if failed
        else "completed"
        if status in {"completed", "exited"}
        else "running"
    )
    output = message.get("output")
    text = output.get("output", "") if isinstance(output, dict) else message.get("text", "")
    target = message.get("command") or message.get("description", "")
    if source == "shell" and message.get("shellID"):
        metadata["shellID"] = message["shellID"]
    if metadata.get("childID"):
        metadata["sessionID"] = metadata["childID"]
    return [
        {
            "type": "tool",
            "id": message["id"],
            "name": source,
            "state": {
                "status": state,
                "input": {"command" if source == "shell" else "description": target},
                "content": [{"type": "text", "text": text}],
                "metadata": metadata,
            },
        }
    ]
