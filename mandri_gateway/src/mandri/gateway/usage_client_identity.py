import json
from typing import Any


def client_response_id(value: Any) -> str | None:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        value = dump()
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str):
        if len(value) > 1024 * 1024:
            return None
        for line in value.splitlines():
            data = line.removeprefix("data:").strip()
            try:
                identity = client_response_id(json.loads(data))
            except (ValueError, RecursionError):
                continue
            if identity is not None:
                return identity
        return None
    if not isinstance(value, dict):
        return None
    for key in ("message", "response"):
        if isinstance(value.get(key), dict):
            return client_response_id(value[key])
    if value.get("type") in {"function_call", "function_call_output", "reasoning"}:
        return None
    identity = value.get("id") or value.get("responseId")
    return (
        identity
        if isinstance(identity, str) and identity.strip() and len(identity) <= 512
        else None
    )
