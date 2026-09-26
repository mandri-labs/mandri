from typing import Any


def has_output(payload: dict[str, Any]) -> bool:
    event = payload.get("type", "")
    if (
        isinstance(event, str)
        and event.startswith("response.")
        and event.endswith(".delta")
        and payload.get("delta")
    ):
        return True
    for key in ("delta", "content_block"):
        block = payload.get(key)
        if isinstance(block, dict) and any(
            block.get(field) for field in ("text", "thinking", "partial_json", "name")
        ):
            return True
    for key in ("response", "message"):
        envelope = payload.get(key)
        if isinstance(envelope, dict) and has_output(envelope):
            return True
    if any(
        payload.get(field)
        for field in ("content", "output", "text", "thinking", "tool_calls", "function_call")
    ):
        return True
    if isinstance(payload.get("response"), str) and payload["response"]:
        return True
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            block = choice.get("delta", choice.get("message", choice))
            if isinstance(block, dict) and any(
                block.get(field)
                for field in (
                    "content",
                    "text",
                    "reasoning",
                    "reasoning_content",
                    "tool_calls",
                    "function_call",
                    "audio",
                )
            ):
                return True
    candidates = payload.get("candidates")
    return isinstance(candidates, list) and any(
        isinstance(candidate, dict) and candidate.get("content") for candidate in candidates
    )
