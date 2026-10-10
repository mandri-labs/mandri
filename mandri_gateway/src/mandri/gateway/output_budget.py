import re
from typing import Any


def remaining_output(error: Exception) -> int | None:
    if getattr(error, "status_code", None) not in (400, 413, 422):
        return None
    text = str(error)
    hint = re.search(
        r"\bmax(?:imum)?[ _-]?(?:output[ _-]?)?tokens[^\n]{0,80}?\bat most\s+(\d+)",
        text,
        re.IGNORECASE,
    )
    context = re.search(
        r"\b(?:maximum\s+)?context(?:\s+(?:window|length))?\s*(?:is\s*)?[:( ]+(\d+)",
        text,
        re.IGNORECASE,
    )
    prompt = re.search(
        r"\b(?:prompt|input|messages)\s*(?:contains\s*)?[:( ]+(\d+)\s+tokens", text, re.IGNORECASE
    )
    if prompt is None:
        prompt = re.search(r"(\d+)\s+tokens\s+in\s+(?:the\s+)?messages", text, re.IGNORECASE)
    if context is None or prompt is None:
        value = int(hint[1]) if hint is not None else 0
        return value if value > 0 else None
    remaining = int(context[1]) - int(prompt[1])
    if hint is not None:
        remaining = min(remaining, int(hint[1]))
    return remaining if remaining > 0 else None


def retry_kwargs(kwargs: dict[str, Any], error: Exception) -> dict[str, Any] | None:
    remaining = remaining_output(error)
    if remaining is None:
        return None
    result = dict(kwargs)
    if "contents" in kwargs:
        config = dict(kwargs.get("config") or {})
        previous = config.get("maxOutputTokens")
        if type(previous) is int and previous <= remaining:
            return None
        result["config"] = {**config, "maxOutputTokens": remaining}
        return result
    key = next(
        (
            name
            for name in ("max_output_tokens", "max_completion_tokens", "max_tokens")
            if name in kwargs
        ),
        "max_output_tokens" if "input" in kwargs else "max_tokens",
    )
    previous = kwargs.get(key)
    if type(previous) is int and previous <= remaining:
        return None
    result[key] = remaining
    return result
