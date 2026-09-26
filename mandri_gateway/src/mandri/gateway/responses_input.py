from typing import Any

_CALL_TYPES = {
    "function_call": "function_call_output",
    "custom_tool_call": "custom_tool_call_output",
}
_OUTPUT_TYPES = frozenset(_CALL_TYPES.values())


def normalize_tool_results(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    normalized: list[Any] = []
    segment: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict) and (
            item.get("type") in {*_CALL_TYPES, *_OUTPUT_TYPES, "reasoning"}
            or (item.get("role") == "assistant" and item.get("type") in (None, "message"))
        ):
            segment.append(item)
        else:
            normalized.extend(_normalize_segment(segment))
            segment = []
            normalized.append(item)
    normalized.extend(_normalize_segment(segment))
    return normalized


def _normalize_segment(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending: dict[str, str] = {}
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    batch: list[dict[str, Any]] = []
    for item in items:
        kind = item.get("type")
        call_id = item.get("call_id")
        batch.append(item)
        if kind in _CALL_TYPES:
            if not isinstance(call_id, str) or not call_id or call_id in seen:
                return items
            seen.add(call_id)
            pending[call_id] = _CALL_TYPES[kind]
        elif kind in _OUTPUT_TYPES:
            if not isinstance(call_id, str) or pending.get(call_id) != kind:
                return items
            del pending[call_id]
            if not pending:
                normalized.extend(_normalize_batch(batch))
                batch = []
    return items if pending else [*normalized, *batch]


def _normalize_batch(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = [
        item
        for item in items
        if item.get("role") == "assistant" and item.get("type") in (None, "message")
    ]
    if len(messages) > 1:
        content: list[Any] = []
        for message in messages:
            parts = message.get("content")
            if isinstance(parts, str):
                content.append({"type": "output_text", "text": parts})
            elif isinstance(parts, list):
                content.extend(parts)
            elif parts is not None:
                return items
        messages = [{**messages[0], "content": content}]
    return [
        *(item for item in items if item.get("type") == "reasoning"),
        *messages,
        *(item for item in items if item.get("type") in _CALL_TYPES),
        *(item for item in items if item.get("type") in _OUTPUT_TYPES),
    ]
