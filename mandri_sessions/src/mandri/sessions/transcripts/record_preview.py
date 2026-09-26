import json
from collections.abc import Iterator
from typing import BinaryIO

from pydantic import TypeAdapter

MAX_PREVIEW_BYTES = 32 * 1024
MAX_PREVIEW_CHARS = 16 * 1024
_JSON = TypeAdapter(object)
_TEXT_FIELDS = {
    "command",
    "cmd",
    "stdout",
    "stderr",
    "aggregated_output",
    "output",
    "text",
    "content",
    "message",
    "summary",
    "diff",
    "unified_diff",
    "result",
    "description",
    "preview",
}
_BINARY_FIELDS = {"data", "image", "images", "image_url", "encrypted_content"}


def record_preview(handle: BinaryIO, start: int, size: int) -> dict[str, object]:
    handle.seek(start)
    prefix = handle.read(min(size, MAX_PREVIEW_BYTES))
    return preview_from_prefix(prefix)


def preview_from_prefix(prefix: bytes) -> dict[str, object]:
    source = prefix[:MAX_PREVIEW_BYTES].decode("utf-8", errors="ignore")
    try:
        value = _JSON.validate_json(source, experimental_allow_partial="trailing-strings")
    except ValueError:
        return {"preview": source[:MAX_PREVIEW_CHARS], "preview_truncated": True}
    pieces = list(_text_parts(value))
    text = (
        "\n\n".join(dict.fromkeys(pieces))
        if pieces
        else json.dumps(value, ensure_ascii=False, indent=2)
    )
    return {
        "preview": text[:MAX_PREVIEW_CHARS],
        "preview_truncated": True,
        **_event_kind(value),
    }


def _text_parts(value: object, field: str = "", depth: int = 0) -> Iterator[str]:
    if depth > 16 or field in _BINARY_FIELDS:
        return
    if isinstance(value, str):
        if field in _TEXT_FIELDS and value and not value.startswith("data:image/"):
            yield value
    elif isinstance(value, list):
        if field in {"command", "cmd"} and all(isinstance(part, str) for part in value):
            yield " ".join(value)
        else:
            for part in value:
                yield from _text_parts(part, field, depth + 1)
    elif isinstance(value, dict):
        for key, part in value.items():
            yield from _text_parts(part, str(key), depth + 1)


def _event_kind(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    kind = value.get("type")
    kind = kind if isinstance(kind, str) else None
    if kind in {"compacted", "compaction", "ContextCompaction", "contextCompaction"}:
        return {"event_kind": "compaction"}
    if kind == "system" and value.get("subtype") == "compact_boundary":
        return {"event_kind": "compaction"}
    if kind in {"CommandExecution", "commandExecution", "exec_command_end"}:
        return {"event_kind": "command"}
    if kind in {"FileChange", "fileChange"}:
        return {"event_kind": "file_change"}
    if (
        value.get("name") in ("Bash", "bash", "exec_command", "shell")
        or value.get("tool") == "bash"
    ):
        return {"event_kind": "command"}
    for key in ("payload", "item", "message", "properties", "part", "content"):
        child = value.get(key)
        for part in child if isinstance(child, list) else [child]:
            if result := _event_kind(part):
                return result
    return {}
