import json
import re
from collections.abc import Callable
from typing import cast

from mandri.gateway.surrogate.types import JSONValue

TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null')


def unique_object(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON content key")
        result[key] = value
    return result


def rewrite_json(text: str, transform: Callable[[JSONValue], JSONValue]) -> str | None:
    stripped = text.strip()
    if not stripped.startswith(("{", "[", '"')):
        return None
    try:
        original = cast(JSONValue, json.loads(text, object_pairs_hook=unique_object))
    except (ValueError, RecursionError):
        return None
    try:
        transformed = transform(original)
    except RecursionError:
        return None
    if transformed == original:
        return text
    try:
        serialized = json.dumps(transformed, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        return None
    original_tokens = list(TOKEN.finditer(text))
    transformed_tokens = list(TOKEN.finditer(serialized))
    if len(original_tokens) != len(transformed_tokens):
        return None
    result = []
    cursor = 0
    for old, new in zip(original_tokens, transformed_tokens, strict=True):
        result.append(text[cursor : old.start()])
        old_value = json.loads(old.group())
        new_value = json.loads(new.group())
        replacement = old.group()
        if old_value != new_value:
            replacement = json.dumps(new_value, ensure_ascii="\\u" in old.group(), allow_nan=False)
            if "\\/" in old.group():
                replacement = replacement.replace("/", "\\/")
        result.append(replacement)
        cursor = old.end()
    result.append(text[cursor:])
    return "".join(result)
