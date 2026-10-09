import json
import re
from collections.abc import Callable
from typing import cast

from mandri.gateway.surrogate.types import JSONValue

TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null')
NUMBERED = re.compile(r"^([ \t]*\d+[ ]*(?:\t|\u2192)[ \t]*)(.*?)(\r?\n|$)")


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


def rewrite_numbered_json(text: str, transform: Callable[[JSONValue], JSONValue]) -> str | None:
    lines = text.splitlines(keepends=True)
    result: list[str] = []
    changed = False
    index = 0
    while index < len(lines):
        matches: list[re.Match[str]] = []
        while index + len(matches) < len(lines):
            match = NUMBERED.fullmatch(lines[index + len(matches)])
            if match is None:
                break
            matches.append(match)
        if not matches:
            result.append(lines[index])
            index += 1
            continue
        source = "\n".join(match.group(2) for match in matches)
        rendered = rewrite_json(source, transform)
        new_lines = rendered.split("\n") if rendered is not None else []
        if len(new_lines) == len(matches):
            result.extend(
                match.group(1) + body + match.group(3)
                for match, body in zip(matches, new_lines, strict=True)
            )
            changed = changed or rendered != source
        else:
            result.extend(lines[index : index + len(matches)])
        index += len(matches)
    return "".join(result) if changed else None
