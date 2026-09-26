import json
from typing import BinaryIO

MAX_HEADER_BYTES = 8192
_DECODER = json.JSONDecoder()


def record_metadata(handle: BinaryIO, start: int, size: int) -> dict[str, str]:
    handle.seek(start)
    source = handle.read(min(size, MAX_HEADER_BYTES)).decode("utf-8", errors="replace")
    metadata = {}
    for name, path in (
        ("original_type", ("type",)),
        ("original_event_type", ("payload", "type")),
        ("original_item_type", ("payload", "item", "type")),
    ):
        try:
            offset = 0
            for field in path:
                found = _field_offset(source, field, offset)
                if found is None:
                    break
                offset = found
            else:
                value, _ = _DECODER.raw_decode(source, offset)
                if isinstance(value, str):
                    metadata[name] = value
        except (ValueError, RecursionError):
            continue
    return metadata


def _field_offset(source: str, name: str, offset: int) -> int | None:
    offset = _skip_space(source, offset)
    if source[offset : offset + 1] != "{":
        return None
    offset += 1
    while offset < len(source):
        key, offset = _DECODER.raw_decode(source, _skip_space(source, offset))
        offset = _skip_space(source, offset)
        if source[offset : offset + 1] != ":":
            return None
        offset = _skip_space(source, offset + 1)
        if key == name:
            return offset
        _, offset = _DECODER.raw_decode(source, offset)
        offset = _skip_space(source, offset)
        if source[offset : offset + 1] != ",":
            return None
        offset += 1
    return None


def _skip_space(source: str, offset: int) -> int:
    while offset < len(source) and source[offset] in " \r\n\t":
        offset += 1
    return offset
