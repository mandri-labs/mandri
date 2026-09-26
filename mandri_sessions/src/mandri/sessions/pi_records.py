import json
from typing import BinaryIO

_DECODER = json.JSONDecoder()
_FIELDS = {"type", "id", "parentId", "version"}


def pi_entry_fields(prefix: bytes) -> dict[str, object]:
    source = prefix.decode("utf-8", errors="replace")
    fields: dict[str, object] = {}
    offset = 0
    try:
        while offset < len(source) and source[offset].isspace():
            offset += 1
        if source[offset : offset + 1] != "{":
            return fields
        offset += 1
        while offset < len(source):
            while offset < len(source) and source[offset].isspace():
                offset += 1
            key, offset = _DECODER.raw_decode(source, offset)
            while offset < len(source) and source[offset].isspace():
                offset += 1
            if source[offset : offset + 1] != ":":
                break
            offset += 1
            while offset < len(source) and source[offset].isspace():
                offset += 1
            value, offset = _DECODER.raw_decode(source, offset)
            if isinstance(key, str) and key in _FIELDS:
                fields[key] = value
            if {"type", "id", "parentId"} <= fields.keys():
                break
            while offset < len(source) and source[offset].isspace():
                offset += 1
            if source[offset : offset + 1] != ",":
                break
            offset += 1
    except (ValueError, RecursionError):
        pass
    return fields


def read_pi_entry_fields(
    handle: BinaryIO, start: int, end: int, prefix: bytes
) -> dict[str, object]:
    fields = pi_entry_fields(prefix)
    if (
        end - start > len(prefix)
        and fields.get("type") != "session"
        and not {"id", "parentId"} <= fields.keys()
    ):
        handle.seek(start)
        try:
            entry = json.loads(handle.read(end - start))
            if isinstance(entry, dict):
                fields = {key: entry[key] for key in _FIELDS if key in entry}
        except (ValueError, RecursionError):
            pass
        handle.seek(end)
    return fields
