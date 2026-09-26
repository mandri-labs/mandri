import json

_DECODER = json.JSONDecoder()


def irrelevant_codex_record(prefix: bytes) -> bool:
    source = prefix.decode("utf-8", errors="ignore")
    try:
        offset = _field_offset(source, "type")
        if offset is None:
            return False
        kind, _ = _DECODER.raw_decode(source, offset)
        if kind in {
            "response_item",
            "compacted",
            "world_state",
            "inter_agent_communication_metadata",
        }:
            return True
        if kind != "event_msg":
            return False
        payload = _field_offset(source, "payload")
        if payload is None:
            return False
        event_type = _field_offset(source, "type", payload)
        if event_type is None:
            return False
        name, _ = _DECODER.raw_decode(source, event_type)
        return isinstance(name, str) and name != "token_count"
    except (ValueError, TypeError):
        return False


def _field_offset(source: str, name: str, offset: int = 0) -> int | None:
    offset = _space(source, offset)
    if source[offset : offset + 1] != "{":
        return None
    offset += 1
    while offset < len(source):
        key, offset = _DECODER.raw_decode(source, _space(source, offset))
        offset = _space(source, offset)
        if source[offset : offset + 1] != ":":
            return None
        offset = _space(source, offset + 1)
        if key == name:
            return offset
        _, offset = _DECODER.raw_decode(source, offset)
        offset = _space(source, offset)
        if source[offset : offset + 1] != ",":
            return None
        offset += 1
    return None


def _space(source: str, offset: int) -> int:
    while offset < len(source) and source[offset] in " \r\n\t":
        offset += 1
    return offset
