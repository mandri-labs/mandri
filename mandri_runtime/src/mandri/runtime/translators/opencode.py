"""Opencode pipe: SSE data frames as parsed JSON, verbatim."""

import json
from typing import Any

from mandri.runtime.translators.base import EventPipe


class OpencodeEventPipe(EventPipe):
    source = "opencode"

    def _wrap(self, event: Any) -> dict[str, Any] | None:
        if isinstance(event, dict):
            return self._frame(event)
        if isinstance(event, bytes):
            event = event.decode("utf-8")
        if not isinstance(event, str):
            return None
        payload = _data_payload(event)
        if payload is None:
            return None
        return self._frame(payload)


def _data_payload(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    body = stripped[len("data:") :].strip()
    if not body or body == "[DONE]":
        return None
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
