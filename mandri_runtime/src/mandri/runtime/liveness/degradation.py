from typing import Any

from mandri.runtime.session_feed import DEGRADED_SOURCE, DegradationKind


def is_stream_degradation(payload: dict[str, Any]) -> bool:
    raw = payload.get("raw")
    if payload.get("source") != DEGRADED_SOURCE or not isinstance(raw, dict):
        return False
    error = raw.get("error")
    return isinstance(error, str) and error in DegradationKind
