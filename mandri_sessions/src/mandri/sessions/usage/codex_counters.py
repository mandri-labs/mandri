from collections.abc import Mapping
from typing import Any

from mandri.sessions.usage.counters import count

HISTORY_COUNTERS = {
    "input_tokens": "input_tokens",
    "cache_read_tokens": "cached_input_tokens",
    "cache_write_tokens": "cache_write_input_tokens",
    "output_tokens": "output_tokens",
    "reasoning_tokens": "reasoning_output_tokens",
    "total_tokens": "total_tokens",
}


def codex_totals(raw: Mapping[str, Any]) -> dict[str, int] | None:
    values = {}
    for target, source in HISTORY_COUNTERS.items():
        value = count(raw.get(source, 0) if target == "cache_write_tokens" else raw.get(source))
        if value is None:
            return None
        values[target] = value
    return values if valid_totals(values) else None


def valid_totals(values: Mapping[str, int]) -> bool:
    return (
        set(values) == set(HISTORY_COUNTERS)
        and all(type(value) is int and value >= 0 for value in values.values())
        and values["cache_read_tokens"] + values["cache_write_tokens"] <= values["input_tokens"]
        and values["reasoning_tokens"] <= values["output_tokens"]
        and values["total_tokens"] == values["input_tokens"] + values["output_tokens"]
    )
