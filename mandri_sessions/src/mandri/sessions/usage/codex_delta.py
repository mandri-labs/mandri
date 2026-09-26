from mandri.sessions.usage.codex_counters import valid_totals


def codex_request_delta(
    total: dict[str, int],
    last: dict[str, int] | None,
    previous: dict[str, int] | None,
    *,
    allow_initial_total: bool = True,
) -> dict[str, int] | None:
    if total == previous or not total["total_tokens"]:
        return None
    if previous is None:
        if last is not None:
            return last if all(last[key] <= total[key] for key in total) else None
        return total if allow_initial_total else None
    delta = {key: value - previous[key] for key, value in total.items()}
    if valid_totals(delta):
        return delta
    return total if last == total else None


def codex_request_counters(
    delta: dict[str, int], last: dict[str, int] | None
) -> dict[str, int | None]:
    values: dict[str, int | None] = dict(delta)
    values["uncached_input_tokens"] = (
        delta["input_tokens"] - delta["cache_read_tokens"] - delta["cache_write_tokens"]
    )
    values["visible_output_tokens"] = delta["output_tokens"] - delta["reasoning_tokens"]
    values["request_count"] = 1 if delta == last else None
    return values
