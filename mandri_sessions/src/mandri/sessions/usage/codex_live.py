import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from mandri.sessions.usage.codex_counters import codex_totals
from mandri.sessions.usage.codex_delta import codex_request_counters, codex_request_delta
from mandri.sessions.usage.counters import record, text
from mandri.sessions.usage.types import NativeUsageContext, NativeUsageObservation

_LIVE_NAMES = {
    "input_tokens": "inputTokens",
    "cached_input_tokens": "cachedInputTokens",
    "cache_write_input_tokens": "cacheWriteInputTokens",
    "output_tokens": "outputTokens",
    "reasoning_output_tokens": "reasoningOutputTokens",
    "total_tokens": "totalTokens",
}


@dataclass(frozen=True)
class CodexLiveState:
    model: str | None = None
    turn_id: str | None = None
    previous: dict[str, int] | None = None
    previous_model: str | None = None
    reset: int = 0


def normalize_codex_live(
    context: NativeUsageContext,
    event: Mapping[str, Any],
    state: CodexLiveState,
    *,
    observed_at_ms: int,
) -> tuple[tuple[NativeUsageObservation, ...], CodexLiveState]:
    params = record(event.get("params"))
    native_id = text(params.get("threadId")) or context.native_id
    if context.native_id is not None and native_id != context.native_id:
        return (), state
    if event.get("method") == "turn/started":
        turn = record(params.get("turn"))
        return (), replace(state, model=text(turn.get("model")), turn_id=text(turn.get("id")))
    if event.get("method") != "thread/tokenUsage/updated":
        return (), state
    usage = record(params.get("tokenUsage"))
    total, last = _totals(record(usage.get("total"))), _totals(record(usage.get("last")))
    if total is None:
        return (), replace(state, previous=None)
    reset = state.reset + int(
        state.previous is not None and any(total[key] < state.previous[key] for key in total)
    )
    turn_id = text(params.get("turnId"))
    active_turn = state.turn_id is not None and turn_id == state.turn_id
    model = (state.model if active_turn else None) or context.observed_model
    updated = replace(state, previous=total, previous_model=model, reset=reset)
    if context.resumed and state.previous is None and not active_turn:
        return (), updated
    delta = codex_request_delta(
        total, last, state.previous, allow_initial_total=not context.resumed
    )
    if delta is None:
        return (), updated
    if state.previous is not None and state.previous_model != model and delta != last:
        if last is None or not all(last[key] <= delta[key] for key in delta):
            return (), updated
        delta = last
    series = json.dumps(["codex", context.session_id, native_id, "live_request"])
    identity = [series, context.process_epoch, reset, total]
    source_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return (
        (
            NativeUsageObservation(
                source_key=source_key,
                series_key=series,
                context=context,
                scope="request",
                counters=codex_request_counters(delta, last),
                observed_at_ms=observed_at_ms,
                occurred_at_ms=observed_at_ms,
                native_id=native_id,
                turn_id=turn_id,
                model=model,
                observed_model=model,
                cumulative=False,
                includes_descendants=False,
                authoritative=context.routing == "native" and model is not None,
                additive=True,
            ),
        ),
        updated,
    )


def _totals(raw: Mapping[str, Any]) -> dict[str, int] | None:
    return codex_totals(
        {target: raw[source] for target, source in _LIVE_NAMES.items() if source in raw}
    )
