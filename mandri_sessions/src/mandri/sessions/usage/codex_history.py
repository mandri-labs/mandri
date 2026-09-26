import hashlib
import json
from collections.abc import Mapping
from typing import Any

from mandri.sessions.usage.codex_counters import codex_totals
from mandri.sessions.usage.codex_delta import codex_request_counters, codex_request_delta
from mandri.sessions.usage.counters import count, record, text
from mandri.sessions.usage.history_state import HistoryState
from mandri.sessions.usage.normalize import _timestamp
from mandri.sessions.usage.types import NativeUsageContext, NativeUsageObservation


def codex_history_usage(
    context: NativeUsageContext,
    event: Mapping[str, Any],
    state: HistoryState,
    *,
    position: int,
    observed_at_ms: int,
) -> tuple[NativeUsageObservation, ...]:
    payload = record(event.get("payload"))
    if event.get("type") == "session_meta":
        if not state.metadata_seen:
            state.metadata_seen = True
            matches = context.native_id is not None and payload.get("id") == context.native_id
            boundary = count(payload.get("subagent_history_start_ordinal"))
            inherited = bool(payload.get("forked_from_id"))
            state.own_start = boundary if matches and boundary is not None else None
            state.own_usage_proven = matches and (not inherited or boundary is not None)
            state.ownership_blocked = not state.own_usage_proven
            _provider(state, payload, context)
        return ()
    if state.own_start is not None and state.ordinal < state.own_start:
        return ()
    if event.get("type") == "turn_context":
        state.model = text(payload.get("model"))
        state.turn_id = text(payload.get("turn_id"))
        if "model_provider" in payload or "provider" in payload:
            _provider(state, payload, context)
        return ()
    if event.get("type") != "event_msg" or payload.get("type") != "token_count":
        return ()
    info = record(payload.get("info"))
    if not info:
        return ()
    total = codex_totals(record(info.get("total_token_usage")))
    last = codex_totals(record(info.get("last_token_usage")))
    if total is None:
        state.discard("invalid_counters")
        state.previous = None
        return ()
    previous = state.previous
    previous_model = state.previous_model
    state.previous = total
    state.previous_model = state.model
    if total == previous or not total["total_tokens"]:
        return ()
    delta = codex_request_delta(total, last, previous, allow_initial_total=not state.gap)
    if delta is None:
        state.discard("unproven_reset")
        return ()
    if previous is None and last is not None and last != total and state.own_start is None:
        state.discard("unattributed_prefix")
    if previous is not None and previous_model != state.model and delta != last:
        state.discard("unattributed_aggregate")
        if last is None or not all(last[key] <= delta[key] for key in delta):
            return ()
        delta = last
    owned = state.own_usage_proven or (
        context.parent_session_id is None and context.inherited_history in {"none", "excluded"}
    )
    if state.model is None:
        state.discard("missing_model")
        return ()
    if state.ownership_blocked or not owned:
        state.discard("unproven_ownership")
        return ()
    values = codex_request_counters(delta, last)
    series = json.dumps(["codex", context.session_id, context.native_id, "history_request"])
    source_key = hashlib.sha256(json.dumps([series, position]).encode()).hexdigest()
    return (
        NativeUsageObservation(
            source_key=source_key,
            series_key=series,
            context=context,
            scope="request",
            counters=values,
            observed_at_ms=observed_at_ms,
            occurred_at_ms=_timestamp(event),
            native_id=context.native_id,
            turn_id=state.turn_id,
            model=state.model,
            observed_model=state.model,
            cumulative=False,
            includes_descendants=False,
            authoritative=context.routing == "native",
            additive=True,
            origin="history",
            source_position=position,
            own_usage_proven=state.own_usage_proven,
            provider_kind=state.provider_kind
            or (
                context.provider_kind
                if state.observed_provider is None or context.observed_model == state.model
                else None
            ),
            observed_provider=state.observed_provider,
        ),
    )


def _provider(state: HistoryState, payload: Mapping[str, Any], context: NativeUsageContext) -> None:
    provider = text(payload.get("model_provider")) or text(payload.get("provider"))
    state.observed_provider = provider
    state.provider_kind = provider if provider in {"openai", "anthropic", "gemini"} else None
    if provider is None:
        state.provider_kind = context.provider_kind
