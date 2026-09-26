from mandri.core.types.usage import UsageObservation
from mandri.sessions.usage.counters import disjoint_total
from mandri.sessions.usage.types import NativeUsageObservation


def to_usage_observation(item: NativeUsageObservation, *, sequence: int) -> UsageObservation:
    context = item.context
    counters = item.counters
    eligible = (
        item.authoritative
        and (context.parent_session_id is None or item.own_usage_proven)
        and (context.inherited_history in {"none", "excluded"} or item.own_usage_proven)
        and item.scope
        in {"thread", "call_model", "process_session", "turn_main_loop", "request", "message"}
        and (item.scope != "turn_main_loop" or item.turn_id is not None)
        and (item.scope not in {"request", "message"} or item.model is not None)
        and not item.anomalies
        and (context.native_id is None or item.native_id == context.native_id)
    )
    return UsageObservation(
        source=f"native:{context.harness}",
        source_key=item.source_key,
        fact_key=f"native:{item.series_key}:{item.source_key}",
        session_id=context.session_id,
        project_path=context.project_path,
        harness=context.harness,
        billing_mode=context.billing_mode,
        model=item.model,
        native_session_id=item.native_id,
        turn_id=item.turn_id or context.turn_id,
        observed_model=item.observed_model or item.model,
        observed_at=item.observed_at_ms,
        occurred_at=item.occurred_at_ms,
        interval_start=item.occurred_at_ms if item.scope in {"request", "message"} else None,
        input_tokens=counters.get("input_tokens"),
        output_tokens=counters.get("output_tokens"),
        cache_read_tokens=counters.get("cache_read_tokens"),
        cache_write_tokens=counters.get("cache_write_tokens"),
        reasoning_tokens=counters.get("reasoning_tokens"),
        total_tokens=counters.get("total_tokens"),
        native_total_tokens=counters.get("reported_total_tokens"),
        request_count=counters.get("request_count"),
        kind="cumulative" if item.cumulative else "delta",
        epoch=item.series_key,
        sequence=item.source_position if item.source_position is not None else sequence,
        authoritative=eligible,
        complete=(
            item.scope in {"request", "message"}
            and item.model is not None
            and not item.anomalies
            and counters.get("request_count") == 1
            and all(
                counters.get(name) is not None
                for name in (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                    "total_tokens",
                )
            )
        ),
        input_includes_cache=context.harness == "codex",
        output_includes_reasoning=True if context.harness in {"codex", "claude", "pi"} else None,
        reported_cost_usd=item.reported_cost_usd,
        reported_cost_basis="harness_estimate" if item.reported_cost_usd is not None else None,
        pricing_context={
            "comparison_basis": "standard_text_api",
            "provider_kind": (
                item.provider_kind
                if context.harness == "pi"
                or (item.origin == "history" and context.harness == "codex")
                else context.provider_kind
                or (
                    {"codex": "openai", "claude": "anthropic"}.get(context.harness)
                    if item.origin != "history"
                    else None
                )
            ),
            **({"observed_provider": item.observed_provider} if item.observed_provider else {}),
            **(
                {"upstream_request_id": item.upstream_request_id}
                if item.upstream_request_id
                else {}
            ),
            **({"evidence": "history_request"} if item.origin == "history" else {}),
            **(
                {
                    "context_tokens": counters.get("input_tokens")
                    if context.harness == "codex"
                    else disjoint_total(
                        counters, "input_tokens", "cache_read_tokens", "cache_write_tokens"
                    )
                }
                if item.scope in {"request", "message"} and counters.get("request_count") == 1
                else {}
            ),
        }
        if context.harness in {"codex", "claude", "pi"}
        else {},
    )
