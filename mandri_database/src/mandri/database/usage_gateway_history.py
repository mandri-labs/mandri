import sqlite3
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace

from mandri.core.types.usage import UsageObservation
from mandri.database.usage_queries import advance, revision
from mandri.database.usage_serialization import observation


def fingerprint(value: UsageObservation) -> tuple[object, ...] | None:
    context = value.pricing_context
    counters = (
        value.input_tokens,
        value.output_tokens,
        value.cache_read_tokens,
        value.cache_write_tokens,
    )
    if (
        value.kind != "delta"
        or value.request_count != 1
        or not value.model
        or not context.get("route_id")
        or not context.get("provider_kind")
        or value.input_includes_cache is None
        or any(counter is None for counter in counters)
        or not value.output_tokens
    ):
        return None
    prompt = value.input_tokens
    if not value.input_includes_cache:
        prompt = (prompt or 0) + (value.cache_read_tokens or 0) + (value.cache_write_tokens or 0)
    return (
        context["route_id"],
        context["provider_kind"],
        value.model,
        prompt,
        *counters[1:],
    )


def matches(native: UsageObservation, gateway: UsageObservation) -> bool:
    context = native.pricing_context
    if (
        native.source != "native:claude"
        or context.get("evidence") != "history_request"
        or context.get("model_basis") != "historical_gateway_route"
        or context.get("attributed_source") != "gateway"
        or native.non_overlapping
        or gateway.pricing_context.get("usage_protocol") != "openai"
        or gateway.pricing_context.get("status") != "completed"
        or gateway.pricing_context.get("client_response_id")
        or (gateway.session_id is not None and gateway.session_id != native.session_id)
        or (
            gateway.root_session_id is not None
            and gateway.root_session_id not in {native.session_id, native.root_session_id}
        )
        or native.occurred_at is None
        or gateway.occurred_at is None
        or gateway.observed_at is None
        or not gateway.occurred_at <= native.occurred_at <= gateway.observed_at
    ):
        return False
    identity = fingerprint(native)
    return identity is not None and identity == fingerprint(gateway)


def reconcile(
    db: sqlite3.Connection, save: Callable[[sqlite3.Connection, UsageObservation], None]
) -> int:
    natives = {
        value.fact_key: value
        for row in db.execute(
            "SELECT payload FROM usage_observation WHERE source='native:claude'"
            " AND json_extract(payload,'$.authoritative')=1"
            " AND json_extract(payload,'$.pricing_context.attributed_source')='gateway'"
        )
        if (value := observation(row[0])).pricing_context.get("evidence") == "history_request"
    }
    gateways = [
        observation(row[0])
        for row in db.execute(
            "SELECT payload FROM usage_fact WHERE source='gateway'"
        )
    ]
    by_counters: dict[tuple[object, ...], list[UsageObservation]] = defaultdict(list)
    for gateway in gateways:
        if (identity := fingerprint(gateway)) is not None:
            by_counters[identity].append(gateway)
    candidates: dict[str, list[UsageObservation]] = defaultdict(list)
    owners: dict[str, set[str]] = defaultdict(set)
    for native in natives.values():
        identity = fingerprint(native)
        for gateway in by_counters.get(identity, ()) if identity is not None else ():
            if matches(native, gateway):
                candidates[native.fact_key].append(gateway)
                owners[gateway.fact_key].add(native.fact_key)
    linked = {
        values[0].fact_key: natives[key]
        for key, values in candidates.items()
        if len(values) == 1 and len(owners[values[0].fact_key]) == 1
    }
    changed = False
    for gateway in gateways:
        context = dict(gateway.pricing_context)
        previous = context.pop("historical_native_fact_key", None)
        context.pop("historical_identity_evidence", None)
        linked_native = linked.get(gateway.fact_key)
        if (
            previous
            and (linked_native is None or linked_native.fact_key != previous)
            and previous in natives
        ):
            save(db, natives[previous])
            changed = True
        if linked_native is not None:
            context["historical_native_fact_key"] = linked_native.fact_key
            context["historical_identity_evidence"] = "unique_route_counters_request_interval"
            changed |= bool(
                db.execute(
                    "DELETE FROM usage_fact WHERE fact_key=?", (linked_native.fact_key,)
                ).rowcount
            )
        if context != gateway.pricing_context:
            save(db, replace(gateway, pricing_context=context))
            changed = True
    return advance(db) if changed else revision(db)
