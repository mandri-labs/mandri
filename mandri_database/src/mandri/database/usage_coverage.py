import hashlib
import json
import sqlite3

from mandri.core.types.usage import UsageObservation
from mandri.database.usage_serialization import observation


def shared_scope(left: UsageObservation, right: UsageObservation) -> bool:
    if left.session_id is not None and right.session_id is not None:
        return left.session_id == right.session_id
    left_ids = {left.session_id, left.root_session_id} - {None}
    right_ids = {right.session_id, right.root_session_id} - {None}
    return bool(left_ids & right_ids) or (
        (not left_ids or not right_ids)
        and (same_request(left, right) if right.source == "gateway" else same_request(right, left))
    )


def gateway_facts(db: sqlite3.Connection, value: UsageObservation) -> list[UsageObservation]:
    rows = db.execute(
        "SELECT f.payload FROM usage_fact f WHERE f.source='gateway'"
        " AND (f.session_id IN (?,?) OR f.root_session_id IN (?,?)"
        " OR (f.session_id IS NULL AND f.root_session_id IS NULL))",
        (
            value.session_id,
            value.root_session_id,
            value.session_id,
            value.root_session_id,
        ),
    ).fetchall()
    return [observation(row["payload"]) for row in rows]


def scoped_facts(db: sqlite3.Connection, value: UsageObservation) -> list[UsageObservation]:
    ids = tuple(sorted({key for key in (value.session_id, value.root_session_id) if key}))
    placeholders = ",".join("?" for _ in ids)
    if value.source.startswith("native"):
        return [item for item in gateway_facts(db, value) if shared_scope(value, item)]
    elif value.source == "gateway":
        identity = request_identity(value)
        response_ids = tuple(
            key
            for key in (
                identity[1] if identity else None,
                value.pricing_context.get("client_response_id"),
            )
            if isinstance(key, str) and key
        )
        identities = ",".join("?" for _ in response_ids)
        scope = f"(session_id IN ({placeholders}) OR root_session_id IN ({placeholders}))"
        params: tuple[object, ...] = (*ids, *ids)
        if identity and response_ids:
            scope += (
                " OR (json_extract(payload,'$.pricing_context.provider_kind')=?"
                " AND json_extract(payload,'$.pricing_context.upstream_request_id')"
                f" IN ({identities}))"
            )
            params += (identity[0], *response_ids)
        rows = db.execute(
            f"SELECT payload FROM usage_observation WHERE ({scope})"
            " AND json_extract(payload,'$.kind')='delta'"
            " AND json_extract(payload,'$.authoritative')=1"
            " AND source LIKE 'native%'"
            f" UNION ALL SELECT payload FROM usage_fact WHERE ({scope})"
            " AND source LIKE 'native%'",
            (*params, *params),
        ).fetchall()
    else:
        return []
    values = {}
    for row in rows:
        item = observation(row["payload"])
        if shared_scope(value, item):
            values[item.fact_key] = item
    return list(values.values())


def request_identity(value: UsageObservation) -> tuple[str, str] | None:
    provider = value.pricing_context.get("provider_kind")
    identity = value.pricing_context.get("upstream_request_id")
    if not identity and value.source == "gateway":
        identity = value.pricing_context.get("client_response_id")
    if (
        value.request_count == 1
        and isinstance(provider, str)
        and provider
        and isinstance(identity, str)
        and identity
    ):
        return provider, identity
    return None


def same_request(native: UsageObservation, gateway: UsageObservation) -> bool:
    native_id, gateway_id = request_identity(native), request_identity(gateway)
    return bool(
        native_id is not None
        and gateway_id is not None
        and native_id[0] == gateway_id[0]
        and (
            native_id[1] == gateway_id[1]
            or native_id[1] == gateway.pricing_context.get("client_response_id")
        )
    )


def gateway_end(value: UsageObservation) -> int | None:
    status = value.pricing_context.get("status")
    if status not in ("completed", "failed", "cancelled") and not (
        status is None and value.complete
    ):
        return None
    start, end = value.occurred_at, value.observed_at
    if start is None or end is None or end < start:
        return None
    return end


def coverage(native: UsageObservation, gateway: UsageObservation) -> tuple[bool, bool]:
    native_id, gateway_id = request_identity(native), request_identity(gateway)
    if native_id is not None and gateway_id is not None:
        duplicate = same_request(native, gateway)
        uncertain_owner = native.session_id is None or gateway.session_id is None
        return duplicate, duplicate and uncertain_owner
    if native.non_overlapping:
        return False, False
    for native_value, gateway_value in (
        (native.model, gateway.model),
        (
            native.pricing_context.get("provider_kind"),
            gateway.pricing_context.get("provider_kind"),
        ),
    ):
        if native_value and gateway_value and native_value != gateway_value:
            return False, False
    native_end = native.occurred_at if native.occurred_at is not None else native.coverage_end
    native_start = native.interval_start
    if native_start is None and native.kind == "delta" and native.epoch is None:
        native_start = native.occurred_at
    if native.pricing_context.get("evidence") == "history_request":
        native_start = native.interval_start if native.interval_start is not None else native_end
    start, end = gateway.occurred_at, gateway_end(gateway)
    if native_end is not None and start is not None and native_end < start:
        return False, False
    if native_start is not None and end is not None and native_start > end:
        return False, False
    bounded_gateway = start is not None and end is not None
    known_native_extent = native_end is not None and (
        native_start is not None or native.coverage_end is not None
    )
    return bounded_gateway and known_native_extent, True


def gap(db: sqlite3.Connection, value: UsageObservation, ambiguous: bool) -> None:
    key = hashlib.sha256(
        json.dumps((value.source, value.epoch, value.fact_key)).encode()
    ).hexdigest()
    if ambiguous:
        db.execute(
            "INSERT OR IGNORE INTO usage_gap VALUES (?,?,?,?)",
            (key, key, value.session_id or value.root_session_id, "ambiguous_gateway_coverage"),
        )
    else:
        db.execute(
            "DELETE FROM usage_gap WHERE source_key=? AND event_key=?"
            " AND reason='ambiguous_gateway_coverage'",
            (key, key),
        )


def select_coverage(db: sqlite3.Connection, fact: UsageObservation) -> bool:
    if fact.source.startswith("native"):
        natives = [fact]
    elif fact.source == "gateway":
        natives = scoped_facts(db, fact)
    else:
        return True
    scopes: dict[tuple[str | None, str | None], dict[str, UsageObservation]] = {}
    keep = True
    for native in natives:
        scope = native.session_id, native.root_session_id
        if scope not in scopes:
            scopes[scope] = {value.fact_key: value for value in gateway_facts(db, native)}
            if fact.source == "gateway":
                scopes[scope][fact.fact_key] = fact
        suppress = ambiguous = False
        for gateway in scopes[scope].values():
            if not shared_scope(native, gateway):
                continue
            duplicate, uncertain = coverage(native, gateway)
            suppress |= duplicate
            ambiguous |= uncertain
            if suppress and ambiguous:
                break
        if (
            fact.source == "gateway"
            and not suppress
            and not db.execute(
                "SELECT 1 FROM usage_fact WHERE fact_key=?", (native.fact_key,)
            ).fetchone()
        ):
            ambiguous = True
        gap(db, native, ambiguous)
        if suppress:
            db.execute("DELETE FROM usage_fact WHERE fact_key=?", (native.fact_key,))
            if native.fact_key == fact.fact_key:
                keep = False
    return keep
