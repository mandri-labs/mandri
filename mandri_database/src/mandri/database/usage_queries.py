import json
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mandri.core.clock import system_now_ms
from mandri.core.types.usage import UsageFilters
from mandri.database.usage_serialization import metrics


def revision(connection: sqlite3.Connection) -> int:
    return int(connection.execute("SELECT revision FROM usage_revision WHERE id=1").fetchone()[0])


def advance(connection: sqlite3.Connection) -> int:
    connection.execute("UPDATE usage_revision SET revision=revision+1 WHERE id=1")
    return revision(connection)


def overview(connection: sqlite3.Connection, filters: UsageFilters) -> dict[str, Any]:
    try:
        zone = ZoneInfo(filters.timezone)
    except ZoneInfoNotFoundError as error:
        raise ValueError("Unknown usage timezone") from error
    if not 1 <= filters.limit <= 1000 or filters.offset < 0:
        raise ValueError("Invalid usage pagination")
    if (
        filters.from_ms is not None
        and filters.to_ms is not None
        and not 0 < filters.to_ms - filters.from_ms <= 3660 * 86_400_000
    ):
        raise ValueError("Usage range must be increasing and at most 3660 days")
    current_revision = revision(connection)
    if filters.expected_revision is not None and filters.expected_revision != current_revision:
        raise ValueError("Usage revision is stale")
    clauses: list[str] = []
    params: list[object] = []
    for name in ("root_session_id", "project_path", "model"):
        value = getattr(filters, name)
        if value is not None:
            clauses.append(f"f.{name}=?")
            params.append(value)
    if filters.session_id is not None:
        if filters.include_descendants:
            clauses.append("(f.session_id=? OR f.root_session_id=?)")
            params.extend((filters.session_id, filters.session_id))
        else:
            clauses.append("f.session_id=?")
            params.append(filters.session_id)
    if not filters.include_deleted:
        clauses.append("COALESCE(s.deleted,0)=0")
    if filters.from_ms is not None:
        clauses.append("(f.occurred_at IS NULL OR f.occurred_at>=?)")
        params.append(filters.from_ms)
    if filters.to_ms is not None:
        clauses.append(
            "(f.occurred_at IS NULL OR COALESCE("
            "json_extract(f.payload,'$.interval_start'),f.occurred_at)<?)"
        )
        params.append(filters.to_ms)
    for name in ("harness", "provider", "billing_mode"):
        value = getattr(filters, name)
        if value is not None:
            clauses.append(f"json_extract(f.payload,'$.{name}')=?")
            params.append(value)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    records = connection.execute(
        "SELECT f.payload,f.usd_equivalent,p.payload AS price_payload,"
        "COALESCE(s.deleted,0) AS deleted"
        " FROM usage_fact f LEFT JOIN session s ON s.id=COALESCE(f.session_id,f.root_session_id)"
        " LEFT JOIN usage_price p ON p.price_id=f.price_id" + where + " LIMIT 100001",
        params,
    ).fetchall()
    if len(records) > 100000:
        raise ValueError("Usage overview exceeds 100000 facts; narrow the filters")
    selected: list[dict[str, Any]] = []
    undated: list[dict[str, Any]] = []
    unallocated: list[dict[str, Any]] = []
    days: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    sources: dict[str, list[dict[str, Any]]] = {
        key: [] for key in ("gateway", "codex", "claude", "agy", "other")
    }
    last_observed_at: int | None = None
    grouping = {
        "session": "session_id",
        "project": "project_path",
        "model": "model",
        "harness": "harness",
        "billing_mode": "billing_mode",
    }
    if filters.group_by not in grouping:
        raise ValueError("Invalid usage breakdown dimension")
    for record in records:
        row = json.loads(record["payload"])
        row["usd_equivalent"] = record["usd_equivalent"]
        row["deleted"] = bool(record["deleted"])
        row["valuation_basis"] = (
            json.loads(record["price_payload"]).get("valuation_basis", "historical_tariff")
            if record["price_payload"]
            else None
        )
        observed = row.get("observed_at")
        if observed is not None and (last_observed_at is None or observed > last_observed_at):
            last_observed_at = observed
        if any(
            getattr(filters, key) is not None and getattr(filters, key) != row[key]
            for key in ("harness", "provider", "billing_mode")
        ):
            continue
        end = row["occurred_at"]
        start = row["interval_start"] if row["interval_start"] is not None else end
        if end is None:
            undated.append(row)
            if filters.from_ms is not None or filters.to_ms is not None:
                continue
        else:
            if (filters.from_ms is not None and end < filters.from_ms) or (
                filters.to_ms is not None and start >= filters.to_ms
            ):
                continue
            if (filters.from_ms is not None and start < filters.from_ms) or (
                filters.to_ms is not None and end >= filters.to_ms
            ):
                unallocated.append(row)
                continue
            date = datetime.fromtimestamp(end / 1000, UTC).astimezone(zone).date().isoformat()
            start_date = (
                datetime.fromtimestamp(start / 1000, UTC).astimezone(zone).date().isoformat()
            )
            if date == start_date:
                days[date].append(row)
            else:
                unallocated.append(row)
        selected.append(row)
        source = {
            "gateway": "gateway",
            "native:codex": "codex",
            "native:claude": "claude",
            "native:agy": "agy",
        }.get(row["source"], "other")
        sources[source].append(row)
        group_key = row[grouping[filters.group_by]]
        if filters.group_by == "session" and group_key is None:
            group_key = row["root_session_id"]
        groups[group_key].append(row)
    if len(days) > 3660:
        raise ValueError("Usage timeseries exceeds 3660 buckets")
    keys = sorted(groups, key=lambda key: (key is None, key or ""))
    status_counts = {
        str(row[0] or "unknown"): int(row[1])
        for row in connection.execute(
            "SELECT json_extract(payload,'$.status'),count(*) FROM usage_cursor GROUP BY 1"
        )
    }
    gap_count = int(connection.execute("SELECT count(*) FROM usage_gap").fetchone()[0])
    discarded: Counter[str] = Counter()
    for checkpoint in connection.execute("SELECT payload FROM usage_cursor"):
        parser_state = json.loads(json.loads(checkpoint[0]).get("state_json", "{}"))
        discarded.update(parser_state.get("discard_reasons", {}))
    source_count = sum(status_counts.values())
    status = (
        "partial"
        if gap_count or any(key not in ("ready", "complete") for key in status_counts)
        else "observed"
        if source_count
        else "unavailable"
    )
    return {
        "revision": current_revision,
        "as_of": system_now_ms(),
        "last_observed_at": last_observed_at,
        "history_status": status_counts,
        "timezone": filters.timezone,
        "summary": metrics(selected),
        "source_breakdown": {key: metrics(rows) for key, rows in sources.items()},
        "timeseries": [{"date": day, **metrics(rows)} for day, rows in sorted(days.items())],
        "breakdown": [
            {
                "key": key,
                **metrics(groups[key]),
                **(
                    {"deleted": any(row["deleted"] for row in groups[key])}
                    if filters.group_by == "session"
                    else {}
                ),
            }
            for key in keys[filters.offset : filters.offset + filters.limit]
        ],
        "breakdown_total": len(keys),
        "undated": metrics(undated),
        "unallocated": metrics(unallocated),
        "sync_state": {
            "discarded_event_count": sum(discarded.values()),
            "discard_reasons": dict(discarded),
            "scope": "daemon",
            "status": status,
            "source_count": source_count,
            "gap_count": gap_count,
            "status_counts": status_counts,
        },
        "catalog": {
            str(item[0]): json.loads(item[1])
            for item in connection.execute("SELECT source,payload FROM usage_catalog_state")
        },
    }
