import json
import sqlite3
from collections import Counter
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mandri.core.clock import system_now_ms
from mandri.core.types.usage import UsageFilters
from mandri.database.usage_overview import aggregate


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
    result = aggregate(connection, filters, zone)
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
        **result,
        "history_status": status_counts,
        "timezone": filters.timezone,
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
