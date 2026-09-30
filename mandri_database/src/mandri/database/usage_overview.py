import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

from mandri.core.types.usage import UsageFilters
from mandri.database.usage_metrics import MergedUsageMetrics, UsageMetrics
from mandri.database.usage_serialization import COUNTERS


def aggregate(
    connection: sqlite3.Connection, filters: UsageFilters, zone: ZoneInfo
) -> dict[str, Any]:
    grouping = {
        "session": "COALESCE(f.session_id,f.root_session_id)",
        "project": "f.project_path",
        "model": "f.model",
        "harness": "f.harness",
        "billing_mode": "f.billing_mode",
    }
    if filters.group_by not in grouping:
        raise ValueError("Invalid usage breakdown dimension")
    clauses, params = [], []
    for name in ("root_session_id", "project_path", "model", "harness", "provider", "billing_mode"):
        value = getattr(filters, name)
        if value is not None:
            clauses.append(f"f.{name}=?")
            params.append(value)
    if filters.session_id is not None:
        clauses.append(
            "(f.session_id=? OR f.root_session_id=?)"
            if filters.include_descendants
            else "f.session_id=?"
        )
        params.extend([filters.session_id] * (2 if filters.include_descendants else 1))
    if not filters.include_deleted:
        clauses.append("COALESCE(s.deleted,0)=0")
    if filters.from_ms is not None:
        clauses.append("(f.occurred_at IS NULL OR f.occurred_at>=?)")
        params.append(filters.from_ms)
    if filters.to_ms is not None:
        clauses.append("(f.occurred_at IS NULL OR COALESCE(f.interval_start,f.occurred_at)<?)")
        params.append(filters.to_ms)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    columns = ",".join(f"f.{field}" for field in COUNTERS)
    raw_metrics = (
        "usage_metrics("
        + ",".join(COUNTERS)
        + (",usd_equivalent,reported_cost_usd,complete,model,valuation_basis)")
    )
    boundary = []
    if filters.from_ms is not None:
        boundary.append(f"COALESCE(f.interval_start,f.occurred_at)<{int(filters.from_ms)}")
    if filters.to_ms is not None:
        boundary.append(f"f.occurred_at>={int(filters.to_ms)}")
    crossing = " OR ".join(boundary) or "0"
    undated_selected = int(filters.from_ms is None and filters.to_ms is None)
    metrics = "usage_merge(metrics)"
    query = (
        f"WITH raw AS (SELECT {columns},f.usd_equivalent,f.reported_cost_usd,"
        "f.complete,f.model,p.valuation_basis,f.observed_at,f.occurred_at,"
        f"{grouping[filters.group_by]} AS group_key,COALESCE(s.deleted,0) AS deleted,"
        "CASE WHEN f.source='gateway'"
        " OR json_extract(f.payload,'$.pricing_context.attributed_source')='gateway' THEN 'gateway'"
        " WHEN f.source='native:codex' THEN 'codex'"
        " WHEN f.source='native:claude' THEN 'claude' WHEN f.source='native:agy' THEN 'agy'"
        " ELSE 'other' END AS source,"
        "usage_day(f.occurred_at) AS day,"
        "usage_day(COALESCE(f.interval_start,f.occurred_at)) AS start_day,"
        "f.occurred_at IS NULL AS undated,"
        f"CASE WHEN f.occurred_at IS NULL THEN {undated_selected}"
        f" ELSE NOT ({crossing}) END AS selected"
        " FROM usage_fact f LEFT JOIN session s ON s.id=COALESCE(f.session_id,f.root_session_id)"
        " LEFT JOIN usage_price p ON p.price_id=f.price_id" + where + "),"
        " facts AS MATERIALIZED (SELECT source,group_key,day,start_day,selected,undated,"
        f"{raw_metrics} AS metrics,MAX(deleted) AS deleted,MAX(observed_at) AS observed_at"
        " FROM raw GROUP BY source,group_key,day,start_day,selected,undated) "
        f"SELECT 'summary' AS kind,NULL AS key,{metrics} AS metrics,0 AS deleted"
        " FROM facts WHERE selected"
        f" UNION ALL SELECT 'source',source,{metrics},0 FROM facts WHERE selected GROUP BY source"
        f" UNION ALL SELECT 'undated',NULL,{metrics},0 FROM facts WHERE undated"
        f" UNION ALL SELECT 'unallocated',NULL,{metrics},0 FROM facts"
        " WHERE NOT undated AND (NOT selected OR day<>start_day)"
        f" UNION ALL SELECT * FROM (SELECT 'day',day,{metrics},0 FROM facts"
        " WHERE selected AND day=start_day GROUP BY day ORDER BY day LIMIT 3661)"
        f" UNION ALL SELECT * FROM (SELECT 'group',group_key,{metrics},MAX(deleted) FROM facts"
        " WHERE selected GROUP BY group_key ORDER BY group_key IS NULL,group_key LIMIT ? OFFSET ?)"
        " UNION ALL SELECT 'total',NULL,CAST(COUNT(*) AS TEXT),0 FROM"
        " (SELECT group_key FROM facts WHERE selected GROUP BY group_key)"
        " UNION ALL SELECT 'observed',NULL,CAST(MAX(observed_at) AS TEXT),0 FROM facts"
    )

    def day(timestamp: int | None) -> str | None:
        if timestamp is None:
            return None
        return datetime.fromtimestamp(timestamp / 1000, UTC).astimezone(zone).date().isoformat()

    connection.create_function("usage_day", 1, day, deterministic=True)
    register = cast(
        Callable[[str, int, type[UsageMetrics] | None], None], connection.create_aggregate
    )
    register("usage_metrics", 12, UsageMetrics)
    register("usage_merge", 1, MergedUsageMetrics)
    try:
        rows = connection.execute(query, (*params, filters.limit, filters.offset)).fetchall()
    finally:
        connection.create_function("usage_day", 1, None)
        register("usage_metrics", 12, None)
        register("usage_merge", 1, None)
    result: dict[str, Any] = {
        "source_breakdown": {
            key: UsageMetrics().result() for key in ("gateway", "codex", "claude", "agy", "other")
        },
        "timeseries": [],
        "breakdown": [],
    }
    for kind, key, payload, deleted in rows:
        if kind in ("total", "observed"):
            result["breakdown_total" if kind == "total" else "last_observed_at"] = (
                int(payload) if payload is not None else None
            )
            continue
        value = json.loads(payload) if payload else UsageMetrics().result()
        if kind in ("summary", "undated", "unallocated"):
            result[kind] = value
        elif kind == "source":
            result["source_breakdown"][key] = value
        elif kind == "day":
            result["timeseries"].append({"date": key, **value})
        else:
            result["breakdown"].append(
                {
                    "key": key,
                    **value,
                    **({"deleted": bool(deleted)} if filters.group_by == "session" else {}),
                }
            )
    if len(result["timeseries"]) > 3660:
        raise ValueError("Usage timeseries exceeds 3660 buckets")
    return result
