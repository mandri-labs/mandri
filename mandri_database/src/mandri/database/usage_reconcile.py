import json
import sqlite3
from collections.abc import Callable, Sequence
from typing import Any

from mandri.core.types.usage import UsageObservation, UsagePrice
from mandri.core.usage_normalization import normalize_usage
from mandri.database.usage_accounting import validate
from mandri.database.usage_coverage import CoverageIndex
from mandri.database.usage_prices import PriceIndex
from mandri.database.usage_queries import advance, revision
from mandri.database.usage_serialization import encode, observation


def stage_history(
    db: sqlite3.Connection,
    values: Sequence[UsageObservation],
    source_key: str,
    cursor: dict[str, Any],
    record: Callable[
        [sqlite3.Connection, UsageObservation, Sequence[UsagePrice] | None, CoverageIndex], int
    ],
    *,
    reset: bool,
    complete: bool,
) -> int:
    session_id = cursor.get("session_id")
    if not source_key or not isinstance(session_id, str) or not session_id:
        raise ValueError("History rebuild requires a source and session")
    existing = db.execute(
        "SELECT session_id,payload FROM usage_cursor WHERE source_key=?",
        (source_key,),
    ).fetchone()
    if existing and existing[0] != session_id:
        raise ValueError("History source cannot change session")
    if complete and not reset and existing and json.loads(existing[1]).get("phase") == "active":
        return revision(db)
    if db.execute("SELECT 1 FROM usage_suppression WHERE session_id=?", (session_id,)).fetchone():
        return revision(db)
    for value in values:
        validate(value)
        if value.session_id != session_id or not value.source.startswith("native:"):
            raise ValueError("History rebuild observation is outside its session")
        if value.pricing_context.get("evidence") != "history_request":
            raise ValueError("History rebuild requires request evidence")
    if reset:
        db.execute("DELETE FROM usage_history_stage WHERE source_key=?", (source_key,))
    for value in values:
        db.execute(
            "INSERT INTO usage_history_stage VALUES (?,?,?,?) ON CONFLICT(source_key,event_key)"
            " DO UPDATE SET payload=excluded.payload",
            (source_key, value.source_key, session_id, encode(value)),
        )
    checkpoint = {**cursor, "phase": "active" if complete else "staging"}
    if complete:
        rows = db.execute(
            "SELECT payload FROM usage_history_stage WHERE source_key=?"
            " ORDER BY json_extract(payload,'$.sequence'),rowid LIMIT 100001",
            (source_key,),
        )
        for table in ("usage_fact", "usage_observation", "usage_baseline"):
            db.execute(
                f"DELETE FROM {table} WHERE session_id=? AND "
                "json_extract(payload,'$.source') LIKE 'native:%'",
                (session_id,),
            )
        db.execute("DELETE FROM usage_gap WHERE session_id=?", (session_id,))
        prices = PriceIndex()
        coverage = CoverageIndex()
        for index, row in enumerate(rows):
            if index >= 100000:
                raise ValueError("History rebuild exceeds 100000 requests per session")
            value = normalize_usage(observation(row[0]))
            record(db, value, prices.for_fact(db, value), coverage)
        db.execute("DELETE FROM usage_history_stage WHERE source_key=?", (source_key,))
        checkpoint["authoritative"] = 1
    old = db.execute(
        "SELECT payload FROM usage_cursor WHERE source_key=?", (source_key,)
    ).fetchone()
    payload = json.dumps(checkpoint, sort_keys=True, separators=(",", ":"))
    if old and old[0] == payload and not values and not reset:
        return revision(db)
    db.execute(
        "INSERT INTO usage_cursor VALUES (?,?,?) ON CONFLICT(source_key)"
        " DO UPDATE SET payload=excluded.payload",
        (source_key, session_id, payload),
    )
    if not complete and old:
        previous = json.loads(old[0])
        fields = ("status", "phase", "authoritative")
        previous_reasons = json.loads(previous.get("state_json", "{}")).get("discard_reasons", {})
        current_reasons = json.loads(checkpoint.get("state_json", "{}")).get("discard_reasons", {})
        if (
            all(previous.get(key) == checkpoint.get(key) for key in fields)
            and previous_reasons == current_reasons
        ):
            return revision(db)
    return advance(db)
