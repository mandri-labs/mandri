import asyncio
import sqlite3
from dataclasses import dataclass
from typing import Any

from mandri.core.types.usage import UsageObservation, UsagePrice
from mandri.core.usage_pricing import value_usage
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage_prices import PriceIndex
from mandri.database.usage_queries import advance, revision
from mandri.database.usage_serialization import observation
from mandri.database.usage_transactions import transaction


@dataclass(frozen=True)
class ValuationInput:
    key: str
    payload: str
    amount: str | None
    price_id: str | None
    fact: UsageObservation
    prices: tuple[UsagePrice, ...]


def _compute(rows: list[ValuationInput]) -> list[tuple[ValuationInput, str | None, str | None]]:
    changes = []
    for row in rows:
        amount, price_id = value_usage(row.fact, row.prices)
        serialized = str(amount) if amount is not None else None
        if (serialized, price_id) != (row.amount, row.price_id):
            changes.append((row, serialized, price_id))
    return changes


async def revalue_batch(
    database: AiosqliteDatabase, limit: int, after_key: str | None, all_facts: bool
) -> dict[str, Any]:
    if not 1 <= limit <= 1000:
        raise ValueError("Invalid valuation batch limit")

    def read(db: sqlite3.Connection) -> list[ValuationInput]:
        prices = PriceIndex()
        rows = db.execute(
            "SELECT fact_key,payload,usd_equivalent,price_id FROM usage_fact WHERE "
            + ("1=1" if all_facts else "price_id IS NULL")
            + " AND fact_key>? ORDER BY fact_key LIMIT ?",
            (after_key or "", limit),
        )
        result = []
        for key, payload, amount, price_id in rows:
            fact = observation(payload)
            result.append(
                ValuationInput(
                    key, payload, amount, price_id, fact, tuple(prices.for_fact(db, fact))
                )
            )
        return result

    rows = await transaction(database, read, write=False)
    changes = await asyncio.to_thread(_compute, rows)

    def save(db: sqlite3.Connection) -> dict[str, Any]:
        changed = 0
        for row, amount, price_id in changes:
            changed += db.execute(
                "UPDATE usage_fact SET usd_equivalent=?,price_id=? WHERE fact_key=?"
                " AND payload=? AND usd_equivalent IS ? AND price_id IS ?",
                (amount, price_id, row.key, row.payload, row.amount, row.price_id),
            ).rowcount
        return {
            "revision": advance(db) if changed else revision(db),
            "updated": changed,
            "next_key": rows[-1].key if len(rows) == limit else None,
        }

    return await transaction(database, save, write=bool(changes))
