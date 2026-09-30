import json
import sqlite3
from collections.abc import Sequence

from mandri.core.types.usage import UsageObservation, UsagePrice
from mandri.core.usage_price_identity import PUBLIC_PRICE_SOURCES, pricing_provider
from mandri.core.usage_pricing import validate_price
from mandri.database.usage_queries import advance, revision
from mandri.database.usage_serialization import encode, price


def register_prices(
    db: sqlite3.Connection, values: Sequence[UsagePrice], *, public: bool = False
) -> int:
    for value in values:
        validate_price(value)
        if public and (
            value.source not in PUBLIC_PRICE_SOURCES
            or value.valuation_basis != "current_price_comparison"
        ):
            raise ValueError("Public catalog synchronization requires downloaded current prices")
    changed = False
    for value in values:
        payload = encode(value)
        row = db.execute(
            "SELECT payload FROM usage_price WHERE price_id=?", (value.price_id,)
        ).fetchone()
        if row:
            if row[0] != payload and encode(price(row[0])) != payload:
                raise ValueError("Price versions are immutable")
            continue
        db.execute("INSERT INTO usage_price VALUES (?,?)", (value.price_id, payload))
        changed = True
    if public:
        deleted = db.execute(
            "DELETE FROM usage_price WHERE valuation_basis='current_price_comparison'"
            " AND json_extract(payload,'$.source') IN (?,?)"
            " AND price_id NOT IN (SELECT value FROM json_each(?))",
            (*sorted(PUBLIC_PRICE_SOURCES), json.dumps([value.price_id for value in values])),
        ).rowcount
        changed = changed or bool(deleted)
    if changed:
        db.execute(
            "UPDATE usage_valuation SET catalog_revision=catalog_revision+1,"
            " after_key=NULL WHERE id=1"
        )
    return advance(db) if changed else revision(db)


def prices_for(db: sqlite3.Connection, fact: UsageObservation) -> list[UsagePrice]:
    if fact.model is None:
        return []
    provider = _provider(fact)
    rows = db.execute(
        "SELECT payload FROM usage_price WHERE model=? AND provider IN (?,?)",
        (fact.model, provider, fact.provider),
    )
    return [price(row[0]) for row in rows]


def _provider(fact: UsageObservation) -> str | None:
    return pricing_provider(fact)


class PriceIndex:
    def __init__(self) -> None:
        self._prices: dict[tuple[str | None, str | None, str | None], list[UsagePrice]] = {}

    def for_fact(self, db: sqlite3.Connection, fact: UsageObservation) -> Sequence[UsagePrice]:
        key = fact.model, fact.provider, _provider(fact)
        if key not in self._prices:
            self._prices[key] = prices_for(db, fact)
        return self._prices[key]
