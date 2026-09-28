import sqlite3
from collections.abc import Sequence

from mandri.core.types.usage import UsageObservation, UsagePrice
from mandri.database.usage_serialization import price


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
    value = fact.pricing_context.get("provider_kind", fact.provider)
    return value if isinstance(value, str) else None


class PriceIndex:
    def __init__(self) -> None:
        self._prices: dict[tuple[str | None, str | None, str | None], list[UsagePrice]] = {}

    def for_fact(self, db: sqlite3.Connection, fact: UsageObservation) -> Sequence[UsagePrice]:
        key = fact.model, fact.provider, _provider(fact)
        if key not in self._prices:
            self._prices[key] = prices_for(db, fact)
        return self._prices[key]
