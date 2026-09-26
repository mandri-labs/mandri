import json
from collections import Counter
from dataclasses import asdict
from decimal import Decimal, localcontext
from typing import Any

from mandri.core.types.usage import UsageAccount, UsageObservation, UsagePrice

COUNTERS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "total_tokens",
    "request_count",
)


def encode(value: UsageObservation | UsageAccount | UsagePrice) -> str:
    return json.dumps(asdict(value), default=str, sort_keys=True, separators=(",", ":"))


def observation(payload: str) -> UsageObservation:
    data = json.loads(payload)
    if data["reported_cost_usd"] is not None:
        data["reported_cost_usd"] = Decimal(data["reported_cost_usd"])
    data["included_session_ids"] = tuple(data["included_session_ids"])
    return UsageObservation(**data)


def price(payload: str) -> UsagePrice:
    data = json.loads(payload)
    data["rates"] = {key: Decimal(value) for key, value in data["rates"].items()}
    return UsagePrice(**data)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"fact_count": len(rows)}
    result["missing_fields"] = {}
    for name in COUNTERS:
        known = [row[name] for row in rows if row[name] is not None]
        result[name] = sum(known) if known else None
        result["missing_fields"][name] = len(rows) - len(known)
    for name in ("usd_equivalent", "reported_cost_usd"):
        values = [Decimal(row[name]) for row in rows if row[name] is not None]
        with localcontext() as context:
            context.prec = max(
                28,
                sum(
                    len(value.as_tuple().digits) + abs(int(value.as_tuple().exponent))
                    for value in values
                )
                + 10,
            )
            result[name] = str(sum(values, Decimal(0))) if values else None
    result["unpriced_fact_count"] = sum(row["usd_equivalent"] is None for row in rows)
    result["incomplete_fact_count"] = sum(not row["complete"] for row in rows)
    result["unclassified_fact_count"] = sum(not row.get("model") for row in rows)
    result["valuation_bases"] = dict(
        Counter(row.get("valuation_basis") or "unpriced" for row in rows)
    )
    result["unpriced_reasons"] = dict(
        Counter(
            "model_missing"
            if not row.get("model")
            else "token_usage_missing"
            if row.get("input_tokens") is None or row.get("output_tokens") is None
            else "tariff_or_context_unavailable"
            for row in rows
            if row["usd_equivalent"] is None
        )
    )
    return result
