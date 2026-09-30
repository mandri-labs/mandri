from collections.abc import Sequence
from decimal import Decimal, localcontext

from mandri.core.types.usage import UsageObservation, UsagePrice
from mandri.core.usage_price_identity import pricing_provider
from mandri.core.usage_price_schedule import matches_weekly_intervals, validate_weekly_intervals

CALCULATION_VERSION = "1"
TOKEN_FIELDS = (
    "input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
)
OVERRIDE_SOURCES = {"explicit", "user_override"}
REFERENCE_ASSUMPTIONS: dict[str, object] = {
    "modality": "text",
    "service_tier": "standard",
    "region": "global",
    "modifiers": [],
    "cache_write_ttl_seconds": 300,
    "cache_mode": "implicit",
}


def disjoint_tokens(observation: UsageObservation) -> dict[str, int] | None:
    """Normalize only explicitly established cache and reasoning overlap semantics."""
    counters = {}
    for key in TOKEN_FIELDS:
        value = getattr(observation, key)
        if type(value) is not int or value < 0:
            return None
        counters[key] = value
    if type(observation.input_includes_cache) is not bool:
        return None
    if type(observation.output_includes_reasoning) is not bool:
        return None
    if observation.input_includes_cache:
        counters["input_tokens"] -= counters["cache_read_tokens"] + counters["cache_write_tokens"]
    if observation.output_includes_reasoning:
        counters["output_tokens"] -= counters["reasoning_tokens"]
    return counters if all(value >= 0 for value in counters.values()) else None


def validate_price(price: UsagePrice) -> None:
    """Reject unsupported units, rules, rates and malformed effective intervals."""
    if not price.price_id or not price.provider or not price.model or not price.source:
        raise ValueError("Price identity and source are required")
    if price.currency != "USD" or price.unit != "per_million_tokens":
        raise ValueError("Only USD per million tokens is supported")
    if price.calculation_version != CALCULATION_VERSION:
        raise ValueError("Unsupported calculation version")
    if price.valuation_basis not in {"historical_tariff", "current_price_comparison"}:
        raise ValueError("Unsupported valuation basis")
    if price.valuation_basis == "current_price_comparison" and (
        type(price.reviewed_at) is not int or price.reviewed_at < 0
    ):
        raise ValueError("Current price comparisons require a review timestamp")
    if type(price.effective_from) is not int or price.effective_from < 0:
        raise ValueError("Invalid effective_from")
    if price.effective_to is not None and (
        type(price.effective_to) is not int or price.effective_to <= price.effective_from
    ):
        raise ValueError("Invalid effective_to")
    if not price.rates or set(price.rates) - set(TOKEN_FIELDS):
        raise ValueError("Unsupported token rate dimensions")
    for rate in price.rates.values():
        if not isinstance(rate, Decimal) or not rate.is_finite() or rate < 0:
            raise ValueError("Rates must be finite nonnegative Decimals")
    for key in ("min_context_tokens", "max_context_tokens", "cache_write_ttl_seconds"):
        if key in price.constraints:
            value = price.constraints[key]
            if type(value) is not int or value <= 0:
                raise ValueError(f"Invalid {key}")
    minimum = price.constraints.get("min_context_tokens", 0)
    maximum = price.constraints.get("max_context_tokens")
    if isinstance(minimum, int) and isinstance(maximum, int) and minimum > maximum:
        raise ValueError("Invalid context interval")
    if "utc_weekly_intervals" in price.constraints:
        validate_weekly_intervals(price.constraints["utc_weekly_intervals"])


def _priced_tokens(
    observation: UsageObservation,
    price: UsagePrice,
) -> tuple[dict[str, int], dict[str, Decimal]] | None:
    rates = dict(price.rates)
    if all(rates.get(key) == 0 for key in TOKEN_FIELDS):
        if any(
            value is not None and (type(value) is not int or value < 0)
            for key in TOKEN_FIELDS
            for value in (getattr(observation, key),)
        ):
            return None
        return {}, rates
    if (
        price.source in OVERRIDE_SOURCES
        and observation.output_includes_reasoning is True
        and "reasoning_tokens" not in rates
        and "output_tokens" in rates
    ):
        rates["reasoning_tokens"] = rates["output_tokens"]
    counters: dict[str, int] = {}
    groups = (
        (
            "input_tokens",
            ("cache_read_tokens", "cache_write_tokens"),
            observation.input_includes_cache,
        ),
        ("output_tokens", ("reasoning_tokens",), observation.output_includes_reasoning),
    )
    for parent, children, inclusive in groups:
        quantity = getattr(observation, parent)
        if type(quantity) is not int or quantity < 0 or type(inclusive) is not bool:
            return None
        counters[parent] = quantity
        for child in children:
            subset = getattr(observation, child)
            if subset is None:
                if inclusive and (
                    quantity == 0 or (parent in rates and rates.get(child) == rates[parent])
                ):
                    continue
                return None
            if type(subset) is not int or subset < 0:
                return None
            counters[child] = subset
            if inclusive:
                counters[parent] -= subset
        if counters[parent] < 0:
            return None
    if any(count and key not in rates for key, count in counters.items()):
        return None
    return counters, rates


def request_context_tokens(observation: UsageObservation) -> int | None:
    """Read request prompt size without mistaking cumulative session volume for context."""
    if observation.kind != "delta" or observation.baseline:
        return None
    if type(observation.request_count) is int and observation.request_count > 1:
        return None
    explicit = observation.pricing_context.get("context_tokens")
    if "context_tokens" in observation.pricing_context:
        return explicit if type(explicit) is int and explicit >= 0 else None
    if type(observation.request_count) is not int:
        return None
    if observation.request_count != 1:
        return None
    inputs = observation.input_tokens
    if type(inputs) is not int or inputs < 0:
        return None
    if observation.input_includes_cache is True:
        return inputs
    if observation.input_includes_cache is not False:
        return None
    for subset in (observation.cache_read_tokens, observation.cache_write_tokens):
        if type(subset) is not int or subset < 0:
            return None
        inputs += subset
    return inputs


def _applicable(price: UsagePrice, observation: UsageObservation, counters: dict[str, int]) -> bool:
    when = observation.occurred_at
    start = observation.interval_start
    if when is not None and (type(when) is not int or when < 0):
        return False
    if start is not None and (
        type(start) is not int or start < 0 or (when is not None and start > when)
    ):
        return False
    if price.valuation_basis == "historical_tariff":
        if when is None or when < price.effective_from:
            return False
        if price.effective_to is not None and when >= price.effective_to:
            return False
        if start is not None and start < price.effective_from:
            return False
    context = observation.pricing_context
    if context.get("comparison_basis") == "standard_text_api":
        context = {**REFERENCE_ASSUMPTIONS, **context}
        if (
            price.valuation_basis == "current_price_comparison"
            and price.source not in OVERRIDE_SOURCES
            and price.constraints.get("modality") == "text"
            and context.get("modality") == "multimodal"
        ):
            context = {**context, "modality": "text"}
    for key, expected in price.constraints.items():
        if key in {"min_context_tokens", "max_context_tokens"}:
            actual = request_context_tokens(observation)
            if actual is None or not isinstance(expected, int):
                return False
            if key == "max_context_tokens" and actual > expected:
                return False
            if key == "min_context_tokens" and actual < expected:
                return False
        elif key == "utc_weekly_intervals":
            if (
                type(observation.request_count) is not int
                or observation.request_count != 1
                or not matches_weekly_intervals(when, start, expected)
            ):
                return False
        elif key == "cache_write_ttl_seconds":
            if counters.get("cache_write_tokens", 0) and context.get(key) != expected:
                return False
        elif key == "cache_mode":
            if counters.get("cache_read_tokens", 0) and context.get(key) != expected:
                return False
        elif key not in context or context[key] != expected:
            return False
    return True


def _source_priority(price: UsagePrice) -> tuple[int, bool]:
    rank = (
        4
        if price.source in OVERRIDE_SOURCES
        else 3
        if price.provider == "openrouter" and price.source == "https://openrouter.ai/api/v1/models"
        else 2
        if price.source == "https://models.dev/api.json"
        else 1
    )
    direct = (
        price.source_model is None
        or price.source_model == price.model
        or price.source_model.partition("/")[2] == price.model
    )
    return rank, direct


def value_usage(
    observation: UsageObservation, prices: Sequence[UsagePrice]
) -> tuple[Decimal | None, str | None]:
    """Value one normalized delta using an exact applicable tariff, never an invoice charge."""
    if observation.kind != "delta" or observation.baseline:
        return None, None
    matching = []
    for price in prices:
        provider = pricing_provider(observation)
        instance_override = (
            price.source in OVERRIDE_SOURCES and price.provider == observation.provider
        )
        if (
            price.provider != provider and not instance_override
        ) or price.model != observation.model:
            continue
        try:
            validate_price(price)
        except ValueError:
            continue
        matching.append(price)
    current = [price for price in matching if price.valuation_basis == "current_price_comparison"]
    source_rank = max((_source_priority(price) for price in current), default=(0, False))
    latest = max(
        (price.reviewed_at or 0 for price in current if _source_priority(price) == source_rank),
        default=0,
    )
    candidates = []
    for price in matching:
        if price.valuation_basis == "current_price_comparison" and (
            _source_priority(price) != source_rank or price.reviewed_at != latest
        ):
            continue
        quantities = _priced_tokens(observation, price)
        if quantities is not None and _applicable(price, observation, quantities[0]):
            candidates.append(price)
    if not candidates:
        return None, None

    def priority(price: UsagePrice) -> tuple[bool, bool, bool, int]:
        return (
            price.source in OVERRIDE_SOURCES,
            price.provider == observation.provider,
            price.valuation_basis == "current_price_comparison",
            price.reviewed_at or 0
            if price.valuation_basis == "current_price_comparison"
            else price.effective_from,
        )

    best = max(priority(price) for price in candidates)
    selected = [price for price in candidates if priority(price) == best]
    price = selected[0]
    if any(other != price for other in selected[1:]):
        return None, None
    quantities = _priced_tokens(observation, price)
    assert quantities is not None
    counters, rates = quantities
    precision = (
        max(
            len(str(counters.get(key, 0)))
            + len(rate.as_tuple().digits)
            + abs(int(rate.as_tuple().exponent))
            for key, rate in rates.items()
        )
        + 20
    )
    with localcontext() as context:
        context.prec = max(28, precision)
        amount = sum(
            (Decimal(counters.get(key, 0)) * rate for key, rate in rates.items()), Decimal(0)
        )
        return amount / Decimal(1_000_000), price.price_id
