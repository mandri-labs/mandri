from dataclasses import replace
from decimal import Decimal

from mandri.core.types.usage import UsageObservation
from mandri.database.usage_serialization import COUNTERS


def validate(value: UsageObservation) -> None:
    if not value.source or not value.source_key or not value.fact_key:
        raise ValueError("Usage source and fact identities must be nonempty")
    for name in (
        *COUNTERS,
        "native_total_tokens",
        "sequence",
        "occurred_at",
        "observed_at",
        "interval_start",
    ):
        number = getattr(value, name)
        if number is not None and (type(number) is not int or number < 0):
            raise ValueError(f"Invalid usage {name}")
    if value.kind not in ("delta", "cumulative"):
        raise ValueError("Invalid usage kind")
    if value.kind == "cumulative" and not value.epoch:
        raise ValueError("Cumulative usage requires an explicit counter epoch")
    if value.baseline and value.kind != "cumulative":
        raise ValueError("Only cumulative usage supports baselines")
    if value.interval_start is not None and (
        value.occurred_at is None or value.interval_start > value.occurred_at
    ):
        raise ValueError("Invalid usage interval")
    cost = value.reported_cost_usd
    if cost is not None and (not isinstance(cost, Decimal) or not cost.is_finite() or cost < 0):
        raise ValueError("Invalid reported USD amount")


def advances(current: UsageObservation, previous: UsageObservation) -> bool:
    if current.sequence != previous.sequence:
        return current.sequence > previous.sequence
    if (
        current.source_key == previous.source_key
        or current.occurred_at is None
        or current.occurred_at != previous.occurred_at
        or current.baseline
    ):
        return False
    increased = False
    for name in (*COUNTERS, "reported_cost_usd"):
        before, after = getattr(previous, name), getattr(current, name)
        if before is not None:
            if after is None or after < before:
                return False
            increased = increased or after > before
    return increased


def contribution(
    current: UsageObservation, previous: UsageObservation | None
) -> UsageObservation | None:
    if previous is None:
        known = [getattr(current, name) for name in COUNTERS if getattr(current, name) is not None]
        if known and not any(known) and not current.reported_cost_usd:
            return None
        return (
            None
            if current.baseline
            else replace(
                current, occurred_at=None, interval_start=None, coverage_end=current.occurred_at
            )
        )
    if current.baseline:
        raise ValueError("A cumulative baseline cannot replace an existing epoch")
    if current.session_id != previous.session_id:
        raise ValueError("A counter epoch cannot change its session")
    if not advances(current, previous):
        return None
    if (
        current.occurred_at is not None
        and previous.occurred_at is not None
        and current.occurred_at < previous.occurred_at
    ):
        raise ValueError("Cumulative observation time moved backwards")
    differences: dict[str, int | None] = {}
    for name in COUNTERS:
        before, after = getattr(previous, name), getattr(current, name)
        if before is not None and after is not None and after < before:
            raise ValueError("Cumulative counters decreased without a new proven epoch")
        differences[name] = after - before if before is not None and after is not None else None
    cost = None
    if current.reported_cost_usd is not None and previous.reported_cost_usd is not None:
        cost = current.reported_cost_usd - previous.reported_cost_usd
        if cost < 0:
            raise ValueError("Cumulative reported cost decreased")
    known = [value for value in differences.values() if value is not None]
    if known and not any(known) and not cost:
        return None
    same_model = (current.provider, current.model) == (previous.provider, previous.model)
    same_project = current.project_path == previous.project_path
    same_billing = current.billing_mode == previous.billing_mode
    return replace(
        current,
        input_tokens=differences["input_tokens"],
        output_tokens=differences["output_tokens"],
        cache_read_tokens=differences["cache_read_tokens"],
        cache_write_tokens=differences["cache_write_tokens"],
        reasoning_tokens=differences["reasoning_tokens"],
        total_tokens=differences["total_tokens"],
        request_count=differences["request_count"],
        reported_cost_usd=cost,
        kind="delta",
        interval_start=previous.occurred_at,
        occurred_at=current.occurred_at if previous.occurred_at is not None else None,
        model=current.model if same_model else None,
        project_path=current.project_path if same_project else None,
        billing_mode=current.billing_mode if same_billing else "unknown",
    )
