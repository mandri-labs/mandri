from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any


def record(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def text(value: Any) -> str | None:
    return value if isinstance(value, str) and 0 < len(value) <= 512 else None


def count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def money(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        return None
    return result if result.is_finite() and result >= 0 else None


def counters(raw: Mapping[str, Any], names: Mapping[str, str]) -> dict[str, int | None]:
    return {target: count(raw.get(source)) for target, source in names.items()}


def disjoint_total(values: Mapping[str, int | None], *names: str) -> int | None:
    components = [values.get(name) for name in names]
    if any(value is None for value in components):
        return None
    return sum(value for value in components if value is not None)


def subtract(
    values: dict[str, int | None], total: str, subset: str, target: str
) -> tuple[str, ...]:
    left, right = values.get(total), values.get(subset)
    values[target] = None
    if left is None or right is None:
        return ()
    if right > left:
        return (f"{subset}_exceeds_{total}",)
    values[target] = left - right
    return ()
