import math
from collections.abc import Mapping
from typing import Any

from mandri.core.types.usage import UsageAccount
from mandri.sessions.usage.counters import count, money, record, text


def codex_account_snapshot(
    profile_id: str, payload: Mapping[str, Any], *, observed_at_ms: int
) -> UsageAccount:
    buckets = record(payload.get("rateLimitsByLimitId"))
    if not buckets and record(payload.get("rateLimits")):
        buckets = {"default": payload["rateLimits"]}
    windows: list[dict[str, object]] = []
    plan: str | None = None
    for bucket, raw in buckets.items():
        entry = record(raw)
        plan = text(entry.get("planType")) or plan
        for window in ("primary", "secondary"):
            value = record(entry.get(window))
            if not value:
                continue
            item: dict[str, object] = {"bucket_id": bucket, "window": window}
            used = _percentage(value.get("usedPercent"))
            if used is not None:
                item["used_percent"] = used
            reset = count(value.get("resetsAt"))
            if reset is not None:
                item["resets_at"] = reset * 1000
            duration = count(value.get("windowDurationMins"))
            if duration is not None:
                item["window_duration_minutes"] = duration
            windows.append(item)
    credits = record(record(payload.get("rateLimits")).get("credits"))
    return UsageAccount(
        account_id=profile_id,
        harness="codex",
        observed_at=observed_at_ms,
        status="available" if windows else "unavailable",
        plan=plan,
        windows=tuple(windows),
        credits=money(credits.get("balance")),
        verified=False,
    )


def claude_account_snapshot(
    profile_id: str, event: Mapping[str, Any], *, observed_at_ms: int
) -> UsageAccount:
    info = record(event.get("rate_limit_info"))
    value = text(info.get("status"))
    windows: list[dict[str, object]] = []
    if value in {"allowed", "allowed_warning", "rejected"}:
        window: dict[str, object] = {"status": value}
        bucket = text(info.get("rateLimitType"))
        if bucket:
            window["bucket_id"] = bucket
        reset = count(info.get("resetsAt"))
        if reset is not None:
            window["resets_at"] = reset * 1000
        windows.append(window)
    return UsageAccount(
        account_id=profile_id,
        harness="claude",
        observed_at=observed_at_ms,
        status="available" if windows else "unsupported",
        windows=tuple(windows),
    )


def agy_statusline_snapshot(
    profile_id: str, payload: Mapping[str, Any], *, observed_at_ms: int
) -> UsageAccount:
    windows: list[dict[str, object]] = []
    for bucket, raw in record(payload.get("quota")).items():
        value = record(raw)
        fraction = _percentage(value.get("remaining_fraction"), maximum=1)
        if fraction is None:
            continue
        window: dict[str, object] = {"bucket_id": bucket, "remaining_fraction": fraction}
        reset = text(value.get("reset_time"))
        if reset:
            window["reset_time"] = reset
        windows.append(window)
    return UsageAccount(
        account_id=profile_id,
        harness="agy",
        observed_at=observed_at_ms,
        status="available" if windows else "unsupported",
        windows=tuple(windows),
    )


def _percentage(value: Any, maximum: int = 100) -> float | None:
    if type(value) not in {int, float}:
        return None
    number = float(value)
    return number if math.isfinite(number) and 0 <= number <= maximum else None
