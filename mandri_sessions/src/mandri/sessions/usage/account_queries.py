from collections.abc import Mapping
from typing import Any

from mandri.core.types.usage import UsageAccount
from mandri.sessions.usage.accounts import _percentage, codex_account_snapshot
from mandri.sessions.usage.counters import record, text


def claude_usage_snapshot(
    profile_id: str, payload: Mapping[str, Any], *, observed_at_ms: int
) -> UsageAccount:
    windows: list[dict[str, object]] = []
    for bucket, raw in record(payload.get("rate_limits")).items():
        value = record(raw)
        used = _percentage(value.get("utilization"))
        if used is None:
            continue
        window: dict[str, object] = {"bucket_id": bucket, "used_percent": used}
        if bucket == "five_hour":
            window["window_duration_minutes"] = 300
        elif bucket.startswith("seven_day"):
            window["window_duration_minutes"] = 10080
        reset = text(value.get("resets_at"))
        if reset:
            window["reset_time"] = reset
        windows.append(window)
    return UsageAccount(
        account_id=profile_id,
        harness="claude",
        observed_at=observed_at_ms,
        status="available" if windows else "unavailable",
        plan=text(payload.get("subscription_type")),
        windows=tuple(windows),
    )


def agy_usage_snapshot(
    profile_id: str, payload: Mapping[str, Any], *, observed_at_ms: int
) -> UsageAccount:
    command = record(payload.get("command"))
    data = record(command.get("data"))
    plan = (
        text(data.get("plan_name"))
        or text(data.get("tier_display_name"))
        or text(data.get("plan_tier"))
    )
    windows: list[dict[str, object]] = []
    groups = data.get("groups")
    for group in groups if isinstance(groups, list) else []:
        group = record(group)
        buckets = group.get("buckets")
        for raw in buckets if isinstance(buckets, list) else []:
            value = record(raw)
            remaining = _percentage(value.get("remaining_fraction"), maximum=1)
            if remaining is None:
                continue
            window: dict[str, object] = {"remaining_fraction": remaining}
            label = text(group.get("label")) or text(group.get("name"))
            if label:
                window["label"] = label
            bucket = text(value.get("id"))
            if bucket:
                window["bucket_id"] = bucket
            duration = {"5h": 300, "weekly": 10080}.get(str(value.get("window")))
            if duration:
                window["window_duration_minutes"] = duration
            for key in ("window", "reset_time"):
                if text(value.get(key)):
                    window[key] = value[key]
            windows.append(window)
    return UsageAccount(
        account_id=profile_id,
        harness="agy",
        observed_at=observed_at_ms,
        status="available" if windows else "unavailable",
        plan=plan,
        windows=tuple(windows),
    )


def pi_usage_snapshots(
    profile_id: str, payload: Mapping[str, Any], *, observed_at_ms: int
) -> list[UsageAccount]:
    accounts = []
    rows = payload.get("accounts")
    for raw in rows if isinstance(rows, list) else []:
        row = record(raw)
        provider = text(row.get("provider"))
        data = record(row.get("data"))
        if not data:
            continue
        if provider == "anthropic":
            snapshot = claude_usage_snapshot(
                profile_id, {"rate_limits": data}, observed_at_ms=observed_at_ms
            )
        elif provider == "openai-codex":
            limits = record(data.get("rate_limit"))
            converted: dict[str, Any] = {"planType": data.get("plan_type")}
            for source, target in (
                ("primary_window", "primary"),
                ("secondary_window", "secondary"),
            ):
                value = record(limits.get(source))
                if not value:
                    continue
                duration = value.get("limit_window_seconds")
                converted[target] = {
                    "usedPercent": value.get("used_percent"),
                    "resetsAt": value.get("reset_at"),
                    "windowDurationMins": duration // 60 if type(duration) is int else None,
                }
            snapshot = codex_account_snapshot(
                profile_id, {"rateLimits": converted}, observed_at_ms=observed_at_ms
            )
        else:
            continue
        accounts.append(
            UsageAccount(
                account_id=f"{profile_id}:{provider}",
                harness="pi",
                observed_at=observed_at_ms,
                status=snapshot.status,
                plan=f"{provider} · {snapshot.plan}" if snapshot.plan else provider,
                windows=snapshot.windows,
            )
        )
    return accounts
