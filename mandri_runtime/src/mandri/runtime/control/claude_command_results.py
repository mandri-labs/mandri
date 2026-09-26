import math
from typing import Any


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _fields(data: dict[str, Any], labels: dict[str, str], prefix: str = "") -> list[dict[str, Any]]:
    return [
        {"label": f"{prefix}{label}", "value": data[key]}
        for key, label in labels.items()
        if _number(data.get(key))
    ]


def _context(data: dict[str, Any]) -> dict[str, Any] | None:
    if not all(_number(data.get(key)) for key in ("total_tokens", "raw_max_tokens", "percentage")):
        return None
    fields = _fields(
        data,
        {
            "total_tokens": "Tokens in use",
            "raw_max_tokens": "Compaction window (tokens)",
            "percentage": "Window used (%)",
        },
    )
    if isinstance(data.get("model"), str):
        fields.insert(0, {"label": "Model", "value": data["model"]})
    over = data.get("over_limit")
    if isinstance(over, dict) and _number(over.get("tokens_over")):
        labels = {
            "hard_limit": "Tokens over model limit",
            "compaction_window": "Tokens over compaction window",
        }
        if over.get("kind") in labels:
            fields.append({"label": labels[over["kind"]], "value": over["tokens_over"]})
    kinds = {
        "used": "In use",
        "free": "Available",
        "buffer": "Compaction reserve",
        "deferred": "Outside window",
    }
    categories = data.get("categories", [])
    for row in categories if isinstance(categories, list) else []:
        if (
            isinstance(row, dict)
            and isinstance(row.get("name"), str)
            and row.get("kind") in kinds
            and _number(row.get("tokens"))
        ):
            fields.append(
                {"label": f"{row['name']} · {kinds[row['kind']]} (tokens)", "value": row["tokens"]}
            )
    for key, title, name_key, detail_key in (
        ("mcp_tools", "MCP tool", "name", "server_name"),
        ("memory_files", "Memory", "path", "type"),
        ("agents", "Agent", "agent_type", "source"),
        ("skills", "Skill", "name", "source"),
    ):
        rows = data.get(key, [])
        for row in rows if isinstance(rows, list) else []:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get(name_key), str)
                or not _number(row.get("tokens"))
            ):
                continue
            detail = row.get(detail_key)
            label = f"{title} · {row[name_key]}"
            if isinstance(detail, str):
                label += f" · {detail}"
            if key == "skills" and isinstance(row.get("plugin_name"), str):
                label += f" · {row['plugin_name']}"
            fields.append({"label": f"{label} (tokens)", "value": row["tokens"]})
    return {"kind": "fields", "title": "Context usage", "fields": fields}


def _usage(data: dict[str, Any]) -> dict[str, Any] | None:
    session = data.get("session")
    if not isinstance(session, dict):
        return None
    fields = _fields(
        session,
        {
            "total_cost_usd": "Session cost (USD)",
            "total_api_duration_ms": "API duration (ms)",
            "total_duration_ms": "Session duration (ms)",
            "total_lines_added": "Lines added",
            "total_lines_removed": "Lines removed",
        },
    )
    if not fields:
        return None
    models = session.get("model_usage")
    if isinstance(models, dict):
        for name, usage in models.items():
            if not isinstance(usage, dict):
                continue
            fields.extend(
                _fields(
                    usage,
                    {
                        "inputTokens": "Input tokens",
                        "outputTokens": "Output tokens",
                        "thinkingTokens": "Thinking tokens (included in output)",
                        "cacheReadInputTokens": "Cache read tokens",
                        "cacheCreationInputTokens": "Cache write tokens",
                        "webSearchRequests": "Web searches",
                        "costUSD": "Cost (USD)",
                        "contextWindow": "Context window (tokens)",
                        "maxOutputTokens": "Maximum output tokens",
                    },
                    f"{name} · ",
                )
            )
    rate_limits = data.get("rate_limits")
    if not isinstance(rate_limits, dict):
        fields.append({"label": "Plan usage", "value": "Unavailable for this session"})
        return {"kind": "fields", "title": "Usage", "fields": fields}
    limits = rate_limits.get("limits")
    if limits is None:
        fields.append({"label": "Plan usage meters", "value": "Not supplied by the native server"})
    elif limits == []:
        fields.append({"label": "Plan usage meters", "value": "No meters reported"})
    elif isinstance(limits, list):
        for row in limits:
            if not isinstance(row, dict) or not _number(row.get("percent")):
                continue
            kind, group = row.get("kind"), row.get("group")
            if not isinstance(kind, str) or not isinstance(group, str):
                continue
            label = f"{group} · {kind}"
            scope = row.get("scope")
            if isinstance(scope, dict):
                for key in ("model", "surface"):
                    detail = scope.get(key)
                    if isinstance(detail, dict) and isinstance(detail.get("display_name"), str):
                        label += f" · {detail['display_name']}"
            fields.append({"label": f"{label} · Used (%)", "value": row["percent"]})
            if isinstance(row.get("resets_at"), str):
                fields.append({"label": f"{label} · Resets", "value": row["resets_at"]})
            if isinstance(row.get("severity"), str):
                fields.append({"label": f"{label} · Status", "value": row["severity"]})
            if isinstance(row.get("is_active"), bool):
                fields.append({"label": f"{label} · Current meter", "value": row["is_active"]})
    extra = rate_limits.get("extra_usage")
    if isinstance(extra, dict):
        if isinstance(extra.get("is_enabled"), bool):
            fields.append({"label": "Extra usage enabled", "value": extra["is_enabled"]})
        currency = extra.get("currency")
        unit = f"{currency} minor units" if isinstance(currency, str) else "minor currency units"
        fields.extend(
            _fields(
                extra,
                {
                    "monthly_limit": f"Extra usage limit ({unit})",
                    "used_credits": f"Extra usage spent ({unit})",
                    "utilization": "Extra usage used (%)",
                },
            )
        )
    return {"kind": "fields", "title": "Usage", "fields": fields}


def command_result(frame: dict[str, Any]) -> dict[str, Any] | None:
    context = frame.get("context_usage")
    if isinstance(context, dict):
        result = _context(context)
        if result is not None:
            return result
    usage = frame.get("usage_report")
    return _usage(usage) if isinstance(usage, dict) else None
