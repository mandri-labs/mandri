import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from typing import Any

from mandri.sessions.usage.counters import counters, disjoint_total, money, record, subtract, text
from mandri.sessions.usage.types import NativeUsageContext, NativeUsageObservation

_CODEX = {
    "input_tokens": "inputTokens",
    "cache_read_tokens": "cachedInputTokens",
    "cache_write_tokens": "cacheWriteInputTokens",
    "output_tokens": "outputTokens",
    "reasoning_tokens": "reasoningOutputTokens",
    "total_tokens": "totalTokens",
}
_CLAUDE = {
    "input_tokens": "input_tokens",
    "cache_read_tokens": "cache_read_input_tokens",
    "cache_write_tokens": "cache_creation_input_tokens",
    "output_tokens": "output_tokens",
}
_CLAUDE_MODEL = {
    "input_tokens": "inputTokens",
    "cache_read_tokens": "cacheReadInputTokens",
    "cache_write_tokens": "cacheCreationInputTokens",
    "output_tokens": "outputTokens",
}
_AGY = {
    "input_tokens": "input_tokens",
    "cache_read_tokens": "cache_read_tokens",
    "cache_write_tokens": "cache_write_tokens",
    "output_tokens": "output_tokens",
    "total_tokens": "total_tokens",
    "reasoning_tokens": "thinking_tokens",
}
_PI = {
    "input_tokens": "input",
    "cache_read_tokens": "cacheRead",
    "cache_write_tokens": "cacheWrite",
    "output_tokens": "output",
    "total_tokens": "totalTokens",
}


def normalize_native_usage(
    context: NativeUsageContext,
    event: Mapping[str, Any],
    *,
    observed_at_ms: int,
    origin: str = "live",
) -> tuple[NativeUsageObservation, ...]:
    observations: list[NativeUsageObservation] = []
    if event.get("parent_tool_use_id") is not None:
        context = replace(context, parent_session_id="unattributed_parent")
    if context.harness == "codex":
        params = record(event.get("params"))
        usage = record(record(params.get("tokenUsage")).get("total"))
        if event.get("method") == "thread/tokenUsage/updated" and usage:
            values = counters(usage, _CODEX)
            anomalies = subtract(
                values, "input_tokens", "cache_read_tokens", "uncached_input_tokens"
            )
            anomalies += subtract(
                values, "output_tokens", "reasoning_tokens", "visible_output_tokens"
            )
            observations.append(
                _observation(
                    context,
                    values,
                    observed_at_ms,
                    "thread",
                    text(params.get("threadId")),
                    turn_id=text(params.get("turnId")),
                    anomalies=anomalies,
                    observed_model=text(params.get("model")) or context.observed_model,
                    model=context.observed_model if context.model_scope_proven else None,
                )
            )
        payload = record(event.get("payload"))
        if event.get("type") == "event_msg" and payload.get("type") == "token_count":
            usage = record(record(payload.get("info")).get("total_token_usage"))
            if usage:
                names = {key: key for key in _CODEX}
                names["cache_read_tokens"] = "cached_input_tokens"
                names["reasoning_tokens"] = "reasoning_output_tokens"
                values = counters(usage, names)
                anomalies = subtract(
                    values, "input_tokens", "cache_read_tokens", "uncached_input_tokens"
                )
                anomalies += subtract(
                    values, "output_tokens", "reasoning_tokens", "visible_output_tokens"
                )
                observations.append(
                    _observation(
                        context,
                        values,
                        observed_at_ms,
                        "thread",
                        context.native_id,
                        anomalies=anomalies,
                    )
                )
    elif context.harness == "claude":
        if event.get("type") == "result":
            if origin == "history":
                return ()
            models = record(event.get("modelUsage") or event.get("model_usage"))
            for model, raw in models.items():
                usage = record(raw)
                if usage:
                    values = counters(usage, _CLAUDE_MODEL)
                    values["uncached_input_tokens"] = values["input_tokens"]
                    observations.append(
                        _observation(
                            context,
                            values,
                            observed_at_ms,
                            "call_model",
                            context.native_id,
                            model=text(model),
                            includes_descendants=True,
                            reported_cost_usd=money(usage.get("costUSD")),
                        )
                    )
            if record(event.get("usage")):
                values = counters(record(event.get("usage")), _CLAUDE)
                values["uncached_input_tokens"] = values["input_tokens"]
                observations.append(
                    _observation(
                        context,
                        values,
                        observed_at_ms,
                        "turn_main_loop",
                        context.native_id,
                        turn_id=text(event.get("uuid")),
                        cumulative=False,
                        includes_descendants=False,
                    )
                )
        elif event.get("type") == "assistant":
            message = record(event.get("message"))
            identity = text(message.get("id"))
            if identity and record(message.get("usage")):
                values = counters(record(message.get("usage")), _CLAUDE)
                values["uncached_input_tokens"] = values["input_tokens"]
                values["request_count"] = 1
                observations.append(
                    _observation(
                        context,
                        values,
                        observed_at_ms,
                        "message" if origin == "history" else "message_detail",
                        text(event.get("sessionId")) or context.native_id,
                        turn_id=identity,
                        upstream_request_id=identity,
                        model=text(message.get("model")),
                        cumulative=False,
                        includes_descendants=False,
                    )
                )
    elif context.harness == "pi":
        message = record(event.get("message"))
        usage = record(message.get("usage"))
        timestamp = message.get("timestamp")
        if (
            event.get("type") in {"message", "message_end"}
            and message.get("role") == "assistant"
            and usage
            and isinstance(timestamp, (int, float))
            and not isinstance(timestamp, bool)
            and math.isfinite(timestamp)
            and timestamp >= 0
        ):
            values = counters(usage, _PI)
            values["uncached_input_tokens"] = values["input_tokens"]
            values["reported_total_tokens"] = values["total_tokens"]
            values["request_count"] = 1
            provider = text(message.get("provider"))
            pi_model = text(message.get("model"))
            identity = json.dumps([timestamp, provider, pi_model])
            observations.append(
                _observation(
                    replace(context, routing="gateway") if provider == "mandri" else context,
                    values,
                    observed_at_ms,
                    "message",
                    context.native_id,
                    turn_id=identity,
                    model=pi_model,
                    observed_provider=provider,
                    provider_kind=provider,
                    cumulative=False,
                    includes_descendants=False,
                    reported_cost_usd=money(record(usage.get("cost")).get("total")),
                )
            )
        auxiliary = _pi_auxiliary(context, event, observed_at_ms, origin)
        if auxiliary is not None:
            observations.append(auxiliary)
    elif context.harness == "agy" and event.get("event") == "step_update":
        step = record(event.get("step_update"))
        usage = record(step.get("usage"))
        if usage:
            values = counters(usage, _AGY)
            if "cache_write_tokens" not in usage:
                values["cache_write_tokens"] = 0
            values["uncached_input_tokens"] = values["input_tokens"]
            observations.append(
                _observation(
                    context,
                    values,
                    observed_at_ms,
                    "step_detail",
                    context.native_id,
                    turn_id=str(step.get("step_index")),
                    cumulative=False,
                    model=text(step.get("model")),
                )
            )
    elif context.harness == "agy" and event.get("event") == "result":
        result = record(event.get("result"))
        usage = record(result.get("usage"))
        if usage:
            values = counters(usage, _AGY)
            if "cache_write_tokens" not in usage:
                values["cache_write_tokens"] = 0
            values["uncached_input_tokens"] = values["input_tokens"]
            observations.append(
                _observation(
                    context,
                    values,
                    observed_at_ms,
                    "process_session",
                    text(result.get("conversation_id")) or context.native_id,
                    observed_model=text(result.get("model")) or context.observed_model,
                    model=context.observed_model if context.model_scope_proven else None,
                )
            )
    if context.harness in {"claude", "agy", "pi"}:
        observations = [_canonical_total(item) for item in observations]
    return tuple(
        replace(item, origin=origin, occurred_at_ms=_timestamp(event)) for item in observations
    )


def _pi_auxiliary(
    context: NativeUsageContext, event: Mapping[str, Any], now: int, origin: str
) -> NativeUsageObservation | None:
    kind = event.get("type")
    if kind in {"message", "message_end"}:
        source = record(event.get("message"))
        if source.get("role") != "toolResult" or _timestamp(event) is None:
            return None
        identity = json.dumps(["toolResult", source.get("timestamp"), source.get("toolCallId")])
        context = replace(context, parent_session_id="unattributed_tool_usage")
    elif origin == "history" and kind in {"usage", "compaction", "branch_summary"}:
        source = dict(event)
        entry_id = text(event.get("id"))
        if entry_id is None:
            return None
        identity = json.dumps([kind, entry_id])
    else:
        return None
    usage = record(source.get("usage"))
    if not usage:
        return None
    values = counters(usage, _PI)
    values["uncached_input_tokens"] = values["input_tokens"]
    values["reported_total_tokens"] = values["total_tokens"]
    provider = text(source.get("provider"))
    return _observation(
        replace(context, routing="gateway") if provider == "mandri" else context,
        values,
        now,
        "message",
        context.native_id,
        turn_id=identity,
        model=text(source.get("model")),
        observed_provider=provider,
        provider_kind=provider,
        cumulative=False,
        includes_descendants=None,
        reported_cost_usd=money(record(usage.get("cost")).get("total")),
    )


def _canonical_total(item: NativeUsageObservation) -> NativeUsageObservation:
    values = dict(item.counters)
    if item.context.harness == "agy":
        values["reported_total_tokens"] = values.get("total_tokens")
        values["total_tokens"] = disjoint_total(
            values,
            "input_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "output_tokens",
        )
    else:
        values["total_tokens"] = disjoint_total(
            values,
            "input_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "output_tokens",
        )
    return replace(item, counters=values)


def _observation(
    context: NativeUsageContext,
    values: Mapping[str, int | None],
    now: int,
    scope: str,
    native_id: str | None,
    **fields: Any,
) -> NativeUsageObservation:
    epoch = (
        native_id or context.session_id if scope in {"thread", "message"} else context.process_epoch
    )
    model = None if scope == "message" else fields.get("model")
    series = json.dumps([context.harness, context.session_id, epoch, scope, model])
    identity = fields.get("turn_id") if not fields.get("cumulative", True) else None
    safe = json.dumps(
        [series, identity] if identity else [series, values, str(fields.get("reported_cost_usd"))],
        sort_keys=True,
    )
    key = hashlib.sha256(safe.encode()).hexdigest()
    return NativeUsageObservation(
        source_key=key,
        series_key=series,
        context=context,
        scope=scope,
        counters=values,
        observed_at_ms=now,
        native_id=native_id,
        authoritative=context.routing == "native"
        and scope not in {"message_detail", "step_detail"},
        **fields,
    )


def _timestamp(event: Mapping[str, Any]) -> int | None:
    value = event.get("timestamp")
    message_time = record(event.get("message")).get("timestamp")
    if (
        event.get("type") in {"message", "message_end"}
        and isinstance(message_time, (int, float))
        and not isinstance(message_time, bool)
        and math.isfinite(message_time)
        and message_time >= 0
    ):
        return int(message_time)
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return int(parsed.timestamp() * 1000) if parsed.tzinfo is not None else None
    except (ValueError, OverflowError):
        return None
