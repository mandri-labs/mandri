import json
from decimal import Decimal, InvalidOperation
from typing import Any

from mandri.gateway.usage import UsageCollector
from mandri.gateway.usage_output import has_output

_COUNTERS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "audio_tokens",
        "text_tokens",
        "image_tokens",
        "accepted_prediction_tokens",
        "rejected_prediction_tokens",
        "ephemeral_5m_input_tokens",
        "ephemeral_1h_input_tokens",
        "promptTokenCount",
        "candidatesTokenCount",
        "totalTokenCount",
        "cachedContentTokenCount",
        "thoughtsTokenCount",
        "toolUsePromptTokenCount",
        "prompt_eval_count",
        "eval_count",
    }
)
_DETAILS = frozenset(
    {
        "prompt_tokens_details",
        "completion_tokens_details",
        "input_tokens_details",
        "output_tokens_details",
        "cache_creation",
    }
)


def sum_usage(*values: dict[str, Any]) -> dict[str, Any]:
    total: dict[str, Any] = {}
    for value in values:
        for key, count in value.items():
            if type(count) is int and count >= 0:
                previous = total.get(key, 0)
                total[key] = (previous if type(previous) is int else 0) + count
            elif isinstance(count, dict):
                nested = total.get(key, {})
                total[key] = sum_usage(nested if isinstance(nested, dict) else {}, count)
    return total


def _count(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= 2**63 - 1 else None


def _amount(value: Any) -> Decimal | None:
    if not isinstance(value, (str, int, float, Decimal)) or isinstance(value, bool):
        return None
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return None
    return amount if amount.is_finite() and amount >= 0 else None


def _allowed(usage: dict[str, Any], *, nested: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in usage.items():
        if key in _COUNTERS and _count(value) is not None:
            result[key] = value
        elif not nested and key in _DETAILS and isinstance(value, dict):
            details = _allowed(value, nested=True)
            if details:
                result[key] = details
        elif key == "cost" and (amount := _amount(value)) is not None:
            result[key] = str(amount)
    return result


def _merge(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict):
            existing = target.setdefault(key, {})
            if isinstance(existing, dict):
                _merge(existing, value)
        else:
            target[key] = value


async def observe_payload(collector: UsageCollector, payload: Any) -> None:
    if isinstance(payload, list):
        for item in payload:
            await observe_payload(collector, item)
        return
    if not isinstance(payload, dict):
        return
    if payload.get("error") is not None or payload.get("type") in {"response.failed", "error"}:
        collector.upstream_failed = True
    if payload.get("type") == "response.incomplete":
        collector.observation_incomplete = True
    envelope = payload
    for key in ("response", "message"):
        if isinstance(payload.get(key), dict):
            envelope = payload[key]
            break
    changes: dict[str, Any] = {}
    if not collector.record.output_observed and has_output(payload):
        changes["output_observed"] = True
    tier = envelope.get("service_tier")
    if isinstance(tier, str) and tier in {
        "auto",
        "default",
        "standard",
        "flex",
        "priority",
        "scale",
    }:
        changes["service_tier"] = tier
    for key, target in (
        ("id", "upstream_request_id"),
        ("responseId", "upstream_request_id"),
        ("model", "observed_model"),
        ("modelVersion", "observed_model"),
        ("provider", "observed_provider"),
    ):
        value = envelope.get(key)
        if isinstance(value, str) and value.strip() and len(value) <= 512:
            changes[target] = value
    usage = envelope.get("usage")
    protocol = "openai"
    if "usageMetadata" in envelope:
        usage = envelope["usageMetadata"]
        protocol = "gemini"
    elif "prompt_eval_count" in envelope or "eval_count" in envelope:
        usage = envelope
        protocol = "ollama"
    elif payload.get("type") in {"message_start", "message_delta", "message"}:
        protocol = "anthropic"
    elif isinstance(usage, dict) and "input_tokens" in usage and "choices" not in envelope:
        protocol = (
            "responses"
            if "response" in payload
            or envelope.get("object") == "response"
            or collector.record.upstream_protocol == "responses"
            or ("output" in envelope and "status" in envelope)
            else "anthropic"
        )
    if isinstance(usage, dict):
        allowed = _allowed(usage)
        _merge(collector.raw_usage, allowed)
        if allowed:
            changes["usage_protocol"] = protocol
            changes["raw_usage_json"] = json.dumps(collector.raw_usage, sort_keys=True)
        if any(
            isinstance(value, dict)
            and any(
                _count(value.get(key)) not in (None, 0) for key in ("audio_tokens", "image_tokens")
            )
            for value in allowed.values()
        ):
            changes["modality"] = "multimodal"
        mapping = {
            "prompt_tokens": "input_tokens",
            "input_tokens": "input_tokens",
            "completion_tokens": "output_tokens",
            "output_tokens": "output_tokens",
            "total_tokens": "total_tokens",
            "cache_read_input_tokens": "cache_read_tokens",
            "cache_creation_input_tokens": "cache_write_tokens",
            "promptTokenCount": "input_tokens",
            "totalTokenCount": "total_tokens",
            "cachedContentTokenCount": "cache_read_tokens",
            "thoughtsTokenCount": "reasoning_tokens",
            "prompt_eval_count": "input_tokens",
            "eval_count": "output_tokens",
        }
        for key, target in mapping.items():
            if (count := _count(usage.get(key))) is not None:
                changes[target] = count
        for key, field, target in (
            ("prompt_tokens_details", "cached_tokens", "cache_read_tokens"),
            ("input_tokens_details", "cached_tokens", "cache_read_tokens"),
            ("completion_tokens_details", "reasoning_tokens", "reasoning_tokens"),
            ("output_tokens_details", "reasoning_tokens", "reasoning_tokens"),
        ):
            details = usage.get(key)
            if isinstance(details, dict) and (count := _count(details.get(field))) is not None:
                changes[target] = count
        candidates = _count(collector.raw_usage.get("candidatesTokenCount"))
        thoughts = _count(collector.raw_usage.get("thoughtsTokenCount"))
        prompt = _count(collector.raw_usage.get("promptTokenCount"))
        total = _count(collector.raw_usage.get("totalTokenCount"))
        if (
            protocol == "gemini"
            and thoughts is None
            and candidates is not None
            and prompt is not None
            and total is not None
            and total >= prompt + candidates
        ):
            thoughts = total - prompt - candidates
            changes["reasoning_tokens"] = thoughts
        if protocol == "gemini" and candidates is not None and thoughts is not None:
            changes["output_tokens"] = candidates + thoughts
        if (amount := _amount(usage.get("cost"))) is not None:
            changes["provider_cost"] = amount
            changes["provider_cost_currency"] = (
                "USD" if collector.record.provider_kind == "openrouter" else None
            )
    for key in tuple(changes):
        if key.endswith("_tokens"):
            previous = getattr(collector.record, key)
            if previous is not None and changes[key] < previous:
                changes.pop(key)
                collector.observation_incomplete = True
    changes = {
        key: value for key, value in changes.items() if getattr(collector.record, key) != value
    }
    if changes:
        await collector.publish(**changes)
