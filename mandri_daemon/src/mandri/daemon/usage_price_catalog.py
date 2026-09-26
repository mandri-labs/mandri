import hashlib
import json
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from mandri.core.types.usage import UsagePrice
from mandri.core.usage_price_schedule import OPENCODE_GO_SCHEDULED_MODELS
from mandri.core.usage_pricing import validate_price

MODELS_DEV_URL = "https://models.dev/api.json"
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
PROVIDER_IDS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "gemini",
    "opencode": "opencode",
    "opencode-go": "opencode_go",
}
MAX_PRICES = 5000
MAX_TIERS = 32
MODELS_DEV_FIELDS = {
    "input": "input_tokens",
    "output": "output_tokens",
    "reasoning": "reasoning_tokens",
    "cache_read": "cache_read_tokens",
    "cache_write": "cache_write_tokens",
}
OPENROUTER_FIELDS = {
    "prompt": "input_tokens",
    "completion": "output_tokens",
    "input_cache_read": "cache_read_tokens",
    "input_cache_write": "cache_write_tokens",
}


def _supports_text(value: object) -> bool:
    return isinstance(value, list) and "text" in value


def decimal_rate(value: object, unit: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("Invalid token price")
    try:
        rate = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("Invalid token price") from error
    if not rate.is_finite() or rate < 0 or rate > 1_000_000:
        raise ValueError("Invalid token price")
    if len(rate.as_tuple().digits) > 50 or abs(int(rate.as_tuple().exponent)) > 50:
        raise ValueError("Token price precision exceeds catalog bounds")
    if unit == "per_million_tokens":
        return rate
    if unit != "per_token":
        raise ValueError("Unsupported source price unit")
    with localcontext() as context:
        context.prec = 64
        return rate * 1_000_000


def _rates(cost: dict[str, Any], fields: dict[str, str], unit: str) -> dict[str, Decimal]:
    result = {
        target: decimal_rate(cost[source], unit)
        for source, target in fields.items()
        if source in cost and cost[source] is not None
    }
    if "input_tokens" not in result or "output_tokens" not in result:
        raise ValueError("Catalog model requires input and output prices")
    result.setdefault("reasoning_tokens", result["output_tokens"])
    if all(rate == 0 for rate in result.values()):
        result.setdefault("cache_read_tokens", Decimal(0))
        result.setdefault("cache_write_tokens", Decimal(0))
    return result


def _price(
    provider: str,
    model: str,
    rates: dict[str, Decimal],
    constraints: dict[str, object],
    source: str,
    reviewed_at: int,
) -> UsagePrice:
    if not model or len(model) > 256:
        raise ValueError("Invalid catalog model identity")
    payload = json.dumps(
        [provider, model, rates, constraints, source, reviewed_at],
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    price = UsagePrice(
        price_id="current:" + hashlib.sha256(payload.encode()).hexdigest(),
        provider=provider,
        model=model,
        effective_from=0,
        valuation_basis="current_price_comparison",
        reviewed_at=reviewed_at,
        source=source,
        rates=rates,
        constraints=constraints,
    )
    validate_price(price)
    return price


def _tiered_prices(
    provider: str,
    model: str,
    cost: dict[str, Any],
    fields: dict[str, str],
    tiers: list[tuple[int, dict[str, Any]]],
    source: str,
    reviewed_at: int,
    unit: str,
) -> list[UsagePrice]:
    constraints: dict[str, object] = {
        "modality": "text",
        "service_tier": "standard",
        "region": "global",
        "modifiers": [],
    }
    if provider == "anthropic":
        constraints["cache_write_ttl_seconds"] = 300
    if provider == "gemini":
        constraints["cache_mode"] = "implicit"
    if len(tiers) > MAX_TIERS:
        raise ValueError("Catalog model exceeds tier limit")
    tiers.sort(key=lambda item: item[0])
    thresholds = [threshold for threshold, _ in tiers]
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Duplicate context tiers")
    result = []
    previous = 0
    for threshold, next_cost in [*tiers, (0, {})]:
        bounds = dict(constraints)
        if previous:
            bounds["min_context_tokens"] = previous + 1
        if threshold:
            bounds["max_context_tokens"] = threshold
        result.append(
            _price(provider, model, _rates(cost, fields, unit), bounds, source, reviewed_at)
        )
        cost = {**cost, **next_cost}
        previous = threshold
    return result


def _models_dev_tiers(cost: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    entries = cost.get("tiers")
    if entries is None:
        legacy = cost.get("context_over_200k")
        if legacy is None:
            return []
        if not isinstance(legacy, dict):
            raise ValueError("Invalid legacy context tier")
        return [(200_000, legacy)]
    if not isinstance(entries, list) or len(entries) > MAX_TIERS:
        raise ValueError("Invalid context tiers")
    result = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("tier"), dict):
            raise ValueError("Invalid context tier")
        rule = entry["tier"]
        size = rule.get("size")
        if rule.get("type") != "context" or type(size) is not int or size <= 0:
            raise ValueError("Unsupported context tier")
        result.append((size, entry))
    return result


def parse_models_dev(payload: object, reviewed_at: int) -> tuple[UsagePrice, ...]:
    if not isinstance(payload, dict):
        raise ValueError("Invalid models.dev catalog")
    result: list[UsagePrice] = []
    for source_provider, provider in PROVIDER_IDS.items():
        entry = payload.get(source_provider)
        if not isinstance(entry, dict) or not isinstance(entry.get("models"), dict):
            continue
        for model, details in entry["models"].items():
            if not isinstance(details, dict) or not isinstance(model, str):
                continue
            if details.get("id", model) != model:
                continue
            if provider == "opencode_go" and model in OPENCODE_GO_SCHEDULED_MODELS:
                continue
            modalities = details.get("modalities", {})
            if not isinstance(modalities, dict) or not _supports_text(modalities.get("output")):
                continue
            cost = details.get("cost")
            if not isinstance(cost, dict):
                continue
            try:
                prices = _tiered_prices(
                    provider,
                    model,
                    cost,
                    MODELS_DEV_FIELDS,
                    _models_dev_tiers(cost),
                    MODELS_DEV_URL,
                    reviewed_at,
                    "per_million_tokens",
                )
            except ValueError:
                continue
            if len(result) + len(prices) > MAX_PRICES:
                raise ValueError("Catalog exceeds price limit")
            result.extend(prices)
    if not result:
        raise ValueError("No supported models.dev prices")
    return tuple(result)


def _openrouter_tiers(cost: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    entries = cost.get("overrides", [])
    if not isinstance(entries, list) or len(entries) > MAX_TIERS:
        raise ValueError("Invalid OpenRouter overrides")
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid OpenRouter override")
        size = entry.get("min_prompt_tokens")
        if (
            type(size) is not int
            or size <= 0
            or set(entry) - set(OPENROUTER_FIELDS) - {"min_prompt_tokens"}
        ):
            raise ValueError("Unsupported OpenRouter override")
        result.append((size, entry))
    return result


def parse_openrouter(payload: object, reviewed_at: int) -> tuple[UsagePrice, ...]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Invalid OpenRouter catalog")
    result: list[UsagePrice] = []
    seen: set[str] = set()
    for entry in payload["data"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        model = entry["id"]
        if model in seen:
            raise ValueError("Duplicate OpenRouter model identity")
        seen.add(model)
        architecture = entry.get("architecture", {})
        if not isinstance(architecture, dict) or not _supports_text(
            architecture.get("output_modalities")
        ):
            continue
        cost = entry.get("pricing")
        if not isinstance(cost, dict):
            continue
        try:
            if "request" in cost and decimal_rate(cost["request"], "per_token") != 0:
                continue
            prices = _tiered_prices(
                "openrouter",
                model,
                cost,
                OPENROUTER_FIELDS,
                _openrouter_tiers(cost),
                OPENROUTER_URL,
                reviewed_at,
                "per_token",
            )
        except ValueError:
            continue
        if len(result) + len(prices) > MAX_PRICES:
            raise ValueError("Catalog exceeds price limit")
        result.extend(prices)
    if not result:
        raise ValueError("No supported OpenRouter prices")
    return tuple(result)
