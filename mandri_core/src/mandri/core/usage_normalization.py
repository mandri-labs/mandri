from dataclasses import replace

from mandri.core.ids import ProviderKind
from mandri.core.model_refs import MODEL_REF_PREFIXES
from mandri.core.types.usage import UsageObservation


def normalize_usage(value: UsageObservation) -> UsageObservation:
    context = value.pricing_context
    if (
        value.kind == "delta"
        and value.request_count == 1
        and context.get("status") in {"failed", "cancelled"}
        and value.output_tokens in (None, 0)
        and value.reasoning_tokens in (None, 0)
        and value.reported_cost_usd in (None, 0)
        and context.get("output_observed") is not True
    ):
        value = replace(value, authoritative=False)
    if value.source.startswith("native:"):
        if (
            value.provider == "mandri"
            or context.get("observed_provider") == "mandri"
            or context.get("routing") == "gateway"
            or context.get("model_basis") == "mandri_route_model_reference"
        ):
            return replace(value, authoritative=False)
        return value
    if value.source != "gateway":
        return value
    context = dict(context)
    model = value.model
    selected = context.get("selected_model")
    provider = context.get("provider_kind")
    if (
        isinstance(selected, str)
        and selected
        and isinstance(provider, str)
        and provider in MODEL_REF_PREFIXES
    ):
        model = selected.removeprefix(MODEL_REF_PREFIXES[ProviderKind(provider)])
        context["model_basis"] = "route_selection"
    if context.get("service_tier") in {"auto", "default"}:
        context["service_tier"] = "standard"
    cache_write = value.cache_write_tokens
    raw = context.get("raw_usage")
    if (
        cache_write is None
        and context.get("usage_protocol") in {"openai", "responses"}
        and isinstance(raw, dict)
        and raw
        and "cache_creation_input_tokens" not in raw
    ):
        cache_write = 0
    return replace(value, model=model, cache_write_tokens=cache_write, pricing_context=context)
