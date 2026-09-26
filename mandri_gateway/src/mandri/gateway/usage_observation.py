import json

from mandri.core.ids import ProviderKind
from mandri.core.types.usage import UsageObservation
from mandri.core.usage_normalization import normalize_usage
from mandri.gateway.usage import GatewayUsageRecord
from mandri.providers.service import split_model_ref


def to_observation(record: GatewayUsageRecord) -> UsageObservation:
    raw_usage = json.loads(record.raw_usage_json)
    known_protocol = record.usage_protocol in {"openai", "responses", "anthropic", "gemini"}
    context_tokens = record.input_tokens
    if record.usage_protocol == "anthropic":
        context_tokens = (
            record.input_tokens + record.cache_read_tokens + record.cache_write_tokens
            if record.input_tokens is not None
            and record.cache_read_tokens is not None
            and record.cache_write_tokens is not None
            else None
        )
    observation = UsageObservation(
        source="gateway",
        source_key=record.request_id,
        fact_key=f"gateway:{record.request_id}",
        session_id=record.session_id,
        root_session_id=record.root_session_id,
        project_path=record.project_path,
        harness=record.harness,
        billing_mode=record.billing_mode,
        provider=record.provider_name,
        model=split_model_ref(ProviderKind(record.provider_kind), record.selected_model),
        observed_model=record.observed_model,
        occurred_at=record.started_at,
        observed_at=record.updated_at,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        total_tokens=record.total_tokens,
        cache_read_tokens=record.cache_read_tokens,
        cache_write_tokens=record.cache_write_tokens,
        reasoning_tokens=record.reasoning_tokens,
        request_count=1,
        sequence=record.revision,
        complete=not record.incomplete,
        input_includes_cache=(record.usage_protocol != "anthropic") if known_protocol else None,
        output_includes_reasoning=True if known_protocol else None,
        reported_cost_usd=(
            record.provider_cost if record.provider_cost_currency == "USD" else None
        ),
        reported_cost_basis=(
            "provider_reported" if record.provider_cost_currency == "USD" else None
        ),
        pricing_context={
            "comparison_basis": "standard_text_api",
            "provider_kind": record.provider_kind,
            **({"modality": record.modality} if record.modality else {}),
            **(
                {
                    "service_tier": "standard"
                    if record.service_tier == "default"
                    else record.service_tier
                }
                if record.service_tier
                else {}
            ),
            **({"context_tokens": context_tokens} if context_tokens is not None else {}),
            "selected_model": record.selected_model,
            "requested_model": record.requested_model,
            "observed_model": record.observed_model,
            "observed_provider": record.observed_provider,
            "model_basis": "route_selection",
            "billing_mode_basis": "provider_configuration",
            "usage_protocol": record.usage_protocol,
            "upstream_request_id": record.upstream_request_id,
            "route_id": record.route_id,
            "status": record.status,
            "output_observed": record.output_observed,
            "transport_attempts": record.transport_attempts,
            "raw_usage": raw_usage,
            "reasoning_counter_basis": (
                "total_minus_prompt_minus_candidates"
                if record.usage_protocol == "gemini"
                and record.reasoning_tokens is not None
                and "thoughtsTokenCount" not in raw_usage
                else "provider_reported"
                if record.reasoning_tokens is not None
                else "unknown"
            ),
        },
    )
    return normalize_usage(observation)
