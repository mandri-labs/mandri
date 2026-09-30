from dataclasses import replace
from decimal import Decimal

import httpx
import pytest
from mandri.core.ids import ModelRef, ProviderKind, RouteId, SecretRef
from mandri.core.usage_pricing import value_usage
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.types.model import Model
from mandri.gateway.usage import UsageCollector, usage_scope
from mandri.gateway.usage_context import request_modality
from mandri.gateway.usage_observation import to_observation
from mandri.gateway.usage_payload import observe_payload
from mandri.gateway.usage_transport import observe_response
from mandri.providers.service import Provider, ProviderState, provider_model_ref

from mandri_core.tests.usage_fixtures import REVIEWED_AT, synthetic_prices


def collector(kind=ProviderKind.OPENAI):
    async def sink(record):
        pass

    provider = Provider("configured-instance", kind, None, SecretRef(""), ProviderState.VERIFIED)
    route = ResolvedRoute(
        RouteId("route"),
        provider,
        Model(kind, provider_model_ref(kind, "routing-alias"), None, SecretRef("")),
    )
    return UsageCollector(route, "chat", sink, modality="text")


@pytest.mark.parametrize("kind", list(ProviderKind))
def test_route_identity_and_classification_survive_missing_response_model(kind):
    item = to_observation(collector(kind).record)
    assert item.model == "routing-alias"
    assert item.observed_model is None
    assert item.provider == "configured-instance"
    assert item.pricing_context["provider_kind"] == kind.value
    assert item.pricing_context["model_basis"] == "route_selection"
    expected = (
        "subscription"
        if kind == ProviderKind.OPENCODE_GO
        else "local"
        if kind in {ProviderKind.OLLAMA, ProviderKind.LM_STUDIO}
        else "api"
    )
    assert item.billing_mode == expected


async def test_route_model_wins_over_wire_and_response_models():
    usage = collector()
    request = httpx.Request(
        "POST",
        "https://fixture.invalid/v1/responses",
        json={"model": "gpt-4.1-mini", "input": "private-content"},
    )
    response = httpx.Response(
        200,
        request=request,
        json={
            "model": "gpt-4.1-mini-2025-04-14",
            "provider": "upstream-provider",
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 200,
                "input_tokens_details": {"cached_tokens": 600},
            },
            "service_tier": "default",
        },
    )
    usage.record = replace(
        usage.record,
        selected_model="openai/gpt-4.1-mini",
        started_at=REVIEWED_AT,
        service_tier="auto",
    )
    with usage_scope(usage):
        await observe_response(response)
    item = to_observation(usage.record)
    assert item.model == "gpt-4.1-mini"
    assert item.observed_model == "gpt-4.1-mini-2025-04-14"
    assert item.pricing_context["selected_model"] == "openai/gpt-4.1-mini"
    assert item.pricing_context["requested_model"] == "gpt-4.1-mini"
    assert item.pricing_context["observed_provider"] == "upstream-provider"
    assert item.pricing_context["model_basis"] == "route_selection"
    assert item.pricing_context["usage_protocol"] == "responses"
    assert item.pricing_context["service_tier"] == "standard"
    assert "private-content" not in str(item)
    assert value_usage(item, synthetic_prices())[0] == Decimal("0.00054")


async def test_wire_model_does_not_override_route_selection():
    usage = collector()
    request = httpx.Request(
        "POST", "https://fixture.invalid/v1/chat/completions", json={"model": "wire-model"}
    )
    response = httpx.Response(
        200,
        request=request,
        json={"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
    )
    with usage_scope(usage):
        await observe_response(response)
    item = to_observation(usage.record)
    assert item.model == "routing-alias" and item.observed_model is None
    assert item.pricing_context["model_basis"] == "route_selection"


async def test_unknown_response_model_preserves_selected_model_pricing():
    usage = collector()
    usage.record = replace(
        usage.record, selected_model=ModelRef("openai/gpt-4.1-mini"), started_at=REVIEWED_AT
    )
    await observe_payload(
        usage,
        {
            "model": "unknown-resolved-model",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "prompt_tokens_details": {"cached_tokens": 600},
            },
        },
    )
    assert value_usage(to_observation(usage.record), synthetic_prices())[0] == Decimal("0.00054")


async def test_malformed_identity_metadata_does_not_break_capture():
    usage = collector()
    await observe_payload(
        usage,
        {
            "model": "  ",
            "provider": {"secret": True},
            "service_tier": {},
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        },
    )
    assert to_observation(usage.record).model == "routing-alias"
    assert usage.record.input_tokens == 12


async def test_unknown_currency_cost_does_not_claim_reported_usd():
    usage = collector(ProviderKind.CUSTOM)
    await observe_payload(usage, {"usage": {"cost": "0.12"}})
    item = to_observation(usage.record)
    assert item.reported_cost_usd is None and item.reported_cost_basis is None
    assert item.pricing_context["raw_usage"]["cost"] == "0.12"


def test_gemini_text_requests_classified_before_response():
    assert request_modality("gemini", {"contents": [{"parts": [{"text": "fixture"}]}]}) == "text"
    assert (
        request_modality("gemini", {"contents": [{"parts": [{"inlineData": {}}]}]}) == "multimodal"
    )
    assert request_modality("gemini", {"contents": [{"parts": [{"unknown": {}}]}]}) is None


@pytest.mark.parametrize("total,output,reasoning", [(36, 6, 0), (40, 10, 4), (None, None, None)])
async def test_gemini_total_reconciles_missing_thoughts_without_assuming_zero(
    total, output, reasoning
):
    usage = collector(ProviderKind.GEMINI)
    await observe_payload(
        usage,
        {
            "usageMetadata": {
                "promptTokenCount": 30,
                "candidatesTokenCount": 6,
                "totalTokenCount": total,
            }
        },
    )
    assert usage.record.output_tokens == output
    assert usage.record.reasoning_tokens == reasoning
    item = to_observation(usage.record)
    assert "thoughtsTokenCount" not in item.pricing_context["raw_usage"]
    if reasoning is not None:
        assert (
            item.pricing_context["reasoning_counter_basis"] == "total_minus_prompt_minus_candidates"
        )
