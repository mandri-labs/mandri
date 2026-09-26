from dataclasses import replace
from decimal import Decimal

import pytest
from mandri.core.types.usage import UsageObservation
from mandri.core.usage_normalization import normalize_usage


def gateway(**fields):
    return UsageObservation(
        **{
            "source": "gateway",
            "source_key": "request",
            "fact_key": "request",
            "model": "upstream-alias",
            "observed_model": "upstream-alias",
            "input_tokens": 100,
            "output_tokens": 10,
            "pricing_context": {
                "selected_model": "openrouter/z-ai/glm-5.3",
                "provider_kind": "openrouter",
                "usage_protocol": "responses",
                "raw_usage": {"input_tokens": 100, "output_tokens": 10},
                "service_tier": "auto",
            },
            **fields,
        }
    )


def test_route_identity_keeps_vendor_namespace_and_observed_evidence():
    value = normalize_usage(gateway())
    assert value.model == "z-ai/glm-5.3"
    assert value.observed_model == "upstream-alias"
    assert value.pricing_context["model_basis"] == "route_selection"
    assert normalize_usage(value) == value


@pytest.mark.parametrize("protocol", ["openai", "responses"])
def test_implicit_cache_write_does_not_invent_missing_cache_reads(protocol):
    original = gateway()
    value = normalize_usage(
        replace(original, pricing_context={**original.pricing_context, "usage_protocol": protocol})
    )
    assert value.cache_write_tokens == 0
    assert value.cache_read_tokens is None
    assert value.pricing_context["service_tier"] == "standard"
    assert normalize_usage(replace(value, cache_write_tokens=12)).cache_write_tokens == 12


@pytest.mark.parametrize("protocol", [None, "anthropic", "gemini"])
def test_other_protocols_keep_unknown_cache_write(protocol):
    value = gateway()
    value = replace(value, pricing_context={**value.pricing_context, "usage_protocol": protocol})
    assert normalize_usage(value).cache_write_tokens is None


def test_failed_requests_without_usage_remain_unknown():
    value = gateway(input_tokens=None, output_tokens=None)
    value = replace(value, pricing_context={**value.pricing_context, "raw_usage": {}})
    assert normalize_usage(value).cache_write_tokens is None
    assert normalize_usage(value).input_tokens is None


@pytest.mark.parametrize("tier", ["flex", "priority", "scale"])
def test_explicit_service_tiers_are_preserved(tier):
    value = gateway()
    value = replace(value, pricing_context={**value.pricing_context, "service_tier": tier})
    assert normalize_usage(value).pricing_context["service_tier"] == tier


@pytest.mark.parametrize(
    "fields",
    [
        {"provider": "mandri"},
        {"pricing_context": {"observed_provider": "mandri"}},
        {"pricing_context": {"routing": "gateway"}},
        {"pricing_context": {"model_basis": "mandri_route_model_reference"}},
    ],
)
def test_gateway_harness_evidence_never_contributes_usage(fields):
    value = gateway(source="native:opencode", model="mandri_gateway", **fields)
    assert not normalize_usage(value).authoritative
    assert normalize_usage(gateway(source="native:opencode", provider="opencode_go")).authoritative


@pytest.mark.parametrize("status", ["failed", "cancelled"])
@pytest.mark.parametrize("output", [None, 0])
def test_unsuccessful_requests_without_output_are_not_usage(status, output):
    value = gateway(
        input_tokens=100,
        output_tokens=output,
        request_count=1,
        pricing_context={"status": status},
    )
    assert not normalize_usage(value).authoritative


@pytest.mark.parametrize(
    "fields",
    [
        {"output_tokens": 1},
        {"reasoning_tokens": 1},
        {"reported_cost_usd": Decimal("0.01")},
        {"pricing_context": {"status": "failed", "output_observed": True}},
        {"pricing_context": {"status": "completed"}},
        {"pricing_context": {"status": "pending"}},
    ],
)
def test_observed_consumption_and_other_statuses_remain_usage(fields):
    value = gateway(
        **{
            "output_tokens": None,
            "request_count": 1,
            "pricing_context": {"status": "failed"},
            **fields,
        }
    )
    assert normalize_usage(value).authoritative
