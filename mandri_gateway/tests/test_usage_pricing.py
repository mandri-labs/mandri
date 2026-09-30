from dataclasses import replace
from decimal import Decimal

import pytest
from mandri.core.ids import ModelRef, ProviderKind, RouteId, SecretRef
from mandri.core.usage_pricing import value_usage
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.types.model import Model
from mandri.gateway.usage import GatewayUsageRecord, UsageCollector
from mandri.gateway.usage_observation import to_observation
from mandri.gateway.usage_payload import observe_payload
from mandri.providers.service import Provider, ProviderState

from mandri_core.tests.usage_fixtures import REVIEWED_AT, synthetic_prices


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-4.1-mini", "0.00054"),
        ("gpt-5.3-codex", "0.00054"),
    ],
)
async def test_realistic_gateway_payload_values_without_invented_subsets(model: str, expected: str):
    records: list[GatewayUsageRecord] = []

    async def sink(record: GatewayUsageRecord) -> None:
        records.append(record)

    route = ResolvedRoute(
        RouteId("price-fixture"),
        Provider(
            "custom-provider-instance",
            ProviderKind.OPENAI,
            None,
            SecretRef("fixture"),
            ProviderState.VERIFIED,
        ),
        Model(ProviderKind.OPENAI, ModelRef(model), None, SecretRef("fixture")),
        conversation_id=None,
    )
    collector = UsageCollector(route, "chat", sink)
    collector.record = replace(collector.record, started_at=REVIEWED_AT, transport_attempts=1)
    await observe_payload(
        collector,
        {
            "id": "response-fixture",
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "prompt_tokens_details": {"cached_tokens": 600},
            },
        },
    )
    await collector.finish("completed")
    observation = to_observation(records[-1])
    assert observation.provider == "custom-provider-instance"
    assert observation.pricing_context["provider_kind"] == "openai"
    assert observation.pricing_context["raw_usage"]
    assert observation.cache_write_tokens == 0
    assert observation.reasoning_tokens is None
    amount, price_id = value_usage(observation, synthetic_prices(model))
    assert amount == Decimal(expected)
    assert price_id is not None
    assert value_usage(replace(observation, cache_read_tokens=None), synthetic_prices(model)) == (
        None,
        None,
    )
