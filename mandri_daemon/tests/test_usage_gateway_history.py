from dataclasses import replace
from decimal import Decimal

import pytest
from mandri.core.types.usage import UsageFilters, UsageObservation, UsagePrice
from mandri.daemon.usage_history import UsageHistorySync
from mandri.daemon.usage_history_routes import attribute_route
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage import UsageRepository
from mandri.database.usage_migrations import migrate_usage


def route_history():
    return [
        {
            "id": 1,
            "route_id": "route",
            "effective_from": 100,
            "provider_name": "go-account",
            "provider_kind": "opencode_go",
            "model_ref": "custom_openai/glm-5.3-flash",
        },
        {
            "id": 2,
            "route_id": "route",
            "effective_from": 200,
            "provider_name": "zen-account",
            "provider_kind": "opencode",
            "model_ref": "custom_openai/glm-5.3-flash",
        },
        {
            "id": 3,
            "route_id": "route",
            "effective_from": 300,
            "provider_name": "chatgpt-account",
            "provider_kind": "chatgpt",
            "model_ref": "openai/gpt-6-astra",
        },
    ]


def native(when=150):
    return UsageObservation(
        source="native:claude",
        source_key="native-message",
        fact_key="native-message",
        session_id="session",
        harness="claude",
        model="glm-5.3-flash",
        occurred_at=when,
        input_tokens=100,
        output_tokens=10,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=None,
        request_count=1,
        input_includes_cache=False,
        output_includes_reasoning=True,
        pricing_context={
            "evidence": "history_request",
            "comparison_basis": "standard_text_api",
            "upstream_request_id": "upstream-response",
        },
    )


@pytest.mark.parametrize(
    "when,provider,kind", [(150, "go-account", "opencode_go"), (250, "zen-account", "opencode")]
)
def test_history_uses_provider_at_request_time(when, provider, kind):
    value = attribute_route(native(when), route_history())
    assert value is not None
    assert value.provider == provider
    assert value.pricing_context["provider_kind"] == kind
    assert value.pricing_context["attributed_source"] == "gateway"
    assert value.harness == "claude"


@pytest.mark.parametrize("when", [99, 350, None])
def test_route_history_never_guesses_a_provider_for_unknown_model_or_time(when):
    assert attribute_route(native(when), route_history()) is None


@pytest.mark.parametrize("gateway_first", [False, True])
async def test_history_and_gateway_count_once_with_provider_and_harness_preserved(
    tmp_path, gateway_first
):
    db = AiosqliteDatabase()
    await db.connect(":memory:")
    await db.migrate()
    await migrate_usage(db._require_connection())
    repository = UsageRepository(db)
    sync = UsageHistorySync(db, repository, tmp_path)
    value = sync._attributed(
        native(),
        {
            "harness": "claude",
            "model_source": "gateway",
            "model": "chatgpt/gpt-6-astra",
            "route_history": route_history(),
        },
    )
    gateway = replace(
        value,
        source="gateway",
        source_key="gateway-request",
        fact_key="gateway-request",
        occurred_at=400,
        observed_at=500,
        complete=True,
        pricing_context={
            "provider_kind": "opencode_go",
            "comparison_basis": "standard_text_api",
            "upstream_request_id": "upstream-response",
            "status": "completed",
        },
    )
    try:
        await repository.add_price(
            UsagePrice(
                price_id="go-price",
                provider="opencode_go",
                model="glm-5.3-flash",
                effective_from=0,
                rates={
                    "input_tokens": Decimal("0.15"),
                    "output_tokens": Decimal("0.5"),
                    "reasoning_tokens": Decimal("0.5"),
                },
            )
        )
        for item in (gateway, value) if gateway_first else (value, gateway):
            await repository.record(item)
        result = await repository.overview()
        assert result["summary"]["fact_count"] == 1
        assert result["summary"]["request_count"] == 1
        assert result["summary"]["input_tokens"] == 100
        assert Decimal(result["summary"]["usd_equivalent"]) == Decimal("0.00002")
        assert result["source_breakdown"]["gateway"]["fact_count"] == 1
        assert result["source_breakdown"]["claude"]["fact_count"] == 0
        assert (await repository.overview(UsageFilters(provider="go-account")))["summary"][
            "fact_count"
        ] == 1
        assert (await repository.overview(UsageFilters(harness="opencode")))["summary"][
            "fact_count"
        ] == 0
        await repository.stage_history(
            [value], "history", {"session_id": "session"}, reset=True, complete=True
        )
        assert (await repository.overview())["summary"] == result["summary"]
    finally:
        await db.close()
