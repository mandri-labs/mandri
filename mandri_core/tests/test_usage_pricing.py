from dataclasses import replace
from decimal import Decimal, localcontext

import pytest
from mandri.core.types.usage import UsageObservation
from mandri.core.usage_price_identity import pricing_provider
from mandri.core.usage_pricing import (
    TOKEN_FIELDS,
    disjoint_tokens,
    request_context_tokens,
    validate_price,
    value_usage,
)

from mandri_core.tests.usage_fixtures import REVIEWED_AT, synthetic_prices


@pytest.fixture
def observation() -> UsageObservation:
    return UsageObservation(
        source="fixture",
        source_key="request",
        fact_key="request",
        provider="openai",
        model="gpt-4.1-mini",
        occurred_at=REVIEWED_AT,
        input_tokens=1000,
        cache_read_tokens=600,
        cache_write_tokens=0,
        output_tokens=200,
        reasoning_tokens=0,
        input_includes_cache=True,
        output_includes_reasoning=True,
        pricing_context={"comparison_basis": "standard_text_api"},
    )


def test_disjoint_cache_and_reasoning_are_counted_once(observation):
    value = replace(observation, reasoning_tokens=50)
    assert disjoint_tokens(value) == dict(zip(TOKEN_FIELDS, [400, 600, 0, 150, 50], strict=True))
    assert value_usage(value, synthetic_prices())[0] == Decimal("0.00054")
    assert value_usage(replace(value, reported_cost_usd=Decimal(9)), synthetic_prices())[
        0
    ] == Decimal("0.00054")


@pytest.mark.parametrize("field", TOKEN_FIELDS)
@pytest.mark.parametrize("invalid", [None, -1, True, 1.5])
def test_unknown_and_invalid_counters_are_never_zero(observation, field, invalid):
    result = value_usage(replace(observation, **{field: invalid}), synthetic_prices())[0]
    if invalid is None and field in {"cache_write_tokens", "reasoning_tokens"}:
        assert result == Decimal("0.00054")
    else:
        assert result is None


@pytest.mark.parametrize(
    "changes",
    [
        {"input_includes_cache": None},
        {"output_includes_reasoning": None},
        {"cache_read_tokens": 1001},
        {"reasoning_tokens": 201},
    ],
)
def test_overlap_requires_evidence_and_valid_subsets(observation, changes):
    assert disjoint_tokens(replace(observation, **changes)) is None
    assert value_usage(replace(observation, **changes), synthetic_prices()) == (None, None)


@pytest.mark.parametrize("model", ["auto", "other", "GPT-4.1-mini", None])
def test_calculator_requires_exact_imported_identity(observation, model):
    assert value_usage(replace(observation, model=model), synthetic_prices()) == (None, None)


@pytest.mark.parametrize(
    "changes",
    [
        {"modality": "audio"},
        {"modality": "image"},
        {"service_tier": "priority"},
        {"service_tier": "batch"},
        {"region": "us"},
        {"modifiers": ["search"]},
    ],
)
def test_reference_comparison_respects_conflicting_evidence(observation, changes):
    value = replace(observation, pricing_context={**observation.pricing_context, **changes})
    assert value_usage(value, synthetic_prices()) == (None, None)


@pytest.mark.parametrize("kind,baseline", [("cumulative", False), ("delta", True)])
def test_only_nonbaseline_deltas_are_priced(observation, kind, baseline):
    assert value_usage(replace(observation, kind=kind, baseline=baseline), synthetic_prices()) == (
        None,
        None,
    )


@pytest.mark.parametrize("when", [None, 1])
def test_current_comparison_can_value_old_or_undated_usage(observation, when):
    assert value_usage(replace(observation, occurred_at=when), synthetic_prices())[0] == Decimal(
        "0.00054"
    )


def test_historical_override_is_date_bounded(observation):
    price = replace(
        synthetic_prices()[0],
        source="user_override",
        price_id="override",
        valuation_basis="historical_tariff",
        effective_from=10,
        effective_to=20,
        rates={key: Decimal(2) for key in TOKEN_FIELDS},
    )
    assert value_usage(replace(observation, occurred_at=15), [*synthetic_prices(), price]) == (
        Decimal("0.0024"),
        "override",
    )
    assert value_usage(replace(observation, occurred_at=20), [*synthetic_prices(), price])[
        0
    ] == Decimal("0.00054")
    assert value_usage(replace(observation, occurred_at=15, interval_start=9), [price]) == (
        None,
        None,
    )


def test_instance_override_preserves_original_attribution(observation):
    value = replace(
        observation,
        provider="account",
        pricing_context={"provider_kind": "openai", "comparison_basis": "standard_text_api"},
    )
    price = replace(
        synthetic_prices()[0],
        provider="account",
        source="user_override",
        price_id="custom",
        rates={key: Decimal(1) for key in TOKEN_FIELDS},
    )
    assert value_usage(value, [*synthetic_prices(), price]) == (Decimal("0.0012"), "custom")
    assert value.provider == "account"


def test_fresher_snapshot_supersedes_entire_old_snapshot(observation):
    original = synthetic_prices()[0]
    newer = replace(
        original,
        price_id="new",
        reviewed_at=REVIEWED_AT + 1,
        rates={key: Decimal(1) for key in TOKEN_FIELDS},
    )
    assert value_usage(observation, [original, newer]) == (Decimal("0.0012"), "new")
    assert value_usage(
        observation, [original, replace(newer, constraints={"min_context_tokens": 5000})]
    ) == (None, None)


def test_direct_catalog_provider_precedes_router_equivalent(observation):
    direct = replace(
        synthetic_prices()[0], source="https://models.dev/api.json", source_model=observation.model
    )
    router = replace(
        direct,
        price_id="router",
        source="https://openrouter.ai/api/v1/models",
        reviewed_at=REVIEWED_AT + 1,
        rates={key: Decimal(99) for key in TOKEN_FIELDS},
    )
    assert value_usage(observation, [router, direct]) == (Decimal("0.00054"), direct.price_id)


def test_exact_model_precedes_an_imported_canonical_alias(observation):
    exact = replace(
        synthetic_prices()[0], source="https://models.dev/api.json", source_model=observation.model
    )
    alias = replace(
        exact,
        price_id="alias",
        source_model="older-alias",
        rates={key: Decimal(99) for key in TOKEN_FIELDS},
    )
    assert value_usage(observation, [alias, exact]) == (Decimal("0.00054"), exact.price_id)


def test_conflicting_same_priority_prices_are_not_guessed(observation):
    original = synthetic_prices()[0]
    other = replace(original, price_id="conflict", rates={key: Decimal(99) for key in TOKEN_FIELDS})
    assert value_usage(observation, [original, other]) == (None, None)


def test_calculation_is_independent_of_ambient_decimal_context(observation):
    with localcontext() as context:
        context.prec = 2
        assert value_usage(observation, synthetic_prices())[0] == Decimal("0.00054")


def test_unknown_subsets_are_usable_only_with_equal_rates(observation):
    missing = replace(observation, reasoning_tokens=None, cache_write_tokens=None)
    assert value_usage(missing, synthetic_prices())[0] == Decimal("0.00054")
    unequal = replace(
        synthetic_prices()[0], rates={**synthetic_prices()[0].rates, "reasoning_tokens": Decimal(9)}
    )
    assert value_usage(missing, [unequal]) == (None, None)
    assert value_usage(replace(missing, output_includes_reasoning=False), synthetic_prices()) == (
        None,
        None,
    )


def test_zero_parents_make_unknown_inclusive_subsets_irrelevant(observation):
    zero = replace(observation, input_tokens=0, cache_read_tokens=None, cache_write_tokens=None)
    assert value_usage(zero, synthetic_prices())[0] == Decimal("0.00032")
    assert value_usage(replace(zero, input_includes_cache=False), synthetic_prices()) == (
        None,
        None,
    )


def test_free_token_equivalent_does_not_fabricate_counts(observation):
    price = replace(synthetic_prices()[0], rates={key: Decimal(0) for key in TOKEN_FIELDS})
    value = replace(
        observation,
        **dict.fromkeys(TOKEN_FIELDS),
        input_includes_cache=None,
        output_includes_reasoning=None,
    )
    assert value_usage(value, [price]) == (Decimal(0), price.price_id)
    assert value.input_tokens is None
    assert value_usage(
        value, [replace(price, rates={"input_tokens": Decimal(0), "output_tokens": Decimal(0)})]
    ) == (None, None)


def test_tiers_require_request_scoped_full_input_evidence(observation):
    short = replace(synthetic_prices()[0], constraints={"max_context_tokens": 1000})
    long = replace(
        short,
        price_id="long",
        constraints={"min_context_tokens": 1001},
        rates={key: Decimal(1) for key in TOKEN_FIELDS},
    )
    value = replace(observation, request_count=1)
    assert request_context_tokens(value) == 1000
    assert value_usage(value, [short, long])[0] == Decimal("0.00054")
    assert value_usage(replace(value, input_tokens=1001), [short, long])[0] == Decimal("0.001201")
    assert request_context_tokens(replace(value, input_includes_cache=False)) == 1600
    for count in (None, True, 2):
        assert value_usage(replace(value, request_count=count), [short, long]) == (None, None)


@pytest.mark.parametrize(
    "source,provider,expected",
    [
        ("native:codex", None, "openai"),
        ("native:claude", None, "anthropic"),
        ("native:agy", None, "native:agy"),
        ("native:pi", None, "native:pi"),
        ("native:pi", "google", "gemini"),
        ("native:pi", "opencode-go", "opencode_go"),
        ("native:pi", "google-antigravity", "native:agy"),
        ("native:pi", "openai-codex", "openai"),
        ("gateway", "chatgpt", "openai"),
    ],
)
def test_native_reference_provider_is_distinct_from_harness(
    observation, source, provider, expected
):
    value = replace(observation, source=source, provider=provider)
    assert pricing_provider(value) == expected
    assert value.provider == provider


def test_unknown_explicit_provider_is_never_replaced_by_harness(observation):
    value = replace(
        observation,
        source="native:codex",
        provider=None,
        pricing_context={"comparison_basis": "standard_text_api", "observed_provider": "custom"},
    )
    assert pricing_provider(value) == "custom"
    assert value_usage(value, synthetic_prices()) == (None, None)
    assert value_usage(replace(observation, provider="custom"), synthetic_prices()) == (None, None)


@pytest.mark.parametrize(
    "changes",
    [
        {"currency": "EUR"},
        {"unit": "per_token"},
        {"rates": {"input_tokens": Decimal(-1)}},
        {"rates": {"unknown": Decimal(1)}},
        {"effective_to": 0},
        {"calculation_version": "unknown"},
        {"valuation_basis": "invoice"},
        {"reviewed_at": None},
        {"reviewed_at": True},
        {"constraints": {"min_context_tokens": -1}},
        {"constraints": {"min_context_tokens": 100, "max_context_tokens": 99}},
        {"constraints": {"utc_weekly_intervals": []}},
    ],
)
def test_invalid_tariff_metadata(changes):
    with pytest.raises(ValueError):
        validate_price(replace(synthetic_prices()[0], **changes))


def test_explicit_weekly_tariff_boundaries(observation):
    start = 4 * 86_400_000
    price = replace(synthetic_prices()[0], constraints={"utc_weekly_intervals": [[0, 60]]})
    value = replace(observation, occurred_at=start + 1, request_count=1)
    assert value_usage(value, [price])[0] == Decimal("0.00054")
    assert value_usage(replace(value, interval_start=start - 1), [price]) == (None, None)
    assert value_usage(replace(value, occurred_at=start + 3_600_000), [price]) == (None, None)
    assert value_usage(replace(value, request_count=2), [price]) == (None, None)
