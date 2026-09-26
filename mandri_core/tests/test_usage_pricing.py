from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal, localcontext

import pytest
from mandri.core.types.usage import UsageObservation, UsagePrice
from mandri.core.usage_pricing import (
    REVIEWED_AT,
    TOKEN_FIELDS,
    bundled_prices,
    disjoint_tokens,
    request_context_tokens,
    validate_price,
    value_usage,
)


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
        pricing_context={
            "modality": "text",
            "service_tier": "standard",
            "region": "global",
            "modifiers": [],
        },
    )


def test_catalog_is_versioned_reviewed_and_independent() -> None:
    first = bundled_prices()
    assert len(first) == 29
    assert len({p.price_id for p in first}) == len(first)
    assert first == bundled_prices()
    for price in first:
        validate_price(price)
        assert price.reviewed_at == REVIEWED_AT
        assert price.effective_from == 0
        assert price.valuation_basis == "current_price_comparison"
        assert price.source.startswith("https://")
    first[0].rates.clear()
    first[0].constraints["modifiers"].append("batch")
    assert bundled_prices()[0].rates
    assert bundled_prices()[0].constraints["modifiers"] == []


def test_openai_exact_aliases(observation: UsageObservation) -> None:
    for model in ("gpt-4.1-mini", "gpt-4.1-mini-2025-04-14"):
        amount, price_id = value_usage(replace(observation, model=model), bundled_prices())
        assert amount == Decimal("0.00054")
        assert price_id is not None and price_id.endswith(model)


@pytest.mark.parametrize("model", ["auto", "GPT-4.1-mini", "gpt-4.1-mini-latest", "gpt-4.1", None])
def test_no_fuzzy_model_lookup(observation: UsageObservation, model: str | None) -> None:
    assert value_usage(replace(observation, model=model), bundled_prices()) == (None, None)


@pytest.mark.parametrize("when", [None, REVIEWED_AT - 1])
def test_current_catalog_values_old_or_undated_usage(
    observation: UsageObservation, when: int | None
) -> None:
    amount, price_id = value_usage(replace(observation, occurred_at=when), bundled_prices())
    assert amount == Decimal("0.00054")
    price = next(price for price in bundled_prices() if price.price_id == price_id)
    assert price.valuation_basis == "current_price_comparison"
    assert price.reviewed_at == REVIEWED_AT


def test_covered_interval_cannot_cross_tariff_boundary(observation: UsageObservation) -> None:
    historical = replace(
        bundled_prices()[0],
        valuation_basis="historical_tariff",
        effective_from=REVIEWED_AT,
    )
    assert value_usage(replace(observation, interval_start=REVIEWED_AT - 1), [historical]) == (
        None,
        None,
    )


@pytest.mark.parametrize("field", TOKEN_FIELDS)
@pytest.mark.parametrize("invalid", [None, -1, True, 1.5])
def test_invalid_counters_are_not_zero(
    observation: UsageObservation,
    field: str,
    invalid: object,
) -> None:
    if field in {"cache_write_tokens", "reasoning_tokens"} and invalid is None:
        assert value_usage(replace(observation, **{field: None}), bundled_prices())[0] == (
            Decimal("0.00054")
        )
        return
    assert value_usage(replace(observation, **{field: invalid}), bundled_prices()) == (None, None)


def test_overlap_requires_evidence_and_nonnegative_subsets(observation: UsageObservation) -> None:
    for changes in (
        {"input_includes_cache": None},
        {"output_includes_reasoning": None},
        {"cache_read_tokens": 1001},
        {"reasoning_tokens": 201},
    ):
        assert disjoint_tokens(replace(observation, **changes)) is None


def test_synthetic_plan_example_no_double_counting(observation: UsageObservation) -> None:
    price = UsagePrice(
        price_id="synthetic",
        provider="openai",
        model="gpt-4.1-mini",
        effective_from=0,
        rates={
            "input_tokens": Decimal("2"),
            "cache_read_tokens": Decimal("0.20"),
            "output_tokens": Decimal("10"),
            "reasoning_tokens": Decimal("10"),
        },
    )
    observation = replace(observation, reasoning_tokens=50)
    assert disjoint_tokens(observation) == dict(
        zip(TOKEN_FIELDS, [400, 600, 0, 150, 50], strict=True)
    )
    assert value_usage(observation, [price]) == (Decimal("0.00292"), "synthetic")
    assert value_usage(replace(observation, reported_cost_usd=Decimal("9")), [price])[0] == Decimal(
        "0.00292"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"modality": "audio"},
        {"modality": "image"},
        {"service_tier": "batch"},
        {"service_tier": "priority"},
        {"region": "us"},
        {"modifiers": ["web_search"]},
    ],
)
def test_bundle_requires_supported_qualifiers(
    observation: UsageObservation,
    changes: dict[str, object],
) -> None:
    context = {**observation.pricing_context, **changes} if changes else {}
    assert value_usage(replace(observation, pricing_context=context), bundled_prices()) == (
        None,
        None,
    )


def test_unsupported_positive_dimensions_stay_unpriced(observation: UsageObservation) -> None:
    observation = replace(
        observation,
        provider="gemini",
        model="gemini-2.5-flash-lite",
        cache_write_tokens=1,
        pricing_context={"comparison_basis": "standard_text_api"},
    )
    assert value_usage(observation, bundled_prices()) == (None, None)


def test_anthropic_cache_ttl_context_and_disjoint_output(observation: UsageObservation) -> None:
    observation = replace(
        observation,
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        input_includes_cache=False,
        output_includes_reasoning=False,
        input_tokens=100,
        cache_read_tokens=200,
        cache_write_tokens=300,
        output_tokens=40,
        reasoning_tokens=10,
        pricing_context={
            **observation.pricing_context,
            "context_tokens": 600,
            "cache_write_ttl_seconds": 300,
        },
    )
    assert value_usage(observation, bundled_prices())[0] == Decimal("0.000745")
    for key, value in (
        ("cache_write_ttl_seconds", None),
        ("cache_write_ttl_seconds", 3600),
    ):
        assert value_usage(
            replace(
                observation,
                pricing_context={
                    **observation.pricing_context,
                    key: value,
                },
            ),
            bundled_prices(),
        ) == (None, None)
    observation = replace(
        observation,
        cache_write_tokens=0,
        pricing_context={
            **observation.pricing_context,
            "cache_write_ttl_seconds": None,
        },
    )
    assert value_usage(observation, bundled_prices())[0] is not None


def test_gemini_thinking_and_explicit_cache_storage(observation: UsageObservation) -> None:
    observation = replace(
        observation,
        provider="gemini",
        model="gemini-2.5-flash-lite",
        reasoning_tokens=50,
        output_includes_reasoning=False,
        pricing_context={**observation.pricing_context, "cache_mode": "implicit"},
    )
    assert value_usage(observation, bundled_prices())[0] == Decimal("0.000146")
    for cache_mode in (None, "explicit"):
        assert value_usage(
            replace(
                observation,
                pricing_context={
                    **observation.pricing_context,
                    "cache_mode": cache_mode,
                },
            ),
            bundled_prices(),
        ) == (None, None)


def test_explicit_override_and_half_open_history(observation: UsageObservation) -> None:
    price = UsagePrice(
        price_id="override",
        provider=observation.provider,
        model=observation.model,
        effective_from=REVIEWED_AT - 100,
        effective_to=REVIEWED_AT + 100,
        rates={key: Decimal("1") for key in TOKEN_FIELDS},
    )
    assert value_usage(observation, [*bundled_prices(), price]) == (Decimal("0.0012"), "override")
    assert value_usage(replace(observation, occurred_at=REVIEWED_AT - 100), [price])[0] is not None
    assert value_usage(replace(observation, occurred_at=REVIEWED_AT + 100), [price]) == (None, None)
    assert value_usage(observation, [price, replace(price, price_id="ambiguous")]) == (None, None)


def test_zero_requires_known_counters(observation: UsageObservation) -> None:
    zero = replace(observation, **dict.fromkeys(TOKEN_FIELDS, 0))
    assert value_usage(zero, bundled_prices())[0] == Decimal(0)
    assert value_usage(replace(zero, input_tokens=None), bundled_prices()) == (None, None)


@pytest.mark.parametrize(
    "changes",
    [
        {"currency": "EUR"},
        {"unit": "tokens"},
        {"calculation_version": "unknown"},
        {"effective_to": 0},
        {"rates": {"input_tokens": Decimal("NaN")}},
        {"rates": {"input_tokens": Decimal("Infinity")}},
        {"rates": {"input_tokens": Decimal("-1")}},
        {"rates": {"input_tokens": 0.1}},
        {"rates": {"audio_seconds": Decimal("1")}},
    ],
)
def test_bad_tariffs_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        validate_price(replace(bundled_prices()[0], **changes))


def test_decimal_precision_is_independent_of_ambient_context(observation: UsageObservation) -> None:
    expected = bundled_prices()
    with localcontext() as context:
        context.prec = 1
        assert bundled_prices() == expected
        assert value_usage(observation, bundled_prices())[0] == Decimal("0.00054")


def test_cumulative_and_baseline_are_not_billable_deltas(observation: UsageObservation) -> None:
    assert value_usage(replace(observation, kind="cumulative"), bundled_prices()) == (None, None)
    assert value_usage(replace(observation, baseline=True), bundled_prices()) == (None, None)


@pytest.mark.parametrize(
    "provider,model,expected",
    [
        ("openai", "gpt-5.3-codex", "0.003605"),
        ("anthropic", "claude-sonnet-4-6", "0.00438"),
        ("anthropic", "claude-opus-4-6", "0.0073"),
        ("anthropic", "claude-sonnet-5", "0.00292"),
        ("anthropic", "claude-opus-5", "0.0073"),
        ("gemini", "gemini-2.5-pro", "0.002575"),
        ("gemini", "gemini-3.1-pro-preview", "0.00332"),
        ("gemini", "gemini-3.1-pro-preview-customtools", "0.00332"),
    ],
)
def test_reference_scenario_keeps_custom_provider_attribution(
    observation: UsageObservation,
    provider: str,
    model: str,
    expected: str,
) -> None:
    observation = replace(
        observation,
        provider="custom-instance",
        model=model,
        pricing_context={
            "comparison_basis": "standard_text_api",
            "provider_kind": provider,
            "modality": "text",
            "context_tokens": 1000,
            **({"cache_mode": "implicit"} if provider == "gemini" else {}),
        },
    )
    assert value_usage(observation, bundled_prices())[0] == Decimal(expected)
    assert observation.provider == "custom-instance"


def test_reference_scenario_does_not_override_conflicting_evidence(
    observation: UsageObservation,
) -> None:
    for key, value in (
        ("modality", "audio"),
        ("service_tier", "priority"),
        ("modifiers", ["search"]),
        ("region", "us"),
    ):
        context = {"comparison_basis": "standard_text_api", "modality": "text", key: value}
        assert value_usage(replace(observation, pricing_context=context), bundled_prices()) == (
            None,
            None,
        )


@pytest.mark.parametrize("model", ["gemini-2.5-pro", "gemini-3.1-pro-preview"])
def test_pro_context_threshold_requires_request_evidence(
    observation: UsageObservation,
    model: str,
) -> None:
    observation = replace(observation, provider="gemini", model=model, cache_read_tokens=0)
    for context_tokens in (None, True, -1, 200001):
        assert value_usage(
            replace(
                observation,
                pricing_context={
                    **observation.pricing_context,
                    "context_tokens": context_tokens,
                },
            ),
            bundled_prices(),
        ) == (None, None)
    assert (
        value_usage(
            replace(
                observation,
                pricing_context={
                    **observation.pricing_context,
                    "context_tokens": 200000,
                },
            ),
            bundled_prices(),
        )[0]
        is not None
    )


def test_provider_kind_never_guessed_from_model(observation: UsageObservation) -> None:
    assert value_usage(replace(observation, provider="custom-instance"), bundled_prices()) == (
        None,
        None,
    )


def test_metadata_does_not_disqualify_tariff(observation: UsageObservation) -> None:
    enriched = replace(
        observation,
        pricing_context={
            **observation.pricing_context,
            "raw_usage": {"input_tokens": 1000},
            "status": "completed",
            "route_id": "fixture",
        },
    )
    assert value_usage(enriched, bundled_prices()) == value_usage(observation, bundled_prices())


def test_native_claude_reference_groups_unknown_reasoning(observation: UsageObservation) -> None:
    native = replace(
        observation,
        provider="anthropic",
        model="claude-sonnet-4-6",
        input_includes_cache=False,
        reasoning_tokens=None,
        cache_write_tokens=100,
        pricing_context={"comparison_basis": "standard_text_api", "provider_kind": "anthropic"},
    )
    assert value_usage(native, bundled_prices())[0] == Decimal("0.006555")
    assert native.reasoning_tokens is None
    assert "modality" not in native.pricing_context
    assert value_usage(replace(native, cache_write_tokens=None), bundled_prices()) == (None, None)


def test_unknown_inclusive_subsets_require_equal_rates(observation: UsageObservation) -> None:
    native = replace(
        observation, model="gpt-5.3-codex", reasoning_tokens=None, cache_write_tokens=None
    )
    assert value_usage(native, bundled_prices())[0] == Decimal("0.003605")
    assert value_usage(replace(native, cache_read_tokens=None), bundled_prices()) == (None, None)
    assert value_usage(replace(native, output_includes_reasoning=False), bundled_prices()) == (
        None,
        None,
    )


def test_user_override_inclusive_output_and_priority(observation: UsageObservation) -> None:
    price = UsagePrice(
        price_id="custom",
        provider="openai",
        model="gpt-4.1-mini",
        effective_from=0,
        source="user_override",
        rates={
            "input_tokens": Decimal("2"),
            "cache_read_tokens": Decimal("0.2"),
            "output_tokens": Decimal("10"),
        },
    )
    assert value_usage(replace(observation, reasoning_tokens=50), [*bundled_prices(), price]) == (
        Decimal("0.00292"),
        "custom",
    )
    assert value_usage(replace(observation, reasoning_tokens=None), [price])[0] == Decimal(
        "0.00292"
    )


def test_provider_instance_override_takes_precedence_over_vendor(
    observation: UsageObservation,
) -> None:
    observation = replace(
        observation,
        provider="local-provider",
        pricing_context={"provider_kind": "openai", "comparison_basis": "standard_text_api"},
    )
    override = UsagePrice(
        price_id="instance-rate",
        provider="local-provider",
        model=observation.model,
        effective_from=0,
        source="user_override",
        rates={key: Decimal("1") for key in TOKEN_FIELDS},
    )
    assert value_usage(observation, [*bundled_prices(), override]) == (
        Decimal("0.0012"),
        "instance-rate",
    )


@pytest.mark.parametrize(
    "model,amount",
    [
        ("gpt-6-astra", "0.0146"),
        ("gpt-5.6-sol", "0.00584"),
        ("gpt-5.6-terra", "0.00332"),
        ("gpt-5.6-luna", "0.000332"),
    ],
)
def test_latest_official_models_value_native_old_records(
    observation: UsageObservation, model: str, amount: str
) -> None:
    native = replace(
        observation,
        provider=None,
        model=model,
        occurred_at=REVIEWED_AT - 10_000,
        request_count=1,
        pricing_context={"provider_kind": "openai", "comparison_basis": "standard_text_api"},
    )
    assert value_usage(native, bundled_prices())[0] == Decimal(amount)


@pytest.mark.parametrize("prompt,expected", [(272_000, "2.7246"), (272_001, "5.44422")])
def test_astra_long_prompt_changes_full_request_rates(
    observation: UsageObservation, prompt: int, expected: str
) -> None:
    request = replace(observation, model="gpt-6-astra", input_tokens=prompt, request_count=1)
    assert value_usage(request, bundled_prices())[0] == Decimal(expected)


def test_astra_long_prompt_includes_cached_input_and_write(
    observation: UsageObservation,
) -> None:
    request = replace(
        observation,
        model="gpt-6-astra",
        input_tokens=272_001,
        cache_read_tokens=272_000,
        cache_write_tokens=1,
        request_count=1,
    )
    assert value_usage(request, bundled_prices())[0] == Decimal("0.559025")
    assert request_context_tokens(replace(request, input_tokens=0, input_includes_cache=False)) == (
        272_001
    )


def test_context_never_inferred_from_cumulative_or_multiple_requests(
    observation: UsageObservation,
) -> None:
    request = replace(observation, model="gpt-6-astra", input_tokens=900_000)
    assert request_context_tokens(request) is None
    assert value_usage(request, bundled_prices()) == (None, None)
    explicit = replace(request, pricing_context={**request.pricing_context, "context_tokens": 1000})
    assert request_context_tokens(explicit) == 1000
    assert request_context_tokens(replace(explicit, kind="cumulative")) is None
    assert request_context_tokens(replace(explicit, request_count=2)) is None
    assert request_context_tokens(replace(request, request_count=True)) is None


def test_zero_parent_makes_unknown_inclusive_cache_irrelevant(
    observation: UsageObservation,
) -> None:
    zero = replace(observation, input_tokens=0, cache_read_tokens=None, cache_write_tokens=None)
    assert value_usage(zero, bundled_prices())[0] == Decimal("0.00032")
    assert value_usage(replace(zero, input_includes_cache=False), bundled_prices()) == (None, None)


def test_missing_rate_is_irrelevant_when_its_quantity_is_zero(
    observation: UsageObservation,
) -> None:
    price = replace(
        bundled_prices()[0],
        rates={"input_tokens": Decimal(2), "output_tokens": Decimal(10)},
    )
    no_cache = replace(observation, cache_read_tokens=0)
    assert value_usage(no_cache, [price])[0] == Decimal("0.004")
    assert value_usage(observation, [price]) == (None, None)


def test_new_current_snapshot_supersedes_old_snapshot_without_backdating(
    observation: UsageObservation,
) -> None:
    original = bundled_prices()[0]
    newer = replace(
        original,
        price_id="new-price",
        reviewed_at=REVIEWED_AT + 1000,
        rates={key: Decimal(1) for key in TOKEN_FIELDS},
    )
    past = replace(observation, occurred_at=1)
    assert value_usage(past, [original, newer]) == (Decimal("0.0012"), "new-price")
    incompatible = replace(newer, constraints={"min_context_tokens": 5000})
    assert value_usage(past, [original, incompatible]) == (None, None)


def test_historical_override_remains_date_bounded_with_current_fallback(
    observation: UsageObservation,
) -> None:
    price = UsagePrice(
        price_id="historic",
        provider="openai",
        model="gpt-4.1-mini",
        source="user_override",
        effective_from=10,
        effective_to=20,
        rates={key: Decimal(2) for key in TOKEN_FIELDS},
    )
    assert value_usage(replace(observation, occurred_at=15), [*bundled_prices(), price]) == (
        Decimal("0.0024"),
        "historic",
    )
    assert value_usage(replace(observation, occurred_at=20), [*bundled_prices(), price])[0] == (
        Decimal("0.00054")
    )


def test_current_comparison_replaces_legacy_automatic_catalog(
    observation: UsageObservation,
) -> None:
    current = bundled_prices()[0]
    legacy = replace(
        current,
        price_id="2026-09-20.v1:openai:gpt-4.1-mini",
        effective_from=REVIEWED_AT,
        valuation_basis="historical_tariff",
    )
    assert value_usage(observation, [legacy, current]) == (Decimal("0.00054"), current.price_id)


@pytest.mark.parametrize(
    "changes",
    [
        {"valuation_basis": "invoice"},
        {"reviewed_at": None},
        {"reviewed_at": True},
        {"constraints": {"min_context_tokens": -1}},
        {"constraints": {"min_context_tokens": 100, "max_context_tokens": 99}},
    ],
)
def test_invalid_current_snapshot_metadata(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        validate_price(replace(bundled_prices()[0], **changes))


@pytest.mark.parametrize(
    "stamp,expected",
    [
        ("2026-09-21T00:59:59+00:00", "0.0001818"),
        ("2026-09-21T01:00:00+00:00", "0.0003636"),
        ("2026-09-21T03:59:59+00:00", "0.0003636"),
        ("2026-09-21T04:00:00+00:00", "0.0001818"),
        ("2026-09-21T06:00:00+00:00", "0.0003636"),
        ("2026-09-21T10:00:00+00:00", "0.0001818"),
        ("2026-09-20T01:00:00+00:00", "0.0001818"),
    ],
)
def test_opencode_go_exact_model_utc_peak_windows(
    observation: UsageObservation, stamp: str, expected: str
) -> None:
    request = replace(
        observation,
        provider="opencode_go",
        model="deepseek-v4-flash",
        occurred_at=int(datetime.fromisoformat(stamp).timestamp() * 1000),
        cache_write_tokens=None,
        reasoning_tokens=None,
        request_count=1,
    )
    assert value_usage(request, bundled_prices())[0] == Decimal(expected)
    assert request.cache_write_tokens is None
    assert value_usage(replace(request, cache_read_tokens=None), bundled_prices()) == (None, None)
    assert value_usage(replace(request, occurred_at=None), bundled_prices()) == (None, None)


def test_scheduled_price_rejects_cross_period_aggregate_and_old_flat_snapshot(
    observation: UsageObservation,
) -> None:
    when = int(datetime(2026, 9, 21, 1, tzinfo=UTC).timestamp() * 1000)
    request = replace(
        observation,
        provider="opencode_go",
        model="deepseek-v4-flash",
        occurred_at=when,
        request_count=1,
    )
    assert value_usage(replace(request, interval_start=when - 1), bundled_prices()) == (None, None)
    assert value_usage(replace(request, request_count=2), bundled_prices()) == (None, None)
    current = next(p for p in bundled_prices() if p.model == request.model)
    flat = replace(
        current,
        source="https://models.dev/api.json",
        constraints={},
        reviewed_at=REVIEWED_AT + 1000,
        price_id="flat",
    )
    with pytest.raises(ValueError, match="peak schedule"):
        validate_price(flat)
    assert value_usage(request, [flat, *bundled_prices()])[0] == Decimal("0.0003636")


def test_off_peak_request_can_cross_sunday_midnight(observation: UsageObservation) -> None:
    when = int(datetime(2026, 9, 21, 0, 1, tzinfo=UTC).timestamp() * 1000)
    request = replace(
        observation,
        provider="opencode_go",
        model="deepseek-v4-flash",
        occurred_at=when,
        interval_start=when - 120_000,
        request_count=1,
    )
    assert value_usage(request, bundled_prices())[0] == Decimal("0.0001818")


def test_free_token_equivalent_is_zero_without_fabricating_missing_counts(
    observation: UsageObservation,
) -> None:
    price = replace(bundled_prices()[0], rates={key: Decimal(0) for key in TOKEN_FIELDS})
    unknown = replace(
        observation,
        **dict.fromkeys(TOKEN_FIELDS),
        input_includes_cache=None,
        output_includes_reasoning=None,
    )
    assert value_usage(unknown, [price]) == (Decimal(0), price.price_id)
    assert unknown.input_tokens is None and unknown.output_tokens is None
    assert value_usage(replace(unknown, pricing_context={}), [price]) == (None, None)
    tools = replace(unknown, pricing_context={**unknown.pricing_context, "modifiers": ["search"]})
    assert value_usage(tools, [price]) == (None, None)
    partial = replace(price, rates={"input_tokens": Decimal(0), "output_tokens": Decimal(0)})
    assert value_usage(unknown, [partial]) == (None, None)
