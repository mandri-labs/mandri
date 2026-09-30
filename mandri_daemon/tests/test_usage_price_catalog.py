from dataclasses import replace
from decimal import Decimal, localcontext

import pytest
from mandri.core.types.usage import UsageObservation
from mandri.core.usage_normalization import normalize_usage
from mandri.core.usage_pricing import value_usage
from mandri.daemon.usage_catalog_sources import decimal_rate, parse_models_dev, parse_openrouter

REVIEWED_AT = 1000


def model(cost: dict[str, object]) -> dict[str, object]:
    return {"cost": cost, "modalities": {"output": ["text"]}}


def request(provider: str, model: str, tokens: int = 1000) -> UsageObservation:
    return UsageObservation(
        source="fixture",
        source_key="one",
        fact_key="one",
        provider=provider,
        model=model,
        occurred_at=1,
        input_tokens=tokens,
        output_tokens=200,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        input_includes_cache=True,
        output_includes_reasoning=True,
        request_count=1,
        pricing_context={"comparison_basis": "standard_text_api"},
    )


@pytest.mark.parametrize(
    "value,unit,expected",
    [
        ("0.00000125", "per_token", "1.25"),
        (0, "per_token", "0"),
        ("12.5", "per_million_tokens", "12.5"),
        ("0.000000123456789123456789", "per_token", "0.123456789123456789"),
    ],
)
def test_units_use_decimal_without_ambient_rounding(value, unit, expected):
    with localcontext() as context:
        context.prec = 2
        assert decimal_rate(value, unit) == Decimal(expected)


@pytest.mark.parametrize("value", [True, None, "NaN", "Infinity", "-0.1", "oops", "1e-1000"])
def test_invalid_source_rates(value):
    with pytest.raises(ValueError):
        decimal_rate(value, "per_token")


def test_unsupported_unit():
    with pytest.raises(ValueError):
        decimal_rate("1", "per_1000")


def test_models_dev_uses_exact_provider_ids_and_no_cheapest_alias():
    payload = {
        "opencode": {"models": {"deepseek-v4.1-flash": model({"input": "0.3", "output": "1.2"})}},
        "opencode-go": {
            "models": {
                "deepseek-v4.1-flash": model({"input": "0.15", "output": "0.6"}),
                "glm-5.3-flash": model({"input": "0.15", "output": "0.5"}),
                "minimax-m3": model({"input": "0.3", "output": "1.2"}),
                "omen-alpha": model({"input": "0.2", "output": "0.66"}),
            }
        },
        "google": {"models": {"gemini-test": model({"input": 1, "output": 2})}},
    }
    prices = parse_models_dev(payload, REVIEWED_AT)
    assert value_usage(request("opencode", "deepseek-v4.1-flash"), prices)[0] == Decimal("0.00054")
    assert value_usage(request("opencode_go", "deepseek-v4.1-flash"), prices)[0] == Decimal(
        "0.00027"
    )
    assert value_usage(request("opencode_go", "omen-alpha"), prices)[0] == Decimal("0.000332")
    for provider, name in [
        ("opencode", "omen-alpha"),
        ("opencode_go", "omen"),
        ("custom", "omen-alpha"),
    ]:
        assert value_usage(request(provider, name), prices) == (None, None)
    assert value_usage(request("gemini", "gemini-test"), prices)[0] == Decimal("0.0014")


def test_models_dev_tiers_override_legacy_200k_name_and_include_cached_input():
    cost = {
        "input": 10,
        "output": 50,
        "cache_read": 1,
        "cache_write": "12.5",
        "tiers": [
            {
                "input": 20,
                "output": 75,
                "cache_read": 2,
                "cache_write": 25,
                "tier": {"type": "context", "size": 272000},
            }
        ],
        "context_over_200k": {"input": 999, "output": 999},
    }
    prices = parse_models_dev({"openai": {"models": {"gpt-6-astra": model(cost)}}}, REVIEWED_AT)
    short = request("openai", "gpt-6-astra", 272000)
    long = replace(short, input_tokens=272001)
    assert value_usage(short, prices)[0] == Decimal("2.73")
    assert value_usage(long, prices)[0] == Decimal("5.45502")
    cached = replace(long, cache_read_tokens=272000, cache_write_tokens=1)
    assert value_usage(cached, prices)[0] == Decimal("0.559025")
    assert value_usage(replace(long, request_count=None), prices) == (None, None)


def test_legacy_models_dev_tier_and_multiple_explicit_tiers():
    payload = {
        "openai": {
            "models": {
                "model": model(
                    {
                        "input": 1,
                        "output": 2,
                        "context_over_200k": {"input": 3, "output": 4},
                    }
                )
            }
        }
    }
    prices = parse_models_dev(payload, REVIEWED_AT)
    assert value_usage(request("openai", "model", 200001), prices)[0] == Decimal("0.600803")
    payload["openai"]["models"]["model"]["cost"]["tiers"] = [
        {"tier": {"type": "context", "size": 2000}, "input": 5, "output": 6},
        {"tier": {"type": "context", "size": 1000}, "input": 3, "output": 4},
    ]
    prices = parse_models_dev(payload, REVIEWED_AT)
    assert value_usage(request("openai", "model", 1001), prices)[0] == Decimal("0.003803")
    assert value_usage(request("openai", "model", 2001), prices)[0] == Decimal("0.011205")


def test_bad_models_are_skipped_without_losing_valid_peers():
    payload = {
        "openai": {
            "models": {
                "valid": model({"input": 1, "output": 2}),
                "negative": model({"input": -1, "output": 2}),
                "missing": model({"input": 1}),
                "unknown-tier": model(
                    {
                        "input": 1,
                        "output": 2,
                        "tiers": [
                            {"tier": {"type": "daily", "size": 1000}, "input": 3, "output": 4}
                        ],
                    }
                ),
                "duplicate-tiers": model(
                    {
                        "input": 1,
                        "output": 2,
                        "tiers": [
                            {"tier": {"type": "context", "size": 1000}, "input": 3, "output": 4},
                            {"tier": {"type": "context", "size": 1000}, "input": 5, "output": 6},
                        ],
                    }
                ),
                "mismatch": {"id": "different", **model({"input": 1, "output": 2})},
            }
        }
    }
    assert [price.model for price in parse_models_dev(payload, REVIEWED_AT)] == [
        "valid",
        "valid",
        "valid",
    ]


def test_openrouter_free_missing_cache_is_zero_and_provider_is_exact():
    payload = {
        "data": [
            {
                "id": "vendor/model:free",
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {"output_modalities": ["text"]},
            }
        ]
    }
    prices = parse_openrouter(payload, REVIEWED_AT)
    free = replace(
        request("openrouter", "vendor/model:free"),
        cache_read_tokens=None,
        cache_write_tokens=None,
        reasoning_tokens=None,
    )
    assert value_usage(free, prices)[0] == Decimal(0)
    assert value_usage(replace(free, model="vendor/model"), prices) == (None, None)
    assert value_usage(replace(free, provider="vendor"), prices) == (None, None)


def test_openrouter_per_token_rates_and_prompt_tiers():
    payload = {
        "data": [
            {
                "id": "openai/gpt-6-astra",
                "architecture": {"output_modalities": ["text"]},
                "pricing": {
                    "prompt": "0.00001",
                    "completion": "0.00005",
                    "input_cache_read": "0.000001",
                    "input_cache_write": "0.0000125",
                    "web_search": "0.01",
                    "overrides": [
                        {
                            "min_prompt_tokens": 272000,
                            "prompt": "0.00002",
                            "completion": "0.000075",
                            "input_cache_read": "0.000002",
                            "input_cache_write": "0.000025",
                        }
                    ],
                },
            }
        ]
    }
    prices = parse_openrouter(payload, REVIEWED_AT)
    assert value_usage(request("openrouter", "openai/gpt-6-astra", 272000), prices)[0] == Decimal(
        "2.73"
    )
    assert value_usage(request("openrouter", "openai/gpt-6-astra", 272001), prices)[0] == Decimal(
        "5.45502"
    )


@pytest.mark.parametrize(
    "parse,payload",
    [
        (parse_models_dev, []),
        (parse_models_dev, {}),
        (parse_openrouter, {}),
        (parse_openrouter, {"data": []}),
    ],
)
def test_empty_or_incompatible_catalog_is_failure(parse, payload):
    with pytest.raises(ValueError):
        parse(payload, REVIEWED_AT)


@pytest.mark.parametrize(
    "provider,name,cost,expected",
    [
        (
            "opencode-go",
            "glm-5.3-flash",
            {"input": "0.15", "output": "0.5", "cache_read": "0.03"},
            "0.000178",
        ),
        (
            "opencode-go",
            "glm-5.3",
            {"input": "1.4", "output": "4.4", "cache_read": "0.26"},
            "0.001596",
        ),
        (
            "opencode-go",
            "muse-spark-1.3-contributor",
            {"input": "0.1", "output": "0.2", "cache_read": "0.002"},
            "0.0000812",
        ),
        (
            "opencode",
            "muse-spark-1.3",
            {"input": "1.25", "output": "4.25", "cache_read": "0.15"},
            "0.00144",
        ),
        (
            "opencode",
            "muse-spark-1.3-contributor-free",
            {"input": "0", "output": "0", "cache_read": "0"},
            "0",
        ),
        (
            "opencode-go",
            "mimo-v2.6-flash",
            {"input": "0.14", "output": "0.28", "cache_read": "0.0028"},
            "0.00011368",
        ),
        (
            "opencode-go",
            "mimo-v2.6-pro",
            {"input": "0.435", "output": "0.87", "cache_read": "0.003625"},
            "0.000350175",
        ),
        (
            "opencode-go",
            "gpt-5.6-luna",
            {"input": "0.2", "output": "1.2", "cache_read": "0.02", "cache_write": "0.25"},
            "0.000332",
        ),
    ],
)
def test_current_gateway_models_with_optional_usage_fields(provider, name, cost, expected):
    prices = parse_models_dev({provider: {"models": {name: model(cost)}}}, REVIEWED_AT)
    kind = provider.replace("-", "_")
    value = replace(
        request(kind, "upstream-alias"),
        source="gateway",
        cache_read_tokens=600,
        cache_write_tokens=None,
        reasoning_tokens=None,
        pricing_context={
            "comparison_basis": "standard_text_api",
            "provider_kind": kind,
            "selected_model": "custom_openai/" + name,
            "service_tier": "auto",
            "usage_protocol": "openai",
            "raw_usage": {"prompt_tokens": 1000, "completion_tokens": 200},
        },
    )
    assert value_usage(normalize_usage(value), prices)[0] == Decimal(expected)
    if expected == "0":
        assert (
            value_usage(
                normalize_usage(replace(value, input_tokens=None, output_tokens=None)), prices
            )[0]
            == 0
        )


def test_openrouter_responses_auto_tier_and_absent_cache_write_are_valued():
    prices = parse_openrouter(
        {
            "data": [
                {
                    "id": "z-ai/glm-5.3",
                    "architecture": {"output_modalities": ["text"]},
                    "pricing": {
                        "prompt": "0.00000084",
                        "completion": "0.00000264",
                        "input_cache_read": "0.000000156",
                    },
                }
            ]
        },
        REVIEWED_AT,
    )
    value = replace(
        request("openrouter", "z-ai/glm-5.3"),
        source="gateway",
        cache_write_tokens=None,
        pricing_context={
            "comparison_basis": "standard_text_api",
            "service_tier": "auto",
            "usage_protocol": "responses",
            "raw_usage": {"input_tokens": 1000, "output_tokens": 200},
        },
    )
    assert value_usage(normalize_usage(value), prices)[0] == Decimal("0.001368")


def test_multimodal_mimo_uses_text_comparison_without_changing_observed_modality():
    prices = parse_models_dev(
        {
            "opencode-go": {
                "models": {
                    "mimo-v2.6-pro": model(
                        {"input": "0.435", "output": "0.87", "cache_read": "0.003625"}
                    )
                }
            }
        },
        REVIEWED_AT,
    )
    value = replace(
        request("go-account", "mimo-v2.6-pro"),
        source="gateway",
        harness="claude",
        cache_read_tokens=600,
        reasoning_tokens=50,
        pricing_context={
            "comparison_basis": "standard_text_api",
            "provider_kind": "opencode_go",
            "modality": "multimodal",
        },
    )
    assert value_usage(value, prices)[0] == Decimal("0.000350175")
    assert value.pricing_context["modality"] == "multimodal"
    assert value_usage(replace(value, pricing_context={"modality": "multimodal"}), prices) == (
        None,
        None,
    )
    historical = replace(prices[0], valuation_basis="historical_tariff")
    assert value_usage(value, [historical]) == (None, None)
    override = replace(prices[0], source="user_override")
    assert value_usage(value, [override]) == (None, None)
