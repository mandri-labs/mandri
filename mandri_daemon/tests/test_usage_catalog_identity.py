from dataclasses import replace
from decimal import Decimal

import pytest
from mandri.core.types.usage import UsageObservation
from mandri.core.usage_pricing import value_usage
from mandri.daemon.usage_catalog_sources import (
    MODELS_DEV_URL,
    OPENROUTER_URL,
    parse_models_dev,
    parse_openrouter,
)
from mandri.sessions.usage.adapter import to_usage_observation
from mandri.sessions.usage.normalize import normalize_native_usage
from mandri.sessions.usage.types import NativeUsageContext

OLD_MODELS = {
    "openai": (
        "gpt-4.1-mini",
        "gpt-5.3-codex",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-6-astra",
        "gpt-6.1-sol",
        "gpt-6-luna",
    ),
    "anthropic": (
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-sonnet-5",
        "claude-opus-5",
    ),
    "google": (
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-3.1-pro-preview",
        "gemini-3.1-pro-preview-customtools",
    ),
    "opencode-go": (
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-v4-flash-vision-exp",
        "deepseek-v4.1-flash",
    ),
}


def request(harness, model, provider=None):
    return UsageObservation(
        source="native:" + harness,
        source_key="a",
        fact_key="a",
        harness=harness,
        provider=provider,
        model=model,
        request_count=1,
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        input_includes_cache=False,
        output_includes_reasoning=True,
        pricing_context={"comparison_basis": "standard_text_api"},
    )


@pytest.mark.parametrize(
    "provider,model", [(p, m) for p, models in OLD_MODELS.items() for m in models]
)
def test_every_former_bundled_model_uses_downloaded_costs(provider, model):
    payload = {
        provider: {
            "models": {
                model: {
                    "id": model,
                    "cost": {"input": 7, "output": 11},
                    "modalities": {"output": ["text"]},
                }
            }
        }
    }
    prices = parse_models_dev(payload, 1000)
    harness = {"openai": "codex", "anthropic": "claude", "google": "agy"}.get(provider, "pi")
    value = request(harness, model, None if harness != "pi" else provider)
    amount, identity = value_usage(value, prices)
    assert amount == Decimal("0.00029")
    linked = next(price for price in prices if price.price_id == identity)
    assert linked.source == MODELS_DEV_URL
    assert linked.source_provider == provider and linked.source_model == model
    payload[provider]["models"][model]["cost"]["input"] = 17
    assert value_usage(value, parse_models_dev(payload, 2000))[0] == Decimal("0.00039")
    assert value.model == model and value.provider == (provider if harness == "pi" else None)


def test_openrouter_canonical_snapshot_links_native_codex_without_models_dev():
    payload = {
        "data": [
            {
                "id": "openai/gpt-4.1-mini",
                "canonical_slug": "openai/gpt-4.1-mini-2025-04-14",
                "architecture": {"output_modalities": ["text"]},
                "pricing": {"prompt": "0.000007", "completion": "0.000011"},
            }
        ]
    }
    prices = parse_openrouter(payload, 1000)
    for model in ("gpt-4.1-mini", "gpt-4.1-mini-2025-04-14", "openai/gpt-4.1-mini-2025-04-14"):
        value = request("codex", model)
        amount, identity = value_usage(value, prices)
        assert amount == Decimal("0.00029")
        linked = next(price for price in prices if price.price_id == identity)
        assert linked.source == OPENROUTER_URL
        assert linked.source_model == "openai/gpt-4.1-mini"
    assert value_usage(request("codex", "gpt-4.1-mini-2025-04-15"), prices) == (None, None)


@pytest.mark.parametrize(
    "native,canonical,provider",
    [
        ("gemini-3.1-pro-high", "gemini-3.1-pro-preview", "google"),
        ("gemini-3.8-flash-medium", "gemini-3.8-flash-preview", "google"),
        ("claude-opus-4-6-thinking", "claude-opus-4-6", "anthropic"),
    ],
)
def test_antigravity_effort_variants_keep_source_identity(native, canonical, provider):
    payload = {
        provider: {
            "models": {
                canonical: {
                    "cost": {"input": 7, "output": 11},
                    "modalities": {"output": ["text"]},
                }
            }
        }
    }
    prices = parse_models_dev(payload, 1000)
    value = request("agy", native)
    amount, identity = value_usage(value, prices)
    assert amount == Decimal("0.00029")
    assert next(p for p in prices if p.price_id == identity).source_model == canonical
    assert (
        value_usage(replace(value, source="native:pi", provider="google-antigravity"), prices)[0]
        == amount
    )
    assert value_usage(replace(value, source="gateway", provider="custom"), prices) == (None, None)


def test_claude_catalog_version_syntax_does_not_mix_batch_or_free_variants():
    prices = parse_openrouter(
        {
            "data": [
                {
                    "id": "anthropic/claude-haiku-4.5",
                    "canonical_slug": "anthropic/claude-4.5-haiku-20251001",
                    "architecture": {"output_modalities": ["text"]},
                    "pricing": {"prompt": "0.000007", "completion": "0.000011"},
                },
                {
                    "id": "anthropic/claude-haiku-4.5:batch",
                    "canonical_slug": "anthropic/claude-4.5-haiku-20251001",
                    "architecture": {"output_modalities": ["text"]},
                    "pricing": {"prompt": "0", "completion": "0"},
                },
            ]
        },
        1000,
    )
    value = request("claude", "claude-haiku-4-5-20251001")
    assert value_usage(value, prices)[0] == Decimal("0.00029")
    assert value_usage(replace(value, model="claude-haiku-4-5-20251002"), prices) == (None, None)


def test_pi_additional_catalog_providers_are_not_replaced_by_model_brand():
    prices = parse_models_dev(
        {
            "vendor-cloud": {
                "models": {
                    "claude-test": {
                        "cost": {"input": 7, "output": 11},
                        "modalities": {"output": ["text"]},
                    }
                }
            }
        },
        1000,
    )
    assert value_usage(request("pi", "claude-test", "vendor-cloud"), prices)[0] == Decimal(
        "0.00029"
    )
    assert value_usage(request("pi", "claude-test", "anthropic"), prices) == (None, None)


@pytest.mark.parametrize(
    "harness,provider,model,event",
    [
        (
            "codex",
            "openai",
            "gpt-example",
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "threadId": "n",
                    "tokenUsage": {
                        "total": {
                            "inputTokens": 10,
                            "outputTokens": 20,
                            "cachedInputTokens": 5,
                            "cacheWriteInputTokens": 0,
                            "reasoningOutputTokens": 5,
                            "totalTokens": 30,
                        },
                    },
                },
            },
        ),
        (
            "claude",
            "anthropic",
            "claude-example",
            {
                "type": "assistant",
                "message": {
                    "id": "m",
                    "model": "claude-example",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "cache_read_input_tokens": 5,
                        "cache_creation_input_tokens": 2,
                    },
                },
            },
        ),
        (
            "agy",
            "google",
            "gemini-example-high",
            {
                "event": "result",
                "result": {
                    "conversation_id": "n",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "cache_read_tokens": 5,
                        "thinking_tokens": 5,
                        "total_tokens": 30,
                    },
                },
            },
        ),
        (
            "pi",
            "anthropic",
            "claude-example",
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "timestamp": 1,
                    "model": "claude-example",
                    "provider": "anthropic",
                    "usage": {
                        "input": 10,
                        "output": 20,
                        "cacheRead": 5,
                        "cacheWrite": 2,
                        "totalTokens": 37,
                    },
                },
            },
        ),
    ],
)
def test_actual_native_adapter_links_each_harness_to_public_tariff(harness, provider, model, event):
    canonical = "gemini-example-preview" if harness == "agy" else model
    prices = parse_models_dev(
        {
            provider: {
                "models": {
                    canonical: {
                        "cost": {"input": 7, "output": 11, "cache_read": 1, "cache_write": 7},
                        "modalities": {"output": ["text"]},
                    }
                }
            }
        },
        1000,
    )
    context = NativeUsageContext(
        session_id="s",
        native_id="n",
        harness=harness,
        process_epoch="epoch",
        routing="native",
        observed_model=model,
        model_scope_proven=True,
    )
    (item,) = normalize_native_usage(context, event, observed_at_ms=1)
    value = replace(to_usage_observation(item, sequence=1), kind="delta")
    amount, identity = value_usage(value, prices)
    assert amount is not None and amount > 0
    price = next(price for price in prices if price.price_id == identity)
    assert price.source == MODELS_DEV_URL
    assert price.source_provider == provider
    assert price.source_model == canonical
    assert value.harness == harness and value.model == model
