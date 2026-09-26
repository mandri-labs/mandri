from datetime import UTC, datetime
from decimal import Decimal, localcontext

from mandri.core.types.usage import UsagePrice
from mandri.core.usage_price_schedule import opencode_go_prices

CATALOG_VERSION = "2026-09-20.v2"
CALCULATION_VERSION = "1"
REVIEWED_AT = int(datetime(2026, 9, 20, tzinfo=UTC).timestamp() * 1000)
TOKEN_FIELDS = (
    "input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
)
OPENAI_SOURCE = "https://developers.openai.com/api/docs/models/gpt-4.1-mini"
ANTHROPIC_SOURCE = "https://platform.claude.com/docs/en/models/haiku-4-5/overview"
GEMINI_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-flash-lite"


def _long_rate(key: str, rate: str) -> Decimal:
    with localcontext() as context:
        context.prec = 64
        return Decimal(rate) * (Decimal("1.5") if key in TOKEN_FIELDS[3:] else Decimal(2))


def bundled_prices() -> tuple[UsagePrice, ...]:
    """Return independently mutable, versioned USD token tariffs for startup registration."""
    prices = []
    entries: tuple[
        tuple[str, tuple[str, ...], tuple[str | None, ...], str, dict[str, object]], ...
    ] = (
        (
            "openai",
            ("gpt-4.1-mini", "gpt-4.1-mini-2025-04-14"),
            ("0.40", "0.10", "0.40", "1.60", "1.60"),
            OPENAI_SOURCE,
            {},
        ),
        (
            "anthropic",
            ("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
            ("1", "0.10", "1.25", "5", "5"),
            ANTHROPIC_SOURCE,
            {"cache_write_ttl_seconds": 300},
        ),
        (
            "openai",
            ("gpt-5.3-codex",),
            ("1.75", "0.175", "1.75", "14", "14"),
            "https://developers.openai.com/api/docs/models/gpt-5.3-codex",
            {},
        ),
        (
            "openai",
            ("gpt-6-astra",),
            ("10", "1", "12.5", "50", "50"),
            "https://developers.openai.com/api/docs/models/gpt-6-astra",
            {"max_context_tokens": 272_000},
        ),
        (
            "openai",
            ("gpt-5.6-sol",),
            ("4", "0.4", "5", "20", "20"),
            "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
            {"max_context_tokens": 272_000},
        ),
        (
            "openai",
            ("gpt-5.6-terra",),
            ("2", "0.2", "2.5", "12", "12"),
            "https://developers.openai.com/api/docs/models/gpt-5.6-terra",
            {"max_context_tokens": 272_000},
        ),
        (
            "openai",
            ("gpt-5.6-luna",),
            ("0.2", "0.02", "0.25", "1.2", "1.2"),
            "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
            {"max_context_tokens": 272_000},
        ),
        (
            "anthropic",
            ("claude-sonnet-4-6",),
            ("3", "0.30", "3.75", "15", "15"),
            "https://platform.claude.com/docs/en/models/sonnet-4-6/overview",
            {"cache_write_ttl_seconds": 300},
        ),
        (
            "anthropic",
            ("claude-opus-4-6",),
            ("5", "0.50", "6.25", "25", "25"),
            "https://platform.claude.com/docs/en/models/opus-4-6/overview",
            {"cache_write_ttl_seconds": 300},
        ),
        (
            "anthropic",
            ("claude-sonnet-5",),
            ("2", "0.20", "2.50", "10", "10"),
            "https://platform.claude.com/docs/en/about-claude/pricing",
            {"cache_write_ttl_seconds": 300},
        ),
        (
            "anthropic",
            ("claude-opus-5",),
            ("5", "0.50", "6.25", "25", "25"),
            "https://platform.claude.com/docs/en/about-claude/pricing",
            {"cache_write_ttl_seconds": 300},
        ),
        (
            "gemini",
            ("gemini-2.5-pro",),
            ("1.25", "0.125", None, "10", "10"),
            "https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-pro",
            {"cache_mode": "implicit", "max_context_tokens": 200_000},
        ),
        (
            "gemini",
            ("gemini-3.1-pro-preview", "gemini-3.1-pro-preview-customtools"),
            ("2", "0.20", None, "12", "12"),
            "https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-pro-preview",
            {"cache_mode": "implicit", "max_context_tokens": 200_000},
        ),
        (
            "gemini",
            ("gemini-2.5-flash-lite",),
            ("0.10", "0.01", None, "0.40", "0.40"),
            GEMINI_SOURCE,
            {"cache_mode": "implicit"},
        ),
    )
    for provider, models, rates, source, qualifiers in entries:
        for model in models:
            prices.append(
                UsagePrice(
                    price_id=f"{CATALOG_VERSION}:{provider}:{model}",
                    provider=provider,
                    model=model,
                    effective_from=0,
                    valuation_basis="current_price_comparison",
                    reviewed_at=REVIEWED_AT,
                    source=source,
                    calculation_version=CALCULATION_VERSION,
                    rates={
                        key: Decimal(rate)
                        for key, rate in zip(TOKEN_FIELDS, rates, strict=True)
                        if rate is not None
                    },
                    constraints={
                        "modality": "text",
                        "service_tier": "standard",
                        "region": "global",
                        "modifiers": [],
                        **qualifiers,
                    },
                )
            )
            if qualifiers.get("max_context_tokens") == 272_000:
                prices.append(
                    UsagePrice(
                        price_id=f"{CATALOG_VERSION}:long:{provider}:{model}",
                        provider=provider,
                        model=model,
                        effective_from=0,
                        valuation_basis="current_price_comparison",
                        reviewed_at=REVIEWED_AT,
                        source=source,
                        rates={
                            key: _long_rate(key, rate)
                            for key, rate in zip(TOKEN_FIELDS, rates, strict=True)
                            if rate is not None
                        },
                        constraints={
                            "modality": "text",
                            "service_tier": "standard",
                            "region": "global",
                            "modifiers": [],
                            "min_context_tokens": 272_001,
                        },
                    )
                )
    return (*prices, *opencode_go_prices(REVIEWED_AT))
