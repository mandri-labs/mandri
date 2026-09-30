from decimal import Decimal

from mandri.core.types.usage import UsagePrice

REVIEWED_AT = 1000


def synthetic_prices(model: str = "gpt-4.1-mini") -> tuple[UsagePrice, ...]:
    return (
        UsagePrice(
            price_id="synthetic:" + model,
            provider="openai",
            model=model,
            effective_from=0,
            reviewed_at=REVIEWED_AT,
            valuation_basis="current_price_comparison",
            source="https://catalog.example.invalid",
            rates={
                "input_tokens": Decimal("0.4"),
                "cache_read_tokens": Decimal("0.1"),
                "cache_write_tokens": Decimal("0.4"),
                "output_tokens": Decimal("1.6"),
                "reasoning_tokens": Decimal("1.6"),
            },
            constraints={
                "modality": "text",
                "service_tier": "standard",
                "region": "global",
                "modifiers": [],
            },
        ),
    )
