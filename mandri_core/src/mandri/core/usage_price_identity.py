from mandri.core.types.usage import UsageObservation

PUBLIC_PRICE_SOURCES = {"https://models.dev/api.json", "https://openrouter.ai/api/v1/models"}

PROVIDER_ALIASES = {
    "google": "gemini",
    "opencode-go": "opencode_go",
    "chatgpt": "openai",
    "openai-codex": "openai",
    "google-gemini-cli": "gemini",
}


def pricing_provider(value: UsageObservation) -> str | None:
    provider = value.pricing_context.get("provider_kind", value.provider)
    if provider is None and value.source.startswith("native:"):
        provider = value.pricing_context.get("observed_provider")
    if isinstance(provider, str):
        if provider == "mandri":
            return None
        if provider in {"google-antigravity", "antigravity"}:
            return "native:agy"
        return PROVIDER_ALIASES.get(provider, provider)
    if (
        value.source.startswith("native:")
        and value.pricing_context.get("comparison_basis") == "standard_text_api"
        and not value.pricing_context.get("observed_provider")
        and value.pricing_context.get("attributed_source") != "gateway"
    ):
        return {
            "native:codex": "openai",
            "native:claude": "anthropic",
            "native:agy": "native:agy",
            "native:pi": "native:pi",
        }.get(value.source)
    return None
