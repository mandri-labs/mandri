"""Immutable model value object and provider kind helpers."""

import dataclasses

from mandri.core.ids import ModelRef, ProviderKind, SecretRef, Url

MODEL_REF_PREFIXES: dict[ProviderKind, str] = {
    ProviderKind.OPENROUTER: "openrouter/",
    ProviderKind.OPENCODE: "custom_openai/",
    ProviderKind.OPENCODE_GO: "custom_openai/",
    ProviderKind.OLLAMA: "ollama_chat/",
    ProviderKind.LM_STUDIO: "lm_studio/",
    ProviderKind.OPENAI: "openai/",
    ProviderKind.CHATGPT: "openai/",
    ProviderKind.ANTHROPIC: "anthropic/",
    ProviderKind.GEMINI: "gemini/",
    ProviderKind.CUSTOM: "openai/",
}

_LOCAL_PROVIDERS: frozenset[ProviderKind] = frozenset(
    {ProviderKind.OLLAMA, ProviderKind.LM_STUDIO, ProviderKind.CUSTOM}
)


def requires_api_base(kind: ProviderKind) -> bool:
    """Return True when the provider has no hosted default api_base."""
    return kind in _LOCAL_PROVIDERS


@dataclasses.dataclass(frozen=True)
class Model:
    provider: ProviderKind
    model_ref: ModelRef
    api_base: Url | None
    api_key: SecretRef
