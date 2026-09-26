"""Model reference prefixes and api_base requirements per provider kind."""

from mandri.core.ids import ProviderKind
from mandri.core.model_refs import MODEL_REF_PREFIXES as MODEL_REF_PREFIXES

_LOCAL_PROVIDERS: frozenset[ProviderKind] = frozenset(
    {ProviderKind.OLLAMA, ProviderKind.LM_STUDIO, ProviderKind.CUSTOM}
)


def requires_api_base(kind: ProviderKind) -> bool:
    """Return True when the provider has no hosted default api_base."""
    return kind in _LOCAL_PROVIDERS
