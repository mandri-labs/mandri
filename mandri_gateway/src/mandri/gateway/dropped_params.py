"""Detection of request parameters litellm drops for the bound provider."""

from collections.abc import Mapping
from typing import Any

import litellm
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

_NON_PARAMS: frozenset[str] = frozenset({"model", "messages", "stream"})


def _supported_params(model_ref: str) -> frozenset[str] | None:
    try:
        _, provider, _, _ = litellm.get_llm_provider(model=model_ref)
        config = ProviderConfigManager.get_provider_chat_config(
            model=model_ref, provider=LlmProviders(provider)
        )
    except Exception:
        return None
    if config is None or not hasattr(config, "get_supported_openai_params"):
        return None
    return frozenset(config.get_supported_openai_params(model=model_ref))


def dropped_params(model_ref: str, body: Mapping[str, Any]) -> list[str]:
    supported = _supported_params(model_ref)
    if supported is None:
        return []
    return sorted(key for key in body if key not in _NON_PARAMS and key not in supported)
