from typing import Any

import litellm
from litellm.llms.ollama.chat.transformation import OllamaChatConfig
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openrouter.chat.transformation import OpenrouterConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def apply_reasoning_transport(
    model_ref: str, payload: dict[str, Any], *, responses: bool, native_responses: bool = False
) -> None:
    reasoning = payload.get("reasoning")
    effort = (
        reasoning.get("effort")
        if responses and isinstance(reasoning, dict)
        else payload.get("reasoning_effort")
    )
    if not isinstance(effort, str) or not effort:
        return
    model, provider, _, _ = litellm.get_llm_provider(model=model_ref)
    config = ProviderConfigManager.get_provider_chat_config(model, LlmProviders(provider))
    extra: dict[str, Any]
    if isinstance(config, OllamaChatConfig):
        payload.pop("reasoning" if responses else "reasoning_effort", None)
        payload["think"] = {"on": True, "off": False}.get(effort, effort)
        return
    if isinstance(config, OpenrouterConfig):
        if native_responses and effort not in {"on", "off"}:
            return
        extra = {
            "reasoning": (
                {"enabled": effort == "on"} if effort in {"on", "off"} else {"effort": effort}
            )
        }
    elif isinstance(config, OpenAIGPTConfig) and not native_responses:
        extra = {"reasoning_effort": {"off": "none", "on": "medium"}.get(effort, effort)}
    else:
        return
    payload.pop("reasoning" if responses else "reasoning_effort", None)
    payload["extra_body"] = {**payload.get("extra_body", {}), **extra}
