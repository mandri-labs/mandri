"""Model capability metadata resolved from provider model catalogs."""

from typing import Any

import httpx
from mandri.core.ids import ProviderKind, Url
from mandri.core.model_metadata import ModelMetadata as ModelMetadata
from mandri.gateway.model_capabilities import metadata_capabilities

_TIMEOUT_SECONDS = 10.0
_DEFAULT_BASES: dict[ProviderKind, str] = {
    ProviderKind.OPENROUTER: "https://openrouter.ai/api",
    ProviderKind.OPENCODE: "https://opencode.ai/zen/v1",
    ProviderKind.OPENCODE_GO: "https://opencode.ai/zen/go/v1",
    ProviderKind.OPENAI: "https://api.openai.com/v1",
}
_MODELS_PATHS: dict[ProviderKind, str] = {
    ProviderKind.OPENROUTER: "/v1/models",
    ProviderKind.OPENCODE: "/models",
    ProviderKind.OPENCODE_GO: "/models",
    ProviderKind.OPENAI: "/models",
}
_MODEL_REF_PREFIXES: dict[ProviderKind, str] = {
    ProviderKind.OPENROUTER: "openrouter/",
    ProviderKind.OPENCODE: "custom_openai/",
    ProviderKind.OPENCODE_GO: "custom_openai/",
    ProviderKind.OPENAI: "openai/",
}


def openrouter_reasoning_efforts(entry: dict[str, Any]) -> tuple[str, ...]:
    reasoning = entry.get("reasoning")
    if isinstance(reasoning, dict) and isinstance(reasoning.get("supported_efforts"), list):
        return tuple(
            str(effort) for effort in reasoning["supported_efforts"] if isinstance(effort, str)
        )
    return ()


def _openrouter_entry(entry: dict[str, Any]) -> ModelMetadata:
    context = entry.get("context_length")
    provider = entry.get("top_provider")
    if isinstance(provider, dict) and isinstance(provider.get("context_length"), int):
        context = provider["context_length"]
    output = provider.get("max_completion_tokens") if isinstance(provider, dict) else None
    efforts = openrouter_reasoning_efforts(entry)
    return ModelMetadata(
        context_window=_limit(context, ModelMetadata().context_window),
        output_tokens=_limit(output, ModelMetadata().output_tokens),
        reasoning_efforts=efforts,
        hosted_web_search=_hosted_web_search(entry),
        **metadata_capabilities(entry),
    )


def _openai_compatible_entry(entry: dict[str, Any]) -> ModelMetadata:
    limits = entry.get("limit")
    limits = limits if isinstance(limits, dict) else {}
    return ModelMetadata(
        context_window=_limit(
            entry.get("context_length", limits.get("context")), ModelMetadata().context_window
        ),
        output_tokens=_limit(
            entry.get("max_output_tokens", limits.get("output")), ModelMetadata().output_tokens
        ),
        reasoning_efforts=openrouter_reasoning_efforts(entry),
        hosted_web_search=_hosted_web_search(entry),
        **metadata_capabilities(entry),
    )


def _limit(value: Any, default: int) -> int:
    return value if type(value) is int and value > 0 else default


def _hosted_web_search(entry: dict[str, Any]) -> bool:
    capabilities = entry.get("capabilities")
    if isinstance(capabilities, dict) and "hosted_web_search" in capabilities:
        return capabilities["hosted_web_search"] is True
    supported = entry.get("supported_parameters")
    return isinstance(supported, list) and "web_search_options" in supported


def _parse(kind: ProviderKind, payload: Any, model_id: str) -> ModelMetadata | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return None
    for entry in payload["data"]:
        if not isinstance(entry, dict) or entry.get("id") != model_id:
            continue
        if kind is ProviderKind.OPENROUTER:
            return _openrouter_entry(entry)
        return _openai_compatible_entry(entry)
    return None


def _bare_id(model_ref: str) -> str:
    for prefix in _MODEL_REF_PREFIXES.values():
        if model_ref.startswith(prefix):
            return model_ref[len(prefix) :]
    return model_ref


async def fetch(
    kind: ProviderKind, model_ref: str, api_base: Url | None, api_key: str
) -> ModelMetadata | None:
    path = _MODELS_PATHS.get(kind)
    if path is None:
        return None
    base = str(api_base) if api_base is not None else _DEFAULT_BASES[kind]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.get(base.rstrip("/") + path, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    return _parse(kind, payload, _bare_id(model_ref))
