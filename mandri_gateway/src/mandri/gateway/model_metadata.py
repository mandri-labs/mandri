"""Model capability metadata resolved from provider model catalogs."""

from typing import Any

import httpx
from mandri.core.ids import ProviderKind, Url
from mandri.core.model_metadata import ModelMetadata as ModelMetadata
from mandri.gateway.catalog_enrichment import enrich_entries
from mandri.gateway.model_capabilities import metadata_capabilities
from mandri.providers.catalog import model_entries
from mandri.providers.refs import MODEL_REF_PREFIXES
from mandri.providers.verify import models_endpoint, models_headers

_TIMEOUT_SECONDS = 10.0


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
            entry.get("context_length", entry.get("max_context_length", limits.get("context"))),
            ModelMetadata().context_window,
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
    try:
        entries = model_entries(kind, payload)
    except ValueError:
        return None
    for entry in entries:
        if entry["id"] != model_id:
            continue
        if kind is ProviderKind.OPENROUTER:
            return _openrouter_entry(entry)
        return _openai_compatible_entry(entry)
    return None


def _bare_id(kind: ProviderKind, model_ref: str) -> str:
    return model_ref.removeprefix(MODEL_REF_PREFIXES[kind])


async def fetch(
    kind: ProviderKind, model_ref: str, api_base: Url | None, api_key: str
) -> ModelMetadata | None:
    if api_base is None and kind in (
        ProviderKind.LM_STUDIO,
        ProviderKind.OLLAMA,
        ProviderKind.CUSTOM,
    ):
        return None
    headers = models_headers(kind, api_key)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.get(models_endpoint(kind, api_base), headers=headers)
            response.raise_for_status()
            payload = response.json()
            entries = await enrich_entries(kind, model_entries(kind, payload), client)
    except (httpx.HTTPError, ValueError):
        return None
    for entry in entries:
        if entry["id"] == _bare_id(kind, model_ref):
            return (
                _openrouter_entry(entry)
                if kind is ProviderKind.OPENROUTER
                else _openai_compatible_entry(entry)
            )
    return None
