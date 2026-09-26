"""Reasoning effort catalog per provider model, resolved at daemon startup."""

import dataclasses
import logging
from collections.abc import Callable
from typing import Any

import httpx
from mandri.core.ids import ProviderKind
from mandri.gateway.model_metadata import openrouter_reasoning_efforts
from mandri.providers.refs import MODEL_REF_PREFIXES
from mandri.providers.service import Provider, ProvidersRegistry

_TIMEOUT_SECONDS = 10.0
_MODELS_DEV_URL = "https://models.dev/api.json"
_LM_STUDIO_MODELS_PATH = "/api/v1/models"
_OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api"
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_LOCAL_KINDS = frozenset({ProviderKind.OLLAMA, ProviderKind.LM_STUDIO, ProviderKind.CUSTOM})
_MODELS_DEV_IDS: dict[ProviderKind, str] = {
    ProviderKind.OPENROUTER: "openrouter",
    ProviderKind.OPENAI: "openai",
    ProviderKind.ANTHROPIC: "anthropic",
    ProviderKind.GEMINI: "google",
    ProviderKind.LM_STUDIO: "lmstudio",
    ProviderKind.OLLAMA: "ollama-cloud",
    ProviderKind.OPENCODE_GO: "opencode-go",
    ProviderKind.OPENCODE: "opencode",
}
logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ReasoningInfo:
    efforts: list[str]
    default_effort: str | None = None


def parse_models_dev_entry(entry: Any) -> ReasoningInfo | None:
    if not isinstance(entry, dict):
        return None
    options = entry.get("reasoning_options")
    if isinstance(options, dict):
        options = [options]
    if not isinstance(options, list):
        return None
    efforts: list[str] = []
    for option in options:
        if not isinstance(option, dict) or option.get("type") != "effort":
            continue
        values = option.get("values")
        if isinstance(values, list):
            efforts.extend(value for value in values if isinstance(value, str))
    return _normalize(efforts)


def parse_lm_studio_entry(entry: Any) -> ReasoningInfo | None:
    if not isinstance(entry, dict):
        return None
    capabilities = entry.get("capabilities")
    reasoning = capabilities.get("reasoning") if isinstance(capabilities, dict) else None
    if not isinstance(reasoning, dict) or not isinstance(reasoning.get("allowed_options"), list):
        return None
    info = _normalize([value for value in reasoning["allowed_options"] if isinstance(value, str)])
    if info is None:
        return None
    default = reasoning.get("default")
    if isinstance(default, str) and default in info.efforts:
        return ReasoningInfo(efforts=info.efforts, default_effort=default)
    return info


def _normalize(efforts: list[str]) -> ReasoningInfo | None:
    unique: list[str] = []
    for effort in efforts:
        if effort not in unique:
            unique.append(effort)
    if not unique:
        return None
    return ReasoningInfo(efforts=unique)


def _bare_id(model_ref: str) -> str:
    for prefix in MODEL_REF_PREFIXES.values():
        if model_ref.startswith(prefix):
            return model_ref[len(prefix) :]
    return model_ref


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


async def _fetch_json(url: str, headers: dict[str, str]) -> Any:
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
    except (httpx.HTTPError, ValueError) as error:
        logger.warning("reasoning catalog fetch failed for %s: %s", url, error)
        return None


async def _openrouter_live(base: str, api_key: str) -> dict[str, ReasoningInfo]:
    payload = await _fetch_json(base.rstrip("/") + "/v1/models", _auth_headers(api_key))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return {}
    catalog: dict[str, ReasoningInfo] = {}
    for entry in payload["data"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        efforts = list(openrouter_reasoning_efforts(entry))
        info = _normalize(efforts)
        if info is not None:
            catalog[entry["id"]] = info
    return catalog


async def _models_dev_catalog() -> dict[str, dict[str, ReasoningInfo]]:
    payload = await _fetch_json(_MODELS_DEV_URL, {})
    if not isinstance(payload, dict):
        return {}
    catalog: dict[str, dict[str, ReasoningInfo]] = {}
    for provider_id, provider in payload.items():
        if not isinstance(provider, dict):
            continue
        models = provider.get("models")
        if not isinstance(models, dict):
            continue
        parsed = {
            model_id: info
            for model_id, entry in models.items()
            if (info := parse_models_dev_entry(entry)) is not None
        }
        if parsed:
            catalog[provider_id] = parsed
    return catalog


async def _lm_studio_probe(base: str, api_key: str) -> dict[str, ReasoningInfo]:
    payload = await _fetch_json(base.rstrip("/") + _LM_STUDIO_MODELS_PATH, _auth_headers(api_key))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return {}
    catalog: dict[str, ReasoningInfo] = {}
    for entry in payload["models"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        info = parse_lm_studio_entry(entry)
        if info is not None:
            catalog[entry["id"]] = info
    return catalog


def _is_local_base(base: str) -> bool:
    if not base:
        return False
    try:
        return httpx.URL(base).host in _LOCAL_HOSTS
    except ValueError:
        return False


def _models_dev_provider(
    provider: Provider, models_dev: dict[str, dict[str, ReasoningInfo]]
) -> dict[str, ReasoningInfo]:
    key = _MODELS_DEV_IDS.get(provider.kind) or provider.name.lower()
    return models_dev.get(key, {})


def _merge(
    entries: dict[tuple[str, str], ReasoningInfo],
    provider_name: str,
    catalog: dict[str, ReasoningInfo],
) -> None:
    for model_id, info in catalog.items():
        if info.efforts:
            entries[(provider_name, model_id)] = info


class ReasoningCatalog:
    def __init__(self, entries: dict[tuple[str, str], ReasoningInfo] | None = None) -> None:
        self._entries = dict(entries or {})

    @classmethod
    async def build(cls, list_providers: Callable[[], list[Provider]]) -> "ReasoningCatalog":
        entries: dict[tuple[str, str], ReasoningInfo] = {}
        models_dev = await _models_dev_catalog()
        for provider in list_providers():
            await _merge_provider(entries, provider, models_dev)
        return cls(entries)

    def lookup(self, provider_name: str, model_ref: str) -> ReasoningInfo | None:
        prefix = provider_name + "/"
        bare = model_ref[len(prefix) :] if model_ref.startswith(prefix) else _bare_id(model_ref)
        info = self._entries.get((provider_name, bare))
        if info is not None:
            return info
        tail = bare.rpartition("/")[2]
        if tail and tail != bare:
            return self._entries.get((provider_name, tail))
        return None


async def _merge_provider(
    entries: dict[tuple[str, str], ReasoningInfo],
    provider: Provider,
    models_dev: dict[str, dict[str, ReasoningInfo]],
) -> None:
    base = str(provider.api_base) if provider.api_base else ""
    if provider.kind in _LOCAL_KINDS and _is_local_base(base):
        _merge(entries, provider.name, await _lm_studio_probe(base, str(provider.api_key)))
    _merge(entries, provider.name, _models_dev_provider(provider, models_dev))
    if provider.kind is ProviderKind.OPENROUTER:
        openrouter_base = base or _OPENROUTER_DEFAULT_BASE
        _merge(
            entries,
            provider.name,
            await _openrouter_live(openrouter_base, str(provider.api_key)),
        )


async def build_reasoning_catalog(providers: ProvidersRegistry) -> ReasoningCatalog:
    return await ReasoningCatalog.build(providers.list)
