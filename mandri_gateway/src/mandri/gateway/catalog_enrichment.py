import time
from typing import Any

import httpx
from mandri.core.ids import ProviderKind
from mandri.gateway.model_matching import matching_entry

_PROVIDERS = {
    ProviderKind.OPENCODE: "opencode",
    ProviderKind.OPENCODE_GO: "opencode-go",
    ProviderKind.OPENAI: "openai",
    ProviderKind.ANTHROPIC: "anthropic",
    ProviderKind.GEMINI: "google",
    ProviderKind.OPENROUTER: "openrouter",
}
_CATALOG_URL = "https://models.dev/api.json"
_cached: dict[str, Any] = {}
_expires = 0.0


async def enrich_entries(
    kind: ProviderKind, entries: list[dict[str, Any]], client: httpx.AsyncClient
) -> list[dict[str, Any]]:
    provider = _PROVIDERS.get(kind)
    if provider is None or not entries:
        return entries
    await refresh_catalog(client)
    catalog = _cached.get(provider, {})
    models = catalog.get("models", {}) if isinstance(catalog, dict) else {}
    if not isinstance(models, dict):
        return entries
    return [
        merge_entry(models[entry["id"]], entry)
        if isinstance(models.get(entry["id"]), dict)
        else entry
        for entry in entries
    ]


def merge_entry(fallback: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    result = dict(fallback)
    for key, value in live.items():
        if value is None:
            continue
        previous = result.get(key)
        result[key] = (
            merge_entry(previous, value)
            if isinstance(previous, dict) and isinstance(value, dict)
            else value
        )
    return result


async def refresh_catalog(client: httpx.AsyncClient) -> None:
    global _cached, _expires
    if time.monotonic() >= _expires:
        try:
            response = await client.get(_CATALOG_URL, timeout=3.0)
            if response.status_code == 200:
                payload = response.json()
                if isinstance(payload, dict):
                    _cached = payload
        except (httpx.HTTPError, ValueError):
            pass
        _expires = time.monotonic() + 300


async def fallback_entries(
    kind: ProviderKind, model_id: str, client: httpx.AsyncClient
) -> list[dict[str, Any]]:
    await refresh_catalog(client)
    provider = _PROVIDERS.get(kind)
    catalogs = [_cached.get(provider, {})] if provider else list(_cached.values())
    result = []
    for catalog in catalogs:
        models = catalog.get("models") if isinstance(catalog, dict) else None
        if not isinstance(models, dict):
            continue
        entries = [{**entry, "id": key} for key, entry in models.items() if isinstance(entry, dict)]
        entry = matching_entry(entries, model_id)
        if entry is not None:
            result.append(entry)
    return result
