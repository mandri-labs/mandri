import time
from typing import Any

import httpx
from mandri.core.ids import ProviderKind

_PROVIDERS = {
    ProviderKind.OPENCODE: "opencode",
    ProviderKind.OPENCODE_GO: "opencode-go",
    ProviderKind.OPENAI: "openai",
    ProviderKind.ANTHROPIC: "anthropic",
    ProviderKind.GEMINI: "google",
}
_CATALOG_URL = "https://models.dev/api.json"
_cached: dict[str, Any] = {}
_expires = 0.0


async def enrich_entries(
    kind: ProviderKind, entries: list[dict[str, Any]], client: httpx.AsyncClient
) -> list[dict[str, Any]]:
    global _cached, _expires
    provider = _PROVIDERS.get(kind)
    if provider is None or not entries:
        return entries
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
    catalog = _cached.get(provider, {})
    models = catalog.get("models", {}) if isinstance(catalog, dict) else {}
    if not isinstance(models, dict):
        return entries
    return [
        {**models[entry["id"]], **entry} if isinstance(models.get(entry["id"]), dict) else entry
        for entry in entries
    ]
