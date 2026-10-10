import hashlib
import logging
import shlex
from collections import OrderedDict
from typing import Any

import httpx
from mandri.core.ids import ProviderKind, Url
from mandri.core.model_metadata import ModelMetadata as ModelMetadata
from mandri.gateway.catalog_enrichment import fallback_entries
from mandri.gateway.metadata_entry import consensus, parse_entry
from mandri.gateway.model_matching import matching_entry
from mandri.providers.catalog import model_entries
from mandri.providers.refs import MODEL_REF_PREFIXES
from mandri.providers.verify import lm_studio_base, models_endpoint, models_headers, models_params

_TIMEOUT_SECONDS = 10.0
_previous: OrderedDict[tuple[ProviderKind, str, str, str], ModelMetadata] = OrderedDict()
logger = logging.getLogger(__name__)


def _parse(kind: ProviderKind, payload: Any, model_id: str) -> ModelMetadata | None:
    try:
        entry = matching_entry(model_entries(kind, payload), model_id)
    except ValueError:
        entry = None
    return parse_entry(entry) if entry is not None else None


async def fetch_json(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    params: dict[str, str] | None = None,
    body: dict[str, str] | None = None,
) -> Any:
    try:
        response = (
            await client.get(url, headers=headers, params=params)
            if body is None
            else await client.post(url, headers=headers, json=body)
        )
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as error:
        logger.debug("Model discovery unavailable: %s", type(error).__name__)
        return None


async def supplementary_entry(
    client: httpx.AsyncClient,
    kind: ProviderKind,
    base: Url | None,
    model_id: str,
    headers: dict[str, str],
) -> dict[str, Any] | None:
    if base is None:
        return None
    root = lm_studio_base(str(base))
    if kind is ProviderKind.OLLAMA:
        payload = await fetch_json(client, root + "/api/show", headers, body={"model": model_id})
        result = {**payload, "id": model_id} if isinstance(payload, dict) else {"id": model_id}
        parameters = result.get("parameters")
        if isinstance(parameters, str):
            parsed = {}
            for line in parameters.splitlines():
                try:
                    parts = shlex.split(line)
                    if len(parts) == 2:
                        parsed[parts[0]] = int(parts[1])
                except ValueError:
                    continue
            result["parameters"] = parsed
        loaded = await fetch_json(client, root + "/api/ps", headers)
        entries = loaded.get("models") if isinstance(loaded, dict) else None
        active = (
            matching_entry(
                [
                    {**entry, "id": entry.get("model", entry.get("name"))}
                    for entry in entries
                    if isinstance(entry, dict)
                ],
                model_id,
            )
            if isinstance(entries, list)
            else None
        )
        if active is not None:
            result = {**result, **active, "active_context_length": active.get("context_length")}
        return result
    if kind is ProviderKind.CUSTOM:
        payload = await fetch_json(client, root + "/props", headers)
        if isinstance(payload, dict) and model_id in (
            payload.get("model_alias"),
            payload.get("id"),
            payload.get("model_path"),
        ):
            return payload
    return None


async def fetch(
    kind: ProviderKind, model_ref: str, api_base: Url | None, api_key: str
) -> ModelMetadata:
    model_id = model_ref.removeprefix(MODEL_REF_PREFIXES[kind])
    identity = (kind, str(api_base or ""), model_id, hashlib.sha256(api_key.encode()).hexdigest())
    metadata = ModelMetadata()
    headers = models_headers(kind, api_key)
    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
        if api_base is not None or kind not in {
            ProviderKind.CUSTOM,
            ProviderKind.LM_STUDIO,
            ProviderKind.OLLAMA,
        }:
            payload = await fetch_json(
                client, models_endpoint(kind, api_base), headers, models_params(kind)
            )
            metadata = _parse(kind, payload, model_id) or metadata
        if (
            metadata.context_window is None
            or metadata.reasoning_efforts is None
            or metadata.output_tokens is None
        ):
            extra = await supplementary_entry(client, kind, api_base, model_id, headers)
            if extra is not None:
                metadata = parse_entry(extra, "server").with_fallback(metadata)
        cached = _previous.get(identity)
        if cached is not None:
            metadata = metadata.with_fallback(cached)
        entries = await fallback_entries(kind, model_id, client)
        metadata = metadata.with_fallback(consensus([parse_entry(entry) for entry in entries]))
    _previous[identity] = metadata
    _previous.move_to_end(identity)
    if len(_previous) > 1024:
        _previous.popitem(last=False)
    logger.debug("Model metadata resolved for %s: sources=%s", model_id, metadata.sources)
    return metadata
