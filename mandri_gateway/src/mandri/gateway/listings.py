"""Bound-model listing payloads in provider-native shapes."""

import json
from typing import Any

from mandri.core.codex_catalog import catalog
from mandri.core.model_metadata import ModelMetadata
from mandri.gateway.reasoning_catalog import ReasoningInfo


def anthropic_listing(model_id: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": "model",
        "id": model_id,
        "display_name": model_id,
    }
    return {
        "data": [entry],
        "has_more": False,
        "first_id": entry["id"],
        "last_id": entry["id"],
    }


def openai_listing(model_id: str) -> dict[str, Any]:
    entry = {
        "id": model_id,
        "object": "model",
        "owned_by": "mandri",
    }
    return {"object": "list", "data": [entry]}


def codex_listing(
    model_id: str, reasoning: ReasoningInfo | None = None, metadata: ModelMetadata | None = None
) -> dict[str, Any]:
    metadata = metadata or ModelMetadata()
    if reasoning is not None:
        metadata = metadata.with_fallback(
            ModelMetadata(
                reasoning_efforts=tuple(reasoning.efforts),
                default_reasoning_effort=reasoning.default_effort,
                reasoning_supported=bool(reasoning.efforts),
            )
        )
    result: dict[str, Any] = json.loads(catalog(model_id, metadata))
    return result


def gemini_listing(model_id: str, metadata: ModelMetadata) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": f"models/{model_id}", "displayName": model_id}
    if metadata.available_context is not None:
        entry["inputTokenLimit"] = metadata.input_tokens or metadata.available_context
    if metadata.output_tokens is not None:
        entry["outputTokenLimit"] = metadata.output_tokens
    return entry
