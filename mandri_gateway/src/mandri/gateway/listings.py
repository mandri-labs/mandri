"""Bound-model listing payloads in provider-native shapes."""

from typing import Any

from mandri.gateway.reasoning_catalog import ReasoningInfo

_DEFAULT_EFFORTS = ("none", "low", "medium", "high")
_EFFORT_DESCRIPTIONS = {
    "none": "No reasoning effort",
    "off": "No reasoning effort",
}
_TRUNCATION_LIMIT_BYTES = 20000


def _effort_description(effort: str) -> str:
    return _EFFORT_DESCRIPTIONS.get(effort, f"Reasoning effort {effort}")


def _reasoning_levels(reasoning: ReasoningInfo | None) -> list[dict[str, str]]:
    efforts = (
        list(reasoning.efforts)
        if reasoning is not None and reasoning.efforts
        else list(_DEFAULT_EFFORTS)
    )
    return [{"effort": effort, "description": _effort_description(effort)} for effort in efforts]


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


def codex_listing(model_id: str, reasoning: ReasoningInfo | None = None) -> dict[str, Any]:
    default_level = (
        reasoning.default_effort if reasoning is not None and reasoning.default_effort else "none"
    )
    entry = {
        "slug": model_id,
        "display_name": model_id,
        "supported_in_api": True,
        "supported_reasoning_levels": _reasoning_levels(reasoning),
        "default_reasoning_level": default_level,
        "shell_type": "unified_exec",
        "visibility": "list",
        "priority": 0,
        "support_verbosity": False,
        "truncation_policy": {"mode": "bytes", "limit": _TRUNCATION_LIMIT_BYTES},
        "experimental_supported_tools": [],
        "input_modalities": ["text", "image"],
    }
    return {"models": [entry], "base_instructions": ""}
