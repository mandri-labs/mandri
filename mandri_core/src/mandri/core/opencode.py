from typing import Any

from mandri.core.model_metadata import ModelMetadata

GATEWAY_MODEL_ID = "mandri_gateway"
GATEWAY_MODEL_REF = f"mandri/{GATEWAY_MODEL_ID}"
_MODALITIES = ("text", "image", "audio", "video", "pdf")


def gateway_model() -> dict[str, str]:
    return {"providerID": "mandri", "id": GATEWAY_MODEL_ID}


def model_entry(metadata: ModelMetadata | None) -> dict[str, Any]:
    metadata = metadata or ModelMetadata()
    inputs = list(
        metadata.input_modalities if metadata.input_modalities is not None else _MODALITIES
    )
    if metadata.image_input is False and "image" in inputs:
        inputs.remove("image")
    reasoning = metadata.reasoning_supported is not False
    entry: dict[str, Any] = {
        "name": "Mandri Gateway",
        "reasoning": reasoning,
        "attachment": metadata.attachment is not False,
        "tool_call": metadata.tool_call is not False,
        "temperature": metadata.temperature is not False,
        "interleaved": {"field": "reasoning_content"} if reasoning else False,
        "modalities": {
            "input": inputs,
            "output": list(
                metadata.output_modalities
                if metadata.output_modalities is not None
                else _MODALITIES
            ),
        },
    }
    limits = {
        key: value
        for key, value in (
            ("context", metadata.available_context),
            ("input", metadata.input_tokens),
            ("output", metadata.output_tokens),
        )
        if value is not None
    }
    if limits:
        entry["limit"] = limits
    if reasoning and metadata.reasoning_efforts:
        entry["variants"] = {
            effort: {"reasoningEffort": effort} for effort in metadata.reasoning_efforts
        }
    return entry


def provider_entry(route_base: str, token: str, metadata: ModelMetadata | None) -> dict[str, Any]:
    legacy = model_entry(metadata)
    entry: dict[str, Any] = {
        "name": legacy["name"],
        "capabilities": {
            "tools": legacy["tool_call"],
            "input": legacy["modalities"]["input"],
            "output": legacy["modalities"]["output"],
        },
    }
    if "limit" in legacy:
        entry["limit"] = legacy["limit"]
    if "variants" in legacy:
        entry["variants"] = [
            {"id": name, "settings": {"reasoningEffort": name}} for name in legacy["variants"]
        ]
    if legacy["reasoning"]:
        entry["compatibility"] = {"reasoningField": "reasoning_content"}
    return {
        "name": "Mandri Gateway",
        "package": "@opencode/ai/providers/openai-compatible",
        "settings": {"baseURL": route_base, "apiKey": token},
        "models": {GATEWAY_MODEL_ID: entry},
    }
