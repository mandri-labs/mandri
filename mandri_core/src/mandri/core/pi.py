import json
from importlib.resources import files
from typing import Any

from mandri.core.model_metadata import ModelMetadata

GATEWAY_PROVIDER = "mandri"
GATEWAY_MODEL_ID = "gateway"


def gateway_extension_path() -> str:
    return str(files("mandri.core").joinpath("resources/pi_gateway.ts"))


def managed_extension_path() -> str:
    return str(files("mandri.core").joinpath("resources/pi_managed.ts"))


def model_entry(model: str, metadata: ModelMetadata | None = None) -> dict[str, Any]:
    capabilities = metadata or ModelMetadata()
    entry: dict[str, Any] = {
        "id": GATEWAY_MODEL_ID,
        "name": model,
    }
    if capabilities.reasoning_supported is not None or capabilities.reasoning_efforts:
        entry["reasoning"] = capabilities.reasoning_supported is not False
    if capabilities.input_modalities is not None:
        inputs = [item for item in capabilities.input_modalities if item in ("text", "image")]
        if capabilities.image_input is False:
            inputs = [item for item in inputs if item != "image"]
        if inputs:
            entry["input"] = inputs
    elif capabilities.image_input is not None:
        entry["input"] = ["text", "image"] if capabilities.image_input else ["text"]
    if capabilities.available_context is not None:
        entry["contextWindow"] = capabilities.available_context
    if capabilities.output_tokens is not None:
        entry["maxTokens"] = capabilities.output_tokens
    if capabilities.reasoning_efforts:
        entry["thinkingLevelMap"] = {
            level: level if level in capabilities.reasoning_efforts else None
            for level in ("minimal", "low", "medium", "high", "xhigh", "max")
        }
        entry["thinkingLevelMap"]["off"] = "none"
    return entry


def gateway_env(
    base_url: str, token: str, model: str, metadata: ModelMetadata | None = None
) -> dict[str, str]:
    return {
        "MANDRI_PI_BASE_URL": base_url,
        "MANDRI_API_KEY": token,
        "MANDRI_PI_MODEL": json.dumps(model_entry(model, metadata)),
    }
