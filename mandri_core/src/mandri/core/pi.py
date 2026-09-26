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
        "reasoning": capabilities.reasoning_supported is True
        or bool(capabilities.reasoning_efforts),
        "input": ["text", "image"] if capabilities.image_input is True else ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": capabilities.context_window,
        "maxTokens": capabilities.output_tokens,
    }
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
