import json

from mandri.core.model_metadata import ModelMetadata

MODEL_ENV = "MANDRI_AGY_MODEL"


def model_config(metadata: ModelMetadata | None) -> dict[str, int]:
    if metadata is None:
        return {}
    return (
        {"maxTokens": metadata.available_context} if metadata.available_context is not None else {}
    )


def model_env(metadata: ModelMetadata | None) -> dict[str, str]:
    config = model_config(metadata)
    return {MODEL_ENV: json.dumps(config)} if config else {}
