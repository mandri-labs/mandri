import json
from copy import deepcopy
from importlib.resources import files

from mandri.core.model_metadata import ModelMetadata

CATALOG_ENV = "MANDRI_CODEX_MODEL_CATALOG"


def catalog(model: str, metadata: "ModelMetadata") -> str:
    native = json.loads(
        files("mandri.core").joinpath("resources/codex_model.json").read_text(encoding="utf-8")
    )
    entry = deepcopy(native["model"])
    entry.update(slug=model, display_name=model, upgrade=None)
    if metadata.available_context is not None:
        entry["context_window"] = metadata.available_context
        entry["max_context_window"] = metadata.available_context
        entry["auto_compact_token_limit"] = metadata.auto_compact_token_limit
    elif metadata.auto_compact_token_limit is not None:
        entry["auto_compact_token_limit"] = metadata.auto_compact_token_limit
    if metadata.reasoning_efforts is not None or metadata.reasoning_supported is False:
        supported = {level["effort"] for level in entry["supported_reasoning_levels"]}
        efforts = [effort for effort in metadata.reasoning_efforts or () if effort in supported]
        if efforts or metadata.reasoning_supported is False:
            entry["supported_reasoning_levels"] = [
                {"effort": effort, "description": effort} for effort in efforts
            ]
            default = metadata.default_reasoning_effort or entry.get("default_reasoning_level")
            entry["default_reasoning_level"] = (
                default if default in efforts else next(iter(efforts), None)
            )
    if metadata.input_modalities is not None:
        inputs = [item for item in metadata.input_modalities if item in entry["input_modalities"]]
        if inputs:
            entry["input_modalities"] = inputs
    if metadata.image_input is False:
        entry["input_modalities"] = [item for item in entry["input_modalities"] if item != "image"]
    elif metadata.image_input is True and "image" not in entry["input_modalities"]:
        entry["input_modalities"].append("image")
    if metadata.hosted_web_search is not None:
        entry["supports_search_tool"] = metadata.hosted_web_search
    return json.dumps({"models": [entry]}, ensure_ascii=False, sort_keys=True)
