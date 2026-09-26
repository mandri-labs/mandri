import json
from importlib.resources import files

from mandri.core.model_metadata import ModelMetadata

CATALOG_ENV = "MANDRI_CODEX_MODEL_CATALOG"


def catalog(model: str, metadata: "ModelMetadata") -> str:
    limit = max(
        1, min(metadata.context_window * 9 // 10, metadata.context_window - metadata.output_tokens)
    )
    efforts = [
        effort
        for effort in metadata.reasoning_efforts
        if effort in {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
    ]
    entry = {
        "slug": model,
        "display_name": model,
        "description": None,
        "supported_reasoning_levels": [
            {"effort": effort, "description": effort} for effort in efforts
        ],
        "shell_type": "unified_exec",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 1,
        "availability_nux": None,
        "upgrade": None,
        "model_messages": {
            "instructions_template": files("mandri.core")
            .joinpath("resources/codex_prompt_0_154.md")
            .read_text(encoding="utf-8")
        },
        "include_apps_usage_instructions": False,
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": None,
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "context_window": metadata.context_window,
        "max_context_window": metadata.context_window,
        "auto_compact_token_limit": limit,
        "experimental_supported_tools": [],
        "input_modalities": ["text", "image"] if metadata.image_input else ["text"],
        "supports_search_tool": metadata.hosted_web_search,
    }
    return json.dumps({"models": [entry]})
