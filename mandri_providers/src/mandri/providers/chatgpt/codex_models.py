"""Model catalog for the Codex backend behind a ChatGPT subscription."""

from importlib.metadata import version
from typing import Any

MODELS_URL = "https://chatgpt.com/backend-api/codex/models"
CLIENT_VERSION = version("openai-codex-cli-bin")
_FALLBACK_DEFAULT_EFFORT = "medium"


def models_url(api_base: str | None) -> str:
    base = (api_base or MODELS_URL.removesuffix("/models")).rstrip("/")
    return base + "/models"


def catalog_entry(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    slug = entry.get("slug")
    if not isinstance(slug, str) or not slug:
        return None
    result: dict[str, Any] = {
        "id": slug,
        "display_name": entry.get("display_name") or slug,
        "reasoning_efforts": _efforts(entry),
        "default_effort": _default_effort(entry),
        "input_modalities": _modalities(entry),
    }
    return result


def _efforts(entry: dict[str, Any]) -> list[str]:
    levels = entry.get("supported_reasoning_levels")
    if not isinstance(levels, list):
        return []
    efforts: list[str] = []
    for level in levels:
        effort = level.get("effort") if isinstance(level, dict) else None
        if isinstance(effort, str) and effort and effort not in efforts:
            efforts.append(effort)
    return efforts


def _default_effort(entry: dict[str, Any]) -> str | None:
    default = entry.get("default_reasoning_level")
    if isinstance(default, str) and default in _efforts(entry):
        return default
    efforts = _efforts(entry)
    return _FALLBACK_DEFAULT_EFFORT if _FALLBACK_DEFAULT_EFFORT in efforts else None


def _modalities(entry: dict[str, Any]) -> list[str]:
    modalities = entry.get("input_modalities")
    if isinstance(modalities, list):
        values = [item for item in modalities if isinstance(item, str) and item]
        if values:
            return values
    return ["text"]
