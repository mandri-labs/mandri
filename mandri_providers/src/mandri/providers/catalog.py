from typing import Any

from mandri.core.ids import ProviderKind
from mandri.providers.chatgpt.codex_models import catalog_entry

_MODEL_LIST_KEYS: dict[ProviderKind, str] = {
    ProviderKind.LM_STUDIO: "models",
    ProviderKind.GEMINI: "models",
    ProviderKind.CHATGPT: "models",
}


def model_entries(kind: ProviderKind, payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("provider returned an invalid model catalog")
    entries = payload.get(_MODEL_LIST_KEYS.get(kind, "data"))
    if not isinstance(entries, list):
        raise ValueError("provider returned an invalid model catalog")
    if kind is ProviderKind.CHATGPT:
        return [entry for raw in entries if (entry := catalog_entry(raw)) is not None]
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if kind is ProviderKind.LM_STUDIO:
            if entry.get("type") == "embedding":
                continue
            model_id = entry.get("key", entry.get("id"))
        elif kind is ProviderKind.GEMINI:
            name = entry.get("name")
            model_id = name.removeprefix("models/") if isinstance(name, str) else None
        else:
            model_id = entry.get("id")
        if isinstance(model_id, str) and model_id:
            result.append({**entry, "id": model_id})
    return result
