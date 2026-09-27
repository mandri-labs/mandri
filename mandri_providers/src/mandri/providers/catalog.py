from typing import Any

from mandri.core.ids import ProviderKind


def model_entries(kind: ProviderKind, payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("provider returned an invalid model catalog")
    key = "models" if kind in (ProviderKind.LM_STUDIO, ProviderKind.GEMINI) else "data"
    entries = payload.get(key)
    if not isinstance(entries, list):
        raise ValueError("provider returned an invalid model catalog")
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
