import re
from typing import Any


def normalized_model_id(model_id: str) -> str:
    name = model_id.rpartition("/")[2].lower().removesuffix(".gguf")
    return re.sub(r"[-_:](?:i?q[0-9]+(?:_[a-z0-9]+)*|bf16|fp16|f16|fp32|f32)$", "", name)


def matching_entry(entries: list[dict[str, Any]], model_id: str) -> dict[str, Any] | None:
    exact = [entry for entry in entries if model_id in (entry.get("id"), entry.get("alias_of"))]
    if len(exact) == 1:
        return exact[0]
    matches = [
        entry
        for entry in entries
        if isinstance(entry.get("id"), str)
        and normalized_model_id(entry["id"]) == normalized_model_id(model_id)
    ]
    return matches[0] if len(matches) == 1 else None
