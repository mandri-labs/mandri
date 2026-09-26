from typing import Any


def normalize_contents(contents: Any) -> Any:
    if not isinstance(contents, list):
        return contents
    normalized: list[Any] = []
    for content in contents:
        if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
            normalized.append(content)
            continue
        blocks: list[dict[str, Any]] = []
        for part in content["parts"]:
            role = (
                "user"
                if isinstance(part, dict) and "functionResponse" in part
                else content.get("role", "user")
            )
            if not blocks or blocks[-1]["role"] != role:
                blocks.append({**content, "role": role, "parts": []})
            blocks[-1]["parts"].append(part)
        normalized.extend(blocks or [content])
    return normalized
