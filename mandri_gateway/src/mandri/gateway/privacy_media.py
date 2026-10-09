from collections.abc import Callable
from typing import Any

MEDIA_TYPES = frozenset(
    {
        "image",
        "image_url",
        "input_image",
        "input_audio",
        "audio",
        "video",
        "file",
        "input_file",
        "document",
        "redacted_thinking",
    }
)
MEDIA_FIELDS = frozenset({"inlineData", "inline_data", "fileData", "file_data"})


def media_block(value: dict[str, Any], transform: Callable[[Any], Any]) -> dict[str, Any] | None:
    kind = value.get("type")
    if not (isinstance(kind, str) and kind in MEDIA_TYPES) and not MEDIA_FIELDS.intersection(value):
        return None
    result = dict(value)

    def url(text: Any) -> Any:
        if isinstance(text, str) and text.startswith(("https://", "http://", "gs://", "s3://")):
            changed = transform(text)
            return text if changed is None else changed
        return text

    for key in ("image_url", "file_url", "url"):
        child = value.get(key)
        if isinstance(child, str):
            result[key] = url(child)
        elif key == "image_url" and isinstance(child, dict) and "url" in child:
            result[key] = {**child, "url": url(child["url"])}
    source = value.get("source")
    if isinstance(source, dict) and source.get("type") == "url" and "url" in source:
        result["source"] = {**source, "url": url(source["url"])}
    for key in ("fileData", "file_data"):
        child = value.get(key)
        if isinstance(child, dict):
            result[key] = {
                name: url(item) if name in {"fileUri", "file_uri"} else item
                for name, item in child.items()
            }
    return result
