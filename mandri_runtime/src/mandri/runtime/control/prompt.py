import base64
from typing import Any

from mandri.core.types.prompt import UserPrompt, prompt_text


def native_parts(content: str | UserPrompt, harness: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt_text(content)}]
    if isinstance(content, str):
        return parts
    for attachment in content.attachments:
        if not attachment.media_type.startswith("image/"):
            continue
        if harness == "codex":
            parts.append({"type": "localImage", "path": attachment.path})
        elif harness == "claude":
            parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": attachment.media_type,
                        "data": base64.b64encode(attachment.data).decode("ascii"),
                    },
                }
            )
        elif harness == "opencode":
            data = base64.b64encode(attachment.data).decode("ascii")
            parts.append(
                {
                    "type": "file",
                    "mime": attachment.media_type,
                    "url": f"data:{attachment.media_type};base64,{data}",
                }
            )
    return parts
