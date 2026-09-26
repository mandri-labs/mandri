from typing import Any


def _multimodal(value: Any) -> bool:
    if isinstance(value, list):
        return any(_multimodal(item) for item in value)
    if not isinstance(value, dict):
        return False
    kind = value.get("type")
    if isinstance(kind, str) and kind in {
        "image",
        "image_url",
        "input_image",
        "input_audio",
        "audio",
        "document",
        "input_file",
        "video",
        "video_url",
    }:
        return True
    if any(key in value for key in ("inlineData", "fileData", "inline_data", "file_data")):
        return True
    return any(_multimodal(item) for item in value.values() if isinstance(item, (dict, list)))


def _text_content(content: Any) -> bool:
    if isinstance(content, str) or content is None:
        return True
    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, dict):
            return False
        kind = part.get("type")
        if kind in {"text", "input_text", "output_text", "refusal", "tool_use", "thinking"}:
            continue
        if kind == "tool_result" and _text_content(part.get("content")):
            continue
        return False
    return True


def request_modality(protocol: str, body: dict[str, Any]) -> str | None:
    if _multimodal(body):
        return "multimodal"
    if body.get("modalities", ["text"]) != ["text"] or body.get("audio") is not None:
        return "multimodal"
    if protocol == "gemini":
        contents = body.get("contents")
        if not isinstance(contents, list) or not contents:
            return None
        return (
            "text"
            if all(
                isinstance(content, dict)
                and isinstance(content.get("parts"), list)
                and all(
                    isinstance(part, dict)
                    and bool(set(part) & {"text", "functionCall", "functionResponse"})
                    for part in content["parts"]
                )
                for content in contents
            )
            else None
        )
    if protocol not in {"chat", "responses", "anthropic"}:
        return None
    tools = body.get("tools", [])
    if not isinstance(tools, list) or any(
        not isinstance(tool, dict) or tool.get("type", "function") != "function" for tool in tools
    ):
        return None
    messages = body.get("input") if protocol == "responses" else body.get("messages")
    if isinstance(messages, str) and protocol == "responses":
        return "text"
    if not isinstance(messages, list) or not messages:
        return None
    for message in messages:
        if not isinstance(message, dict):
            return None
        if message.get("type") == "function_call_output":
            if not _text_content(message.get("output")):
                return None
        elif message.get("type", "message") not in {
            "message",
            "function_call",
            "reasoning",
        } or not _text_content(message.get("content")):
            return None
    return "text"
