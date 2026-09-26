import copy
import json
from typing import Any

from mandri.core.types.execution import ProtectionError
from mandri.gateway.surrogate import SurrogateEngine
from mandri.gateway.surrogate.json_text import unique_object

_TEXT_FIELDS = frozenset({"text", "refusal", "thinking", "reasoning", "reasoning_content"})


def response_error(message: str) -> ProtectionError:
    return ProtectionError("privacy_response_invalid", message)


def restore_arguments(value: Any, engine: SurrogateEngine, *, partial: bool = False) -> Any:
    if partial and value == "":
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value, object_pairs_hook=unique_object)
        except (ValueError, RecursionError) as error:
            raise response_error(
                "Provider tool arguments are not complete supported JSON"
            ) from error
        if not isinstance(parsed, dict):
            raise response_error("Provider tool arguments must be a JSON object")
        engine.restore(parsed)
        return engine.restore_text(value)
    if not isinstance(value, dict):
        raise response_error("Provider tool arguments must be structured JSON")
    return engine.restore(value)


def text_value(value: Any, engine: SurrogateEngine) -> Any:
    if value is None:
        return value
    if not isinstance(value, str):
        raise response_error("Provider text content must be a string")
    return engine.restore_text(value)


def restore_function(value: Any, engine: SurrogateEngine, *, partial: bool = False) -> None:
    if not isinstance(value, dict):
        raise response_error("Provider function call is not an object")
    if "name" in value:
        value["name"] = text_value(value["name"], engine)
    for key in ("arguments", "args", "input"):
        if key in value:
            value[key] = restore_arguments(value[key], engine, partial=partial)


def restore_annotations(
    value: Any, engine: SurrogateEngine, original_text: str | None = None
) -> Any:
    if not isinstance(value, list):
        raise response_error("Provider annotations must be a list")
    result = copy.deepcopy(value)
    for annotation in result:
        if not isinstance(annotation, dict):
            raise response_error("Provider annotation must be an object")
        if annotation.get("type") not in (None, "url_citation", "citation", "char_location"):
            continue
        body = annotation.get("url_citation", annotation)
        if not isinstance(body, dict):
            raise response_error("Provider citation must be an object")
        for key in ("url", "title", "cited_text", "document_title"):
            if key in body:
                body[key] = text_value(body[key], engine)
        offsets = [
            key
            for key in ("start_index", "end_index", "start_char_index", "end_char_index")
            if key in body
        ]
        if offsets and original_text is not None:
            for key in offsets:
                offset = body[key]
                if type(offset) is not int or not 0 <= offset <= len(original_text):
                    raise response_error("Provider citation offset is invalid")
                body[key] = engine.restore_offsets(original_text, [offset])[0]
    return result


def restore_part(value: Any, engine: SurrogateEngine, *, partial: bool = False) -> None:
    if not isinstance(value, dict):
        raise response_error("Provider content block must be an object")
    kind = value.get("type")
    if not isinstance(kind, (str, type(None))):
        return
    if kind in {"tool_use", "function_call"}:
        restore_function(value, engine, partial=partial)
        return
    if "functionCall" in value:
        restore_function(value["functionCall"], engine, partial=partial)
        return
    if kind not in {
        None,
        "text",
        "output_text",
        "input_text",
        "summary_text",
        "reasoning_text",
        "thinking",
        "refusal",
    }:
        return
    original_text = value.get("text")
    for key in _TEXT_FIELDS:
        if key in value:
            value[key] = text_value(value[key], engine)
    for key in ("annotations", "citations"):
        if key in value:
            value[key] = restore_annotations(value[key], engine, original_text)


def restore_item(value: Any, engine: SurrogateEngine, *, partial: bool = False) -> None:
    if not isinstance(value, dict):
        raise response_error("Provider output item must be an object")
    original_text = value.get("content") if isinstance(value.get("content"), str) else None
    kind = value.get("type")
    if not isinstance(kind, (str, type(None))):
        return
    if kind == "custom_tool_call":
        for key in ("name", "input"):
            if key in value:
                value[key] = text_value(value[key], engine)
        return
    if kind == "function_call":
        restore_function(value, engine, partial=partial)
        return
    if kind == "web_search_call":
        if "action" in value:
            if not isinstance(value["action"], dict):
                raise response_error("Provider search action must be an object")
            value["action"] = {
                key: child if key == "type" else engine.restore(child)
                for key, child in value["action"].items()
            }
        return
    if kind not in {None, "message", "reasoning"}:
        return
    for key in ("content", "summary", "parts"):
        content = value.get(key)
        if isinstance(content, str) or content is None:
            if key in value:
                value[key] = text_value(content, engine)
        elif isinstance(content, list):
            for part in content:
                restore_part(part, engine, partial=partial)
        else:
            raise response_error("Provider message content has an unsupported shape")
    for key in _TEXT_FIELDS:
        if key in value:
            value[key] = text_value(value[key], engine)
    if value.get("tool_calls") is not None:
        calls = value["tool_calls"]
        if not isinstance(calls, list):
            raise response_error("Provider tool calls must be a list")
        for call in calls:
            if not isinstance(call, dict) or call.get("type", "function") != "function":
                continue
            restore_function(call.get("function"), engine, partial=partial)
    if value.get("function_call") is not None:
        restore_function(value["function_call"], engine, partial=partial)
    if value.get("annotations") is not None:
        value["annotations"] = restore_annotations(value["annotations"], engine, original_text)


def restore_json(payload: Any, engine: SurrogateEngine) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise response_error("Provider response must be an object")
    value = copy.deepcopy(payload)
    if "error" in value:
        error = value["error"]
        if isinstance(error, str):
            value["error"] = text_value(error, engine)
        elif isinstance(error, dict) and "message" in error:
            error["message"] = text_value(error["message"], engine)
    if "response" in value:
        value["response"] = restore_json(value["response"], engine)
        return value
    if "choices" in value:
        if not isinstance(value["choices"], list):
            raise response_error("Provider choices must be a list")
        for choice in value["choices"]:
            if not isinstance(choice, dict):
                raise response_error("Provider choice must be an object")
            if "message" in choice:
                restore_item(choice["message"], engine)
            if "text" in choice:
                choice["text"] = text_value(choice["text"], engine)
        return value
    if "output" in value:
        if not isinstance(value["output"], list):
            raise response_error("Provider output must be a list")
        for item in value["output"]:
            restore_item(item, engine)
        return value
    if "candidates" in value:
        if not isinstance(value["candidates"], list):
            raise response_error("Provider candidates must be a list")
        for candidate in value["candidates"]:
            if not isinstance(candidate, dict):
                raise response_error("Provider candidate must be an object")
            if "content" in candidate:
                restore_item(candidate["content"], engine)
        return value
    if value.get("type") == "message" or "content" in value:
        restore_item(value, engine)
        return value
    if set(value) <= {"input_tokens", "totalTokens", "cachedContentTokenCount", "usage"}:
        return value
    if "error" in value:
        return value
    return value
