from collections.abc import Callable
from typing import Any

from mandri.gateway.privacy_context_edit import context_edit

_SCHEMA_MAPS = frozenset(
    {"properties", "$defs", "definitions", "patternProperties", "dependentSchemas"}
)
_SCHEMA_CHILDREN = frozenset(
    {
        "items",
        "additionalProperties",
        "unevaluatedProperties",
        "propertyNames",
        "contains",
        "not",
        "if",
        "then",
        "else",
        "unevaluatedItems",
    }
)
_SCHEMA_LISTS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SCHEMA_DATA = frozenset({"default", "const", "enum", "examples", "example"})
_SCHEMA_TEXT = frozenset({"description", "title", "$comment", "$id"})


class ContentVisitor:
    def __init__(self, transform: Callable[[Any], Any]) -> None:
        self.transform = transform

    def data(self, value: Any) -> Any:
        result = self.transform(value)
        return value if result is None and value is not None else result

    def root(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return self.data(value)
        result = dict(value)
        for key, child in value.items():
            if key == "format" and isinstance(child, dict):
                result[key] = self.schema(child)
            elif key in {"messages", "contents"}:
                result[key] = self.messages(child)
            elif key == "input":
                result[key] = self.messages(child) if isinstance(child, list) else self.data(child)
            elif key in {"system", "systemInstruction", "system_instruction"}:
                result[key] = (
                    self.message(child)
                    if isinstance(child, dict) and "parts" in child
                    else self.blocks(child)
                )
            elif key in {"tools", "functions"}:
                if isinstance(child, list):
                    result[key] = [self.tool(item) for item in child]
            elif key in {"tool_choice", "function_call", "toolConfig", "tool_config"}:
                result[key] = self.choice(child)
            elif key in {"response_format", "text"} and isinstance(child, dict):
                result[key] = self.output_format(child)
            elif key in {
                "thinking",
                "reasoning",
                "output_config",
                "generationConfig",
                "generation_config",
                "stream_options",
                "options",
            }:
                result[key] = self.settings(child)
            elif key == "context_management":
                result[key] = context_edit(child, self.data)
            elif key != "model":
                result[key] = self.data(child)
        return result

    def messages(self, value: Any) -> Any:
        return [self.message(item) for item in value] if isinstance(value, list) else value

    def message(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.data(value)
        if not isinstance(value, dict):
            return value
        if value.get("type") not in (None, "message"):
            return self.block(value)
        result = dict(value)
        for key, child in value.items():
            if key in {"content", "parts"}:
                result[key] = self.blocks(child)
            elif key == "tool_calls" and isinstance(child, list):
                result[key] = [self.call(item) for item in child]
            elif key == "function_call":
                result[key] = self.function(child)
            elif key in {"name", "refusal", "reasoning_content"}:
                result[key] = self.data(child)
        return result

    def blocks(self, value: Any) -> Any:
        return [self.block(item) for item in value] if isinstance(value, list) else self.data(value)

    def block(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.data(value)
        if not isinstance(value, dict):
            return value
        kind = value.get("type")
        if not isinstance(kind, (str, type(None))):
            return value
        if kind not in {
            None,
            "text",
            "input_text",
            "output_text",
            "summary_text",
            "reasoning_text",
            "reasoning",
            "thinking",
            "refusal",
            "tool_use",
            "tool_result",
            "function_call",
            "function_call_output",
            "custom_tool_call",
            "custom_tool_call_output",
            "web_search_call",
            "function",
        }:
            return (
                value
                if kind
                in {
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
                else self.data(value)
            )
        result = dict(value)
        for key, child in value.items():
            if key in {"functionCall", "functionResponse"}:
                result[key] = self.function(child)
            elif key in {"content", "summary"}:
                result[key] = self.blocks(child)
            elif key == "output" and kind in {"function_call_output", "custom_tool_call_output"}:
                result[key] = self.tool_output(child)
            elif key == "action" and kind == "web_search_call" and isinstance(child, dict):
                result[key] = {
                    name: item if name == "type" else self.data(item)
                    for name, item in child.items()
                }
            elif key in {
                "text",
                "thinking",
                "refusal",
                "name",
                "arguments",
                "args",
                "input",
                "output",
                "response",
            }:
                result[key] = self.data(child)
        return result

    def tool_output(self, value: Any) -> Any:
        if not isinstance(value, list):
            return self.data(value)
        return [
            self.block(item)
            if isinstance(item, dict)
            and item.get("type") in ("input_text", "input_image", "input_file")
            else self.data(item)
            for item in value
        ]

    def function(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        for key, child in value.items():
            if key in {"parameters", "parametersJsonSchema", "input_schema"}:
                result[key] = self.schema(child)
            elif key in {"name", "description", "arguments", "args", "input", "response"}:
                result[key] = self.data(child)
        return result

    def call(self, value: Any) -> Any:
        if not isinstance(value, dict) or value.get("type", "function") != "function":
            return value
        result = dict(value)
        if "function" in value:
            result["function"] = self.function(value["function"])
        return result

    def tool(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        kind = value.get("type")
        if not isinstance(kind, (str, type(None))):
            return value
        if kind in {"web_search", "web_search_preview", "web_search_preview_2025_03_11"}:
            return self.search_tool(value)
        if kind not in {None, "function", "custom", "namespace"}:
            return value
        result = self.function(value)
        for key, child in value.items():
            if key == "function":
                result[key] = self.function(child)
            elif key == "tools" and kind == "namespace" and isinstance(child, list):
                result[key] = [self.tool(item) for item in child]
            elif key in {"functionDeclarations", "function_declarations"} and isinstance(
                child, list
            ):
                result[key] = [self.function(item) for item in child]
        return result

    def search_tool(self, value: dict[str, Any]) -> dict[str, Any]:
        result = dict(value)
        for key in ("filters", "allowed_domains", "blocked_domains"):
            if key in value:
                result[key] = self.data(value[key])
        child = value.get("user_location")
        if isinstance(child, dict) and child.get("type") == "approximate":
            result["user_location"] = {
                name: item if name == "type" else self.data(item) for name, item in child.items()
            }
        return result

    def schema(self, value: Any, root: Any = None) -> Any:
        if not isinstance(value, dict):
            return value
        if root is None:
            root = value
        result = dict(value)
        for key, child in value.items():
            if key in _SCHEMA_MAPS and isinstance(child, dict):
                result[key] = {
                    self.data(name): self.schema(definition, root)
                    for name, definition in child.items()
                }
            elif key in _SCHEMA_CHILDREN:
                result[key] = self.schema(child, root)
            elif key in _SCHEMA_LISTS and isinstance(child, list):
                result[key] = [self.schema(item, root) for item in child]
            elif key in {"$ref", "$dynamicRef"} and isinstance(child, str):
                result[key] = (
                    self.reference(child, root) if child.startswith("#/") else self.data(child)
                )
            elif key in _SCHEMA_DATA or key in _SCHEMA_TEXT or key == "required":
                result[key] = self.data(child)
        return result

    def reference(self, value: str, root: Any) -> str:
        cursor = root
        names = False
        result = []
        for raw in value[2:].split("/"):
            part = raw.replace("~1", "/").replace("~0", "~")
            rendered = self.data(part) if names else part
            result.append(rendered.replace("~", "~0").replace("/", "~1"))
            if isinstance(cursor, dict) and part in cursor:
                cursor = cursor[part]
            elif isinstance(cursor, list) and part.isdecimal() and int(part) < len(cursor):
                cursor = cursor[int(part)]
            names = part in _SCHEMA_MAPS and not names
        return "#/" + "/".join(result)

    def output_format(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        for key, child in value.items():
            if key in {"format", "json_schema"}:
                result[key] = self.output_format(child)
            elif key == "schema":
                result[key] = self.schema(child)
            elif key in {"name", "description"}:
                result[key] = self.data(child)
        return result

    def choice(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        for key, child in value.items():
            if key in {"function", "functionCallingConfig", "function_calling_config"}:
                result[key] = self.choice(child)
            elif key in {"name", "allowedFunctionNames", "allowed_function_names"}:
                result[key] = self.data(child)
        return result

    def settings(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        for key, child in value.items():
            if key in {"responseSchema", "response_schema", "responseJsonSchema"}:
                result[key] = self.schema(child)
            elif key in {"thinkingConfig", "thinking_config"}:
                result[key] = self.settings(child)
            elif key in {"stopSequences", "stop_sequences"}:
                result[key] = self.data(child)
        return result
