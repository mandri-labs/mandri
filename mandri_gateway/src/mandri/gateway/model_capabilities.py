from typing import Any

from mandri.core.model_metadata import optional_bool, optional_strings


def modalities(entry: dict[str, Any], direction: str) -> tuple[str, ...] | None:
    for container, key in (("architecture", f"{direction}_modalities"), ("modalities", direction)):
        value = entry.get(container)
        if isinstance(value, dict):
            result = optional_strings(value.get(key))
            if result is not None:
                return tuple("pdf" if item == "file" else item for item in result)
    return optional_strings(entry.get(f"{direction}_modalities"))


def capability(entry: dict[str, Any], name: str, *parameters: str) -> bool | None:
    capabilities = entry.get("capabilities")
    for source in (capabilities, entry):
        if isinstance(source, dict):
            value = optional_bool(source.get(name))
            if value is not None:
                return value
    supported = optional_strings(entry.get("supported_parameters"))
    if parameters and supported is not None:
        return any(parameter in supported for parameter in parameters)
    return None


def metadata_capabilities(entry: dict[str, Any]) -> dict[str, Any]:
    inputs = modalities(entry, "input")
    attachment = capability(entry, "attachment")
    if attachment is None and inputs is not None:
        attachment = any(item != "text" for item in inputs)
    vision = capability(entry, "vision")
    tools = capability(entry, "tool_call", "tools", "tool_choice")
    if tools is None:
        tools = capability(entry, "trained_for_tool_use")
    reasoning = capability(entry, "reasoning", "reasoning", "reasoning_effort", "thinking")
    capabilities = entry.get("capabilities")
    reasoning_options = capabilities.get("reasoning") if isinstance(capabilities, dict) else None
    if reasoning is None and isinstance(reasoning_options, dict):
        options = optional_strings(reasoning_options.get("allowed_options"))
        if options is not None:
            reasoning = any(option != "off" for option in options)
    return {
        "input_modalities": inputs,
        "output_modalities": modalities(entry, "output"),
        "image_input": "image" in inputs if inputs is not None else vision,
        "reasoning_supported": reasoning,
        "tool_call": tools,
        "attachment": attachment,
        "temperature": capability(entry, "temperature", "temperature"),
    }
