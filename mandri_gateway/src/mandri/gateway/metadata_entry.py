from dataclasses import fields, replace
from typing import Any

from mandri.core.model_metadata import ModelMetadata
from mandri.gateway.model_capabilities import capability, metadata_capabilities
from mandri.gateway.model_limits import model_limits
from mandri.gateway.reasoning_metadata import parse_reasoning_entry, parse_template_reasoning


def parse_entry(entry: dict[str, Any], source: str = "provider") -> ModelMetadata:
    reasoning = parse_reasoning_entry(entry) or parse_template_reasoning(entry.get("chat_template"))
    capabilities = metadata_capabilities(entry)
    if capabilities["reasoning_supported"] is False:
        return ModelMetadata(
            **model_limits(entry),
            **capabilities,
            reasoning_efforts=(),
            hosted_web_search=capability(entry, "hosted_web_search", "web_search_options"),
        ).with_source(source)
    if capabilities["reasoning_supported"] is None and reasoning is not None:
        capabilities["reasoning_supported"] = bool(reasoning.efforts)
    return ModelMetadata(
        **model_limits(entry),
        **capabilities,
        reasoning_efforts=tuple(reasoning.efforts) if reasoning is not None else None,
        default_reasoning_effort=reasoning.default_effort if reasoning else None,
        hosted_web_search=capability(entry, "hosted_web_search", "web_search_options"),
    ).with_source(source)


def consensus(entries: list[ModelMetadata]) -> ModelMetadata:
    if not entries:
        return ModelMetadata()
    values = {
        item.name: getattr(entries[0], item.name)
        for item in fields(ModelMetadata)
        if item.name != "sources"
        and all(getattr(entry, item.name) == getattr(entries[0], item.name) for entry in entries)
    }
    return replace(ModelMetadata(), **values).with_source("catalog")
