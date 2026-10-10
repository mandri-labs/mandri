from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any


@dataclass(frozen=True)
class ModelMetadata:
    context_window: int | None = None
    output_tokens: int | None = None
    reasoning_efforts: tuple[str, ...] | None = None
    image_input: bool | None = None
    hosted_web_search: bool | None = None
    input_modalities: tuple[str, ...] | None = None
    output_modalities: tuple[str, ...] | None = None
    reasoning_supported: bool | None = None
    tool_call: bool | None = None
    attachment: bool | None = None
    temperature: bool | None = None
    max_context_window: int | None = None
    input_tokens: int | None = None
    output_budget: int | None = None
    auto_compact_token_limit: int | None = None
    default_reasoning_effort: str | None = None
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def available_context(self) -> int | None:
        return self.context_window or self.input_tokens

    def with_fallback(self, fallback: "ModelMetadata") -> "ModelMetadata":
        changes = {
            item.name: getattr(fallback, item.name)
            for item in fields(self)
            if item.name != "sources" and getattr(self, item.name) is None
        }
        sources = {**fallback.sources, **self.sources}
        if self.reasoning_supported is True and fallback.reasoning_supported is False:
            for key in ("reasoning_efforts", "default_reasoning_effort"):
                if getattr(self, key) is None:
                    changes.pop(key, None)
                    sources.pop(key, None)
        if self.reasoning_supported is False or (
            self.reasoning_supported is None and fallback.reasoning_supported is False
        ):
            changes.update(reasoning_efforts=(), default_reasoning_effort=None)
        return replace(self, **changes, sources=sources)

    def with_source(self, source: str) -> "ModelMetadata":
        return replace(
            self,
            sources={
                item.name: source
                for item in fields(self)
                if item.name != "sources" and getattr(self, item.name) is not None
            },
        )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, tuple):
                payload[key] = list(value)
        return payload

    @classmethod
    def from_payload(cls, payload: Any) -> "ModelMetadata | None":
        if not isinstance(payload, dict):
            return None
        efforts = payload.get("reasoning_efforts")
        reasoning = payload.get("reasoning")
        if efforts is None and isinstance(reasoning, dict):
            efforts = reasoning.get("efforts")
        default = payload.get("default_reasoning_effort")
        if default is None and isinstance(reasoning, dict):
            default = reasoning.get("default_effort")
        sources = payload.get("sources")
        return cls(
            context_window=positive_int(payload.get("context_window")),
            output_tokens=positive_int(payload.get("output_tokens")),
            reasoning_efforts=optional_strings(efforts),
            image_input=optional_bool(payload.get("image_input")),
            hosted_web_search=optional_bool(payload.get("hosted_web_search")),
            input_modalities=optional_strings(payload.get("input_modalities")),
            output_modalities=optional_strings(payload.get("output_modalities")),
            reasoning_supported=optional_bool(payload.get("reasoning_supported")),
            tool_call=optional_bool(payload.get("tool_call")),
            attachment=optional_bool(payload.get("attachment")),
            temperature=optional_bool(payload.get("temperature")),
            max_context_window=positive_int(payload.get("max_context_window")),
            input_tokens=positive_int(payload.get("input_tokens")),
            output_budget=positive_int(payload.get("output_budget")),
            auto_compact_token_limit=positive_int(payload.get("auto_compact_token_limit")),
            default_reasoning_effort=default if isinstance(default, str) and default else None,
            sources={key: value for key, value in sources.items() if isinstance(value, str)}
            if isinstance(sources, dict)
            else {},
        )


def optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def positive_int(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def optional_strings(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return None
