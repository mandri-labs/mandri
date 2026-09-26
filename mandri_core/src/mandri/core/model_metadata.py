from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelMetadata:
    context_window: int = 128000
    output_tokens: int = 8192
    reasoning_efforts: tuple[str, ...] = ()
    image_input: bool | None = None
    hosted_web_search: bool = False
    input_modalities: tuple[str, ...] | None = None
    output_modalities: tuple[str, ...] | None = None
    reasoning_supported: bool | None = None
    tool_call: bool | None = None
    attachment: bool | None = None
    temperature: bool | None = None

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
        context = payload.get("context_window")
        output = payload.get("output_tokens")
        if not isinstance(context, int) or not isinstance(output, int):
            return None
        efforts = payload.get("reasoning_efforts")
        return cls(
            context_window=context,
            output_tokens=output,
            reasoning_efforts=tuple(efforts) if isinstance(efforts, list) else (),
            image_input=optional_bool(payload.get("image_input")),
            hosted_web_search=payload.get("hosted_web_search") is True,
            input_modalities=optional_strings(payload.get("input_modalities")),
            output_modalities=optional_strings(payload.get("output_modalities")),
            reasoning_supported=optional_bool(payload.get("reasoning_supported")),
            tool_call=optional_bool(payload.get("tool_call")),
            attachment=optional_bool(payload.get("attachment")),
            temperature=optional_bool(payload.get("temperature")),
        )


def optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def optional_strings(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return None
