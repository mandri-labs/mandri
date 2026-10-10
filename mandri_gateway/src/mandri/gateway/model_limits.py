from typing import Any, TypedDict

from mandri.core.model_metadata import positive_int


def value_at(entry: dict[str, Any], path: str) -> Any:
    value: Any = entry
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def first_limit(entry: dict[str, Any], *paths: str) -> int | None:
    return next(
        (value for path in paths if (value := positive_int(value_at(entry, path))) is not None),
        None,
    )


class ModelLimits(TypedDict):
    context_window: int | None
    max_context_window: int | None
    input_tokens: int | None
    output_tokens: int | None
    output_budget: int | None
    auto_compact_token_limit: int | None


def model_limits(entry: dict[str, Any]) -> ModelLimits:
    loaded = entry.get("loaded_instances")
    contexts = (
        [
            value
            for instance in loaded
            if isinstance(instance, dict)
            if (value := first_limit(instance, "config.context_length", "config.n_ctx")) is not None
        ]
        if isinstance(loaded, list)
        else []
    )
    maximum = first_limit(entry, "max_context_window", "max_context_length")
    info = entry.get("model_info")
    if maximum is None and isinstance(info, dict):
        maximum = next(
            (
                value
                for key, raw in info.items()
                if key.endswith(".context_length") and (value := positive_int(raw)) is not None
            ),
            None,
        )
    context = (
        min(contexts)
        if contexts
        else first_limit(
            entry,
            "active_context_length",
            "config.context_length",
            "config.n_ctx",
            "parameters.num_ctx",
            "num_ctx",
            "default_generation_settings.n_ctx",
            "meta.n_ctx",
            "n_ctx",
            "top_provider.context_length",
            "context_window",
            "context_length",
            "limit.context",
        )
    )
    return {
        "context_window": context or maximum,
        "max_context_window": maximum,
        "input_tokens": first_limit(
            entry, "input_tokens", "max_input_tokens", "inputTokenLimit", "limit.input"
        ),
        "output_tokens": first_limit(
            entry,
            "top_provider.max_completion_tokens",
            "output_tokens",
            "max_output_tokens",
            "max_completion_tokens",
            "outputTokenLimit",
            "limit.output",
        ),
        "output_budget": first_limit(
            entry,
            "output_budget",
            "parameters.num_predict",
            "default_max_tokens",
            "default_generation_settings.n_predict",
            "default_generation_settings.params.n_predict",
            "generation_config.max_tokens",
        ),
        "auto_compact_token_limit": first_limit(entry, "auto_compact_token_limit"),
    }
