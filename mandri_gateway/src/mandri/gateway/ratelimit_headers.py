"""Rate-limit header allow-list extraction for gateway pass-through."""

from collections.abc import Mapping
from typing import Any, cast

_RETRY_AFTER = "retry-after"
_RESPONSE_HEADERS_PREFIX = "llm_provider-"
_PROVIDER_PREFIXES = ("anthropic-ratelimit-", "x-ratelimit-")


def _normalize(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith(_RESPONSE_HEADERS_PREFIX):
        return lowered[len(_RESPONSE_HEADERS_PREFIX) :]
    return lowered


def _allowed(normalized: str) -> bool:
    if normalized == _RETRY_AFTER:
        return True
    return any(normalized.startswith(prefix) for prefix in _PROVIDER_PREFIXES)


def filter_headers(headers: Mapping[str, str]) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for name, value in headers.items():
        normalized = _normalize(name)
        if _allowed(normalized) and normalized not in filtered:
            filtered[normalized] = value
    return filtered


def from_success(result: Any) -> dict[str, str]:
    hidden = getattr(result, "_hidden_params", None)
    if not isinstance(hidden, dict):
        return {}
    additional = hidden.get("additional_headers")
    if not isinstance(additional, dict):
        return {}
    return filter_headers(additional)


def _exception_headers(error: BaseException) -> Mapping[str, str] | None:
    headers = getattr(error, "headers", None)
    if headers:
        return cast(Mapping[str, str], headers)
    response = getattr(error, "response", None)
    response_headers = getattr(response, "headers", None)
    if response_headers:
        return cast(Mapping[str, str], response_headers)
    return None


def from_exception(error: BaseException) -> dict[str, str]:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        headers = _exception_headers(current)
        if headers is not None:
            filtered = filter_headers(headers)
            if filtered:
                return filtered
        current = current.__cause__ or current.__context__
    return {}
