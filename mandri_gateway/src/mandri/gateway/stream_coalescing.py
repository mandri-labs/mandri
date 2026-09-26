"""Coalescing of terminal-only litellm stream chunks."""

from typing import Any


def _has_payload(delta: Any) -> bool:
    return bool(
        getattr(delta, "content", None)
        or getattr(delta, "tool_calls", None)
        or getattr(delta, "function_call", None)
        or getattr(delta, "reasoning_content", None)
    )


class StreamCoalescer:
    """Merge terminal-only chunks into the preceding chunk and pass usage chunks through."""

    def __init__(self) -> None:
        self._pending: Any | None = None

    def _release_with(self, chunk: Any) -> list[Any]:
        released = [] if self._pending is None else [self._pending]
        self._pending = None
        return [*released, chunk]

    def add(self, chunk: Any) -> list[Any]:
        choices = getattr(chunk, "choices", None)
        if not choices:
            if getattr(chunk, "usage", None) is not None:
                return self._release_with(chunk)
            return []
        if _has_payload(choices[0].delta):
            released = [] if self._pending is None else [self._pending]
            self._pending = chunk
            return released
        if getattr(chunk, "usage", None) is not None:
            return self._release_with(chunk)
        finish_reason = getattr(choices[0], "finish_reason", None)
        if self._pending is not None and finish_reason is not None:
            self._pending.choices[0].finish_reason = finish_reason
        return []

    def flush(self) -> list[Any]:
        released = [] if self._pending is None else [self._pending]
        self._pending = None
        return released
