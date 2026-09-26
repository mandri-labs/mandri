"""Codex pipe: JSON-RPC messages from a connection read loop, verbatim."""

from typing import Any

from mandri.runtime.translators.base import EventPipe


class CodexEventPipe(EventPipe):
    source = "codex"

    def _wrap(self, event: Any) -> dict[str, Any] | None:
        if not isinstance(event, dict):
            return None
        return self._frame(event)
