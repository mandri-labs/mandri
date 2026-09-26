"""Claude pipe: SDK message dataclasses or dicts, serialized verbatim."""

import dataclasses
from typing import Any

from mandri.runtime.translators.base import EventPipe


class ClaudeEventPipe(EventPipe):
    source = "claude"

    def _wrap(self, event: Any) -> dict[str, Any] | None:
        raw: Any = event
        if dataclasses.is_dataclass(raw) and not isinstance(raw, type):
            raw = dataclasses.asdict(raw)
        if not isinstance(raw, dict):
            return None
        return self._frame(raw)
