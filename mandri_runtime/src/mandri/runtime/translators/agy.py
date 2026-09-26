from typing import Any

from mandri.runtime.translators.base import EventPipe


class AgyEventPipe(EventPipe):
    source = "agy"

    def _wrap(self, event: Any) -> dict[str, Any] | None:
        return self._frame(event) if isinstance(event, dict) else None
