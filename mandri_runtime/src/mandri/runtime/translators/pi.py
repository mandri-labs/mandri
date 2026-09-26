from typing import Any

from mandri.runtime.translators.base import EventPipe


class PiEventPipe(EventPipe):
    source = "pi"

    def _wrap(self, event: Any) -> dict[str, Any] | None:
        if not isinstance(event, dict):
            return None
        if (
            event.get("type") == "extension_ui_request"
            and event.get("method") == "setStatus"
            and event.get("statusKey") == "_mandri_history_changed"
        ):
            return {
                "source": "mandri",
                "raw": {"type": "history_changed", "reset": True},
                "ts": self._clock(),
            }
        return self._frame(event)
