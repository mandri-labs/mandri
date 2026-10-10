import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class PromptFingerprint:
    instructions: str
    tools: str
    system: tuple[str, ...]
    input: tuple[str, ...]

    @classmethod
    def from_body(cls, body: dict[str, Any]) -> "PromptFingerprint":
        raw = body.get("input", body.get("messages", body.get("contents", [])))
        messages = raw if isinstance(raw, list) else [raw]
        system = tuple(
            fingerprint(message)
            for message in messages
            if isinstance(message, dict) and message.get("role") in ("system", "developer")
        )
        return cls(
            fingerprint(
                body.get("instructions", body.get("system", body.get("systemInstruction")))
            ),
            fingerprint(body.get("tools")),
            system,
            tuple(fingerprint(message) for message in messages),
        )


class PromptTrace:
    def __init__(self) -> None:
        self._previous: OrderedDict[str, PromptFingerprint] = OrderedDict()

    def observe(self, conversation: str, body: dict[str, Any]) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        current = PromptFingerprint.from_body(body)
        previous = self._previous.get(conversation)
        self._previous[conversation] = current
        self._previous.move_to_end(conversation)
        if len(self._previous) > 1024:
            self._previous.popitem(last=False)
        common = 0
        if previous is not None:
            for before, after in zip(previous.input, current.input, strict=False):
                if before != after:
                    break
                common += 1
        logger.debug(
            "Prompt prefix conversation=%s instructions_changed=%s tools_changed=%s "
            "system_changed=%s common_input_items=%s input_items=%s",
            conversation,
            previous is not None and previous.instructions != current.instructions,
            previous is not None and previous.tools != current.tools,
            previous is not None and previous.system != current.system,
            common,
            len(current.input),
        )
