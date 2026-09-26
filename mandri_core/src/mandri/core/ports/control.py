"""HarnessControl port and shared control outcome types."""

import dataclasses
import enum
import typing

from mandri.core.ids import ApprovalDecision, HarnessSessionId, ModeApplication
from mandri.core.types.approvals import ApprovalRequest
from mandri.core.types.prompt import UserPrompt


class PromptState(enum.StrEnum):
    STEERED = "steered"
    QUEUED = "queued"
    ERROR = "error"


@dataclasses.dataclass(frozen=True)
class PromptOutcome:
    state: PromptState
    code: str | None = None


@dataclasses.dataclass(frozen=True)
class ControlRequest:
    native_request_ref: str
    tool_name: str
    tool_input: dict[str, typing.Any]
    tool_use_id: str


class ControlSink(typing.Protocol):
    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...


class ApprovalDelivery(typing.Protocol):
    async def deliver(self, request: ApprovalRequest) -> bool: ...


class HarnessControl(typing.Protocol):
    async def answer_approval(
        self, native_request_ref: str, decision: ApprovalDecision
    ) -> bool: ...

    async def set_mode(self, mode: str) -> ModeApplication: ...

    async def send_prompt(self, content: str | UserPrompt) -> PromptOutcome: ...

    async def interrupt(self) -> bool: ...

    async def capture_identity(self) -> HarnessSessionId | None: ...

    async def next_native_request(self) -> ControlRequest: ...

    async def aclose(self) -> None: ...
