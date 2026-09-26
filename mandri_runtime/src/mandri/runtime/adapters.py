"""Factory seam for harness control and approval delivery adapters."""

import dataclasses
import typing

from mandri.core.hub import Hub, Topic
from mandri.core.ids import HarnessKind, HarnessSessionId
from mandri.core.ports.control import ApprovalDelivery, HarnessControl
from mandri.core.types.model_selection import ModelSource
from mandri.runtime.control.agy_bridge import AgyBridge
from mandri.runtime.control.agy_commands import AgyCommandRunner
from mandri.runtime.control.modes import LaunchMode
from mandri.runtime.process import ManagedProcess


@dataclasses.dataclass(frozen=True)
class AdapterContext:
    kind: HarnessKind
    process: ManagedProcess | None = None
    hub: Hub | None = None
    topic: Topic | None = None
    launch_mode: LaunchMode | None = None
    listen_port: int | None = None
    native_session_id: str | None = None
    resume_thread_id: HarnessSessionId | None = None
    model: str | None = None
    model_source: ModelSource = ModelSource.GATEWAY
    reasoning_effort: str | None = None
    on_identity: typing.Callable[[HarnessSessionId], None] | None = None
    on_conversation_reset: (
        typing.Callable[[HarnessSessionId], typing.Awaitable[None] | None] | None
    ) = None
    on_session_path: typing.Callable[[HarnessSessionId, str], None] | None = None
    on_model_selection: (
        typing.Callable[[str, str | None], typing.Awaitable[None] | None] | None
    ) = None
    agy_bridge: AgyBridge | None = None
    agy_commands: AgyCommandRunner | None = None
    control_auth: tuple[str, str] | None = None
    fork_thread_id: HarnessSessionId | None = None
    fork_path: str | None = None
    workspace_root: str | None = None


class EventSource(typing.Protocol):
    async def next_event(self) -> dict[str, typing.Any] | None: ...


@dataclasses.dataclass(frozen=True)
class HarnessAdapters:
    control: HarnessControl
    delivery: ApprovalDelivery | None = None
    events: EventSource | None = None


HarnessAdapterFactory = typing.Callable[[AdapterContext], HarnessAdapters | None]
