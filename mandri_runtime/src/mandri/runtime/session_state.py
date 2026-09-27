import asyncio
from dataclasses import dataclass, field

from mandri.core.ids import HarnessSessionId
from mandri.core.ports.control import ApprovalDelivery, HarnessControl
from mandri.core.types.execution import SessionPolicy
from mandri.core.types.model_selection import ModelSource
from mandri.runtime.agy_launch import AgyLaunch
from mandri.runtime.approvals.watcher import ApprovalWatcher
from mandri.runtime.control.agy_commands import AgyCommandRunner
from mandri.runtime.liveness import (
    ClaudeLivenessAdapter,
    CodexLivenessAdapter,
    OpencodeLivenessAdapter,
)
from mandri.runtime.liveness.agy import AgyLivenessAdapter
from mandri.runtime.liveness.pi import PiLivenessAdapter
from mandri.runtime.pi_session_paths import PiSessionCheckpoint
from mandri.runtime.session_feed import SessionFeed
from mandri.sessions.usage.types import NativeUsageContext

LivenessAdapter = (
    ClaudeLivenessAdapter
    | CodexLivenessAdapter
    | OpencodeLivenessAdapter
    | AgyLivenessAdapter
    | PiLivenessAdapter
)
ModelSelection = tuple[ModelSource, str, str | None]


@dataclass
class SessionRuntimeState:
    policy: SessionPolicy = field(default_factory=SessionPolicy)
    policy_revision: int = 1
    agy: AgyLaunch | None = None
    agy_commands: AgyCommandRunner | None = None
    mode_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    prompt_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    native_id: HarnessSessionId | None = None
    pi_checkpoint: PiSessionCheckpoint | None = None
    native_restore_failed: bool = False
    launched_model: ModelSelection | None = None
    control: HarnessControl | None = None
    delivery: ApprovalDelivery | None = None
    feed: SessionFeed | None = None
    feed_start_seq: int = 0
    watcher: ApprovalWatcher | None = None
    liveness_adapter: LivenessAdapter | None = None
    identity_task: asyncio.Task[HarnessSessionId | None] | None = None
    event_pump: asyncio.Task[None] | None = None
    control_tasks: set[asyncio.Task[None]] = field(default_factory=set)
    lifetime_task: asyncio.Task[None] | None = None
    idle_since: float | None = None
    stopping: bool = False
    stop_revision: int = 0
    resuming: bool = False
    viewed: bool = False
    control_auth: tuple[str, str] | None = None
    usage_context: NativeUsageContext | None = None


class RuntimeStates:
    def __init__(self) -> None:
        self._states: dict[str, SessionRuntimeState] = {}

    def session(self, session_id: str) -> SessionRuntimeState:
        return self._states.setdefault(session_id, SessionRuntimeState())

    def items(self) -> list[tuple[str, SessionRuntimeState]]:
        return list(self._states.items())
