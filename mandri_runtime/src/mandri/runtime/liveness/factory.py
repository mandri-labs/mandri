from collections.abc import Awaitable, Callable

from mandri.core.hub import Hub, Topic
from mandri.core.ids import HarnessKind, SessionId
from mandri.runtime.control.codex_liveness import CodexLivenessSnapshot
from mandri.runtime.liveness.agy import AgyLivenessAdapter
from mandri.runtime.liveness.claude import ClaudeLivenessAdapter
from mandri.runtime.liveness.codex import CodexLivenessAdapter
from mandri.runtime.liveness.opencode import OpencodeLivenessAdapter
from mandri.runtime.liveness.pi import PiLivenessAdapter
from mandri.runtime.liveness.port import LivenessPort

LivenessAdapter = (
    ClaudeLivenessAdapter
    | CodexLivenessAdapter
    | OpencodeLivenessAdapter
    | AgyLivenessAdapter
    | PiLivenessAdapter
)


def liveness_adapter(
    kind: HarnessKind,
    hub: Hub,
    topic: Topic,
    port: LivenessPort,
    session_id: SessionId,
    *,
    since: int = 0,
    native_identity: Callable[[], str | None] | None = None,
    read_queue: Callable[[], Awaitable[bool]] | None = None,
    read_state: Callable[[], Awaitable[CodexLivenessSnapshot]] | None = None,
) -> LivenessAdapter:
    if kind is HarnessKind.CODEX:
        return CodexLivenessAdapter(
            hub,
            topic,
            port,
            session_id,
            native_identity,
            since=since,
            read_queue=read_queue,
            read_state=read_state,
        )
    if kind is HarnessKind.OPENCODE:
        return OpencodeLivenessAdapter(
            hub, topic, port, session_id, since=since, native_identity=native_identity
        )
    if kind is HarnessKind.AGY:
        return AgyLivenessAdapter(
            hub,
            topic,
            port,
            session_id,
            since=since,
            native_id=native_identity() if native_identity else None,
        )
    if kind is HarnessKind.PI:
        return PiLivenessAdapter(hub, topic, port, session_id, since=since)
    return ClaudeLivenessAdapter(hub, topic, port, session_id, since=since)
