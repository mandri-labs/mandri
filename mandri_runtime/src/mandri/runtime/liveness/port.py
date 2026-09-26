"""Domain-facing liveness port consumed by the runtime service."""

import typing

from mandri.core.ids import SessionId
from mandri.runtime.liveness.evidence import LivenessEvidence
from mandri.runtime.liveness.types import WorkingState


class LivenessPort(typing.Protocol):
    def register(self, session_id: SessionId) -> None: ...

    def forget(self, session_id: SessionId) -> None: ...

    def observe(self, evidence: LivenessEvidence) -> None: ...

    def working_state(self, session_id: SessionId) -> WorkingState: ...
