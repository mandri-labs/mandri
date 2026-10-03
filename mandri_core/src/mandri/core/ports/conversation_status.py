from collections.abc import Sequence
from typing import Any, Protocol

from mandri.core.types.conversation_status import ConversationStatus, WorkObservation


class ConversationStatusRepositoryPort(Protocol):
    async def all(self) -> list[ConversationStatus]: ...

    async def observe(
        self,
        target: str,
        observations: Sequence[WorkObservation],
        source: str | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> ConversationStatus: ...

    async def checkpoint(self, target: str, source: str) -> dict[str, Any] | None: ...

    async def acknowledge(
        self, target: str, through_revision: int, completion_key: str | None = None
    ) -> ConversationStatus: ...

    async def link(self, source: str, target: str) -> ConversationStatus: ...
