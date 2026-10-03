from typing import Protocol

from mandri.core.types.execution import PrivacyMode
from mandri.core.types.sessions import Session


class SessionPrivacyPort(Protocol):
    async def set_privacy(
        self, expected: Session, mode: PrivacyMode, scope_id: str | None
    ) -> None: ...
