from typing import Protocol


class PiSessionIdentityPort(Protocol):
    async def adopt(self, session_id: str, native_id: str) -> bool: ...
