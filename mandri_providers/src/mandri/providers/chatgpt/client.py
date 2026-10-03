"""Long-lived HTTP client for ChatGPT OAuth and inference endpoints."""

import httpx

DEFAULT_TIMEOUT_SECONDS = 30.0


class ChatGptClient:
    """Owns one pooled client so token refresh reuses the connection pool."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client if client is not None else httpx.AsyncClient()
        self._owned = client is None

    @property
    def http(self) -> httpx.AsyncClient:
        return self._client

    async def aclose(self) -> None:
        if self._owned:
            await self._client.aclose()
