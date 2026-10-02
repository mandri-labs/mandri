"""Opencode session mutation adapters over the opencode HTTP API."""

import asyncio
import contextlib
from collections.abc import AsyncIterator

import httpx
from mandri.core.ids import SessionId, SessionTitle
from mandri.core.ports.sessions import (
    CheckSessionExistsPort,
    DeleteSessionPort,
    RenameSessionPort,
)
from mandri.sessions.errors import (
    ServerCommunicationError,
    SessionDeleteError,
    SessionRenameError,
)

REQUEST_TIMEOUT_S = 10.0


class _OpencodeRestAdapter:
    """Shared per-call HTTP client setup for the opencode REST adapters."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._base_url = str(client.base_url)

    @contextlib.asynccontextmanager
    async def _session(self) -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=REQUEST_TIMEOUT_S) as client:
            yield client

    @staticmethod
    def _communication_error(
        session_id: SessionId, error: httpx.HTTPError
    ) -> ServerCommunicationError:
        return ServerCommunicationError(f"cannot reach opencode server for {session_id}: {error}")


class OpencodeRestRenameSessionAdapter(_OpencodeRestAdapter, RenameSessionPort):
    """Rename sessions through PATCH /api/session/{id}."""

    def rename(self, session_id: SessionId, title: SessionTitle) -> None:
        asyncio.run(self.rename_async(session_id, title))

    async def rename_async(self, session_id: SessionId, title: SessionTitle) -> None:
        if not str(title).strip():
            raise SessionRenameError("title must not be empty")
        async with self._session() as client:
            try:
                response = await client.patch(
                    f"/api/session/{session_id}", json={"title": str(title)}
                )
            except httpx.HTTPError as error:
                raise self._communication_error(session_id, error) from error
            if response.status_code >= 400:
                raise SessionRenameError(
                    f"rename failed for {session_id}: HTTP {response.status_code}"
                )


class OpencodeRestDeleteSessionAdapter(_OpencodeRestAdapter, DeleteSessionPort):
    """Delete sessions through DELETE /api/session/{id}."""

    def delete(self, session_id: SessionId) -> None:
        asyncio.run(self.delete_async(session_id))

    async def delete_async(self, session_id: SessionId) -> None:
        async with self._session() as client:
            try:
                response = await client.delete(f"/api/session/{session_id}")
            except httpx.HTTPError as error:
                raise self._communication_error(session_id, error) from error
            if response.status_code >= 400:
                raise SessionDeleteError(
                    f"delete failed for {session_id}: HTTP {response.status_code}"
                )


class OpencodeRestCheckSessionExistsAdapter(_OpencodeRestAdapter, CheckSessionExistsPort):
    """Check session existence through GET /api/session/{id}."""

    def exists(self, session_id: SessionId) -> bool:
        return asyncio.run(self.exists_async(session_id))

    async def exists_async(self, session_id: SessionId) -> bool:
        async with self._session() as client:
            try:
                response = await client.get(f"/api/session/{session_id}")
            except httpx.HTTPError as error:
                raise self._communication_error(session_id, error) from error
        if response.status_code == 404:
            return False
        if response.status_code >= 400:
            raise ServerCommunicationError(
                f"opencode server error for {session_id}: HTTP {response.status_code}"
            )
        return True
