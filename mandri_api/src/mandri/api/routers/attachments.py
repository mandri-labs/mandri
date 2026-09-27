import asyncio
from collections.abc import AsyncIterator
from typing import Annotated, BinaryIO
from urllib.parse import quote

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from mandri.api.deps import Runtime, Sessions
from mandri.api.errors import ApiError
from mandri.core.ids import SessionId
from mandri.core.protocol.types import SessionId as ApiSessionId
from mandri.runtime.attachments import MAX_FILE_BYTES, AttachmentError, AttachmentStorageError
from mandri.runtime.session_files import open_session_file
from mandri.sessions.errors import SessionNotFoundError
from pydantic import BaseModel
from starlette.types import Receive, Scope, Send

router = APIRouter(prefix="/sessions", tags=["attachments"])


class AttachmentOut(BaseModel):
    id: str
    name: str
    media_type: str
    size: int
    reference: str


class FileDownload(StreamingResponse):
    def __init__(self, stream: BinaryIO, name: str, size: int) -> None:
        self.stream = stream
        super().__init__(
            self.chunks(),
            media_type="application/octet-stream",
            headers={
                "Content-Length": str(size),
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(name, safe='')}",
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    async def chunks(self) -> AsyncIterator[bytes]:
        while chunk := await asyncio.to_thread(self.stream.read, 65536):
            yield chunk

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.stream.close()


@router.post(
    "/{session_id}/attachments",
    operation_id="upload_attachment",
    status_code=201,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
            },
        }
    },
)
async def upload_attachment(
    session_id: ApiSessionId,
    name: Annotated[str, Query(min_length=1, max_length=200)],
    request: Request,
    runtime: Runtime,
    sessions: Sessions,
) -> AttachmentOut:
    try:
        session = await sessions.get_session(SessionId(session_id))
        length = request.headers.get("content-length")
        if length and int(length) > MAX_FILE_BYTES:
            raise AttachmentError("File exceeds 20 MiB")
        attachment = await runtime.attachments.upload(session, name, request.stream())
        prepared = await asyncio.to_thread(
            runtime.attachments.prepare, session, "", [attachment.id]
        )
        return AttachmentOut(
            id=attachment.id,
            name=attachment.name,
            media_type=attachment.media_type,
            size=attachment.size,
            reference=prepared.text,
        )
    except SessionNotFoundError as error:
        raise ApiError(code="not_found", message="Session not found", status=404) from error
    except AttachmentStorageError as error:
        raise ApiError(
            code="attachment_storage_unavailable", message=str(error), status=503
        ) from error
    except ValueError as error:
        raise ApiError(code="invalid_params", message=str(error), status=400) from error


@router.get(
    "/{session_id}/files", operation_id="download_session_file", response_class=StreamingResponse
)
async def download_session_file(
    session_id: ApiSessionId,
    path: Annotated[str, Query(min_length=1, max_length=4096)],
    runtime: Runtime,
    sessions: Sessions,
) -> StreamingResponse:
    try:
        session = await sessions.get_session(SessionId(session_id))
        stream, name, size = await asyncio.to_thread(
            open_session_file, session, runtime.attachments, path
        )
        return FileDownload(stream, name, size)
    except SessionNotFoundError as error:
        raise ApiError(code="not_found", message="Session not found", status=404) from error
    except (AttachmentError, OSError) as error:
        raise ApiError(code="not_found", message=str(error), status=404) from error
