"""Native transcript record downloads."""

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from mandri.api.deps import Sessions
from mandri.api.routers.sessions import HISTORY_RESPONSES, _history_error
from mandri.core.ids import PageToken, SessionId
from mandri.core.protocol.types import SessionId as ApiSessionId
from mandri.sessions.transcripts.errors import TranscriptError
from mandri.sessions.transcripts.record_download import RecordDownload
from starlette.types import Receive, Scope, Send

router = APIRouter(prefix="/sessions", tags=["sessions"])


class RecordResponse(StreamingResponse):
    def __init__(self, record: RecordDownload) -> None:
        self.record = record
        super().__init__(
            record.chunks(),
            media_type="application/octet-stream",
            headers={
                "Content-Length": str(record.size),
                "Content-Disposition": 'attachment; filename="transcript-record.jsonl"',
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.record.close()


@router.get(
    "/{session_id}/history/record",
    operation_id="download_transcript_record",
    response_class=StreamingResponse,
    responses={
        **HISTORY_RESPONSES,
        200: {
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
            }
        },
    },
)
async def download_transcript_record(
    session_id: ApiSessionId,
    service: Sessions,
    reference: str = Query(max_length=4096),
) -> StreamingResponse:
    try:
        record = await service.history_record(SessionId(session_id), PageToken(reference))
    except TranscriptError as error:
        raise _history_error(error, session_id) from None
    return RecordResponse(record)
