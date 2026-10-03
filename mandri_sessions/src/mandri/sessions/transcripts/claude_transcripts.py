"""Claude JSONL transcript reader over the claude projects store."""

from pathlib import Path
from typing import Any

from claude_agent_sdk._internal.sessions import _sanitize_path
from mandri.core.ids import PageToken
from mandri.core.ports.transcripts import SessionRef, TranscriptPage
from mandri.core.types.conversation_status import WorkDelta
from mandri.sessions.transcripts.errors import TranscriptNotFoundError
from mandri.sessions.transcripts.jsonl import page_from_jsonl
from mandri.sessions.transcripts.recent import recent_jsonl
from mandri.sessions.transcripts.record_download import RecordDownload, open_record
from mandri.sessions.transcripts.status import jsonl_status
from mandri.sessions.transcripts.work_delta import jsonl_work_delta


class ClaudeTranscriptReader:
    def __init__(self, transcripts_root: Path) -> None:
        self._transcripts_root = Path(transcripts_root)

    def page(self, session: SessionRef, cursor: PageToken | None, limit: int) -> TranscriptPage:
        return page_from_jsonl(self._resolve(session), cursor, limit)

    def _resolve(self, session: SessionRef) -> Path:
        if session.transcript_path is not None:
            path = Path(str(session.transcript_path))
            if not path.is_file():
                raise TranscriptNotFoundError(f"claude transcript missing: {path.name}")
            return path
        if session.project_path is None:
            raise TranscriptNotFoundError("claude session without project or transcript path")
        path = (
            self._transcripts_root
            / _sanitize_path(str(session.project_path))
            / f"{session.native_id}.jsonl"
        )
        if path.is_file():
            return path
        matches = sorted(self._transcripts_root.rglob(f"{session.native_id}.jsonl"))
        if not matches:
            raise TranscriptNotFoundError(f"no claude transcript for {session.native_id}")
        return matches[0]

    def recent(self, session: SessionRef, cursor: PageToken | None, limit: int) -> TranscriptPage:
        return recent_jsonl(self._resolve(session), cursor, limit)

    def status(self, session: SessionRef) -> tuple[bool | None, str | None]:
        return jsonl_status(self._resolve(session), session.harness)

    def record(self, session: SessionRef, reference: PageToken) -> RecordDownload:
        return open_record(self._resolve(session), reference)

    def work_delta(self, session: SessionRef, checkpoint: dict[str, Any] | None) -> WorkDelta:
        return jsonl_work_delta(self._resolve(session), session, checkpoint)

    def revision(self, session: SessionRef) -> tuple[str, int, int]:
        path = self._resolve(session)
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_size
