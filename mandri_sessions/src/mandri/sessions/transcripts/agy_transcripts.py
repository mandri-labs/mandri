from pathlib import Path

from mandri.core.ids import PageToken
from mandri.core.ports.transcripts import SessionRef, TranscriptPage
from mandri.sessions.agy_store import agy_roots, transcript_path
from mandri.sessions.transcripts.agy_pending import agy_history_pending
from mandri.sessions.transcripts.agy_status import agy_journal_status
from mandri.sessions.transcripts.errors import TranscriptNotFoundError
from mandri.sessions.transcripts.jsonl import page_from_jsonl
from mandri.sessions.transcripts.recent import recent_jsonl
from mandri.sessions.transcripts.record_download import RecordDownload, open_record
from mandri.sessions.transcripts.status import jsonl_status


class AgyTranscriptReader:
    def __init__(self, root: Path, profiles_root: Path | None = None) -> None:
        self.root = root
        self.profiles_root = profiles_root

    def _resolve(self, session: SessionRef) -> Path:
        if session.transcript_path is not None:
            explicit = Path(str(session.transcript_path))
            if explicit.is_file():
                return explicit
        for root in agy_roots(self.root, self.profiles_root):
            path = transcript_path(root, str(session.native_id))
            if path is not None:
                return path
        raise TranscriptNotFoundError(f"No Antigravity transcript for {session.native_id}")

    def page(self, session: SessionRef, cursor: PageToken | None, limit: int) -> TranscriptPage:
        path = self._history_path(session, cursor)
        return page_from_jsonl(path, cursor, limit) if path else TranscriptPage([], None, False)

    def recent(self, session: SessionRef, cursor: PageToken | None, limit: int) -> TranscriptPage:
        path = self._history_path(session, cursor)
        return recent_jsonl(path, cursor, limit) if path else TranscriptPage([], None, False)

    def _history_path(self, session: SessionRef, cursor: PageToken | None) -> Path | None:
        try:
            return self._resolve(session)
        except TranscriptNotFoundError:
            if cursor is None and agy_history_pending(self.root, self.profiles_root, session):
                return None
            raise

    def record(self, session: SessionRef, reference: PageToken) -> RecordDownload:
        return open_record(self._resolve(session), reference)

    def revision(self, session: SessionRef) -> tuple[str, int, int]:
        path = self._history_path(session, None)
        if path is None:
            return f"agy-pending:{session.native_id}", 0, 0
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_size

    def status(self, session: SessionRef) -> tuple[bool | None, str | None]:
        path = self._history_path(session, None)
        if path is None:
            return None, None
        busy, model = jsonl_status(path, session.harness)
        observed = agy_journal_status(
            self.root, self.profiles_root, str(session.native_id), path.stat().st_mtime_ns
        )
        return (observed if observed is not None else busy), model
