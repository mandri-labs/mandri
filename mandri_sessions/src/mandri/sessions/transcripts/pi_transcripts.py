from pathlib import Path

from mandri.core.ids import PageToken
from mandri.core.ports.transcripts import SessionRef, TranscriptPage
from mandri.sessions.native_activity import native_model, native_turn_busy
from mandri.sessions.pi_leaf import read_pi_leaf
from mandri.sessions.pi_store import PiSessionStore
from mandri.sessions.transcripts.errors import TranscriptNotFoundError, TranscriptStoreError
from mandri.sessions.transcripts.pi_branches import PiBranchIndex
from mandri.sessions.transcripts.pi_pages import pi_page
from mandri.sessions.transcripts.record_download import RecordDownload, open_record
from mandri.sessions.transcripts.status import MAX_STATUS_RECORD_BYTES, MAX_STATUS_RECORDS


class PiTranscriptReader:
    def __init__(self, sessions_root: Path | None = None, *, store: PiSessionStore | None = None):
        self.store = store or PiSessionStore(sessions_root)
        self._branches = PiBranchIndex()

    def _resolve(self, session: SessionRef) -> Path:
        path = (
            Path(str(session.transcript_path))
            if session.transcript_path is not None
            else self.store.resolve(str(session.native_id))
        )
        if path is None or not path.is_file():
            raise TranscriptNotFoundError(f"No Pi transcript for {session.native_id}")
        return path

    def page(self, session: SessionRef, cursor: PageToken | None, limit: int) -> TranscriptPage:
        return self._page(session, cursor, limit, recent=False)

    def recent(self, session: SessionRef, cursor: PageToken | None, limit: int) -> TranscriptPage:
        return self._page(session, cursor, limit, recent=True)

    def _page(
        self, session: SessionRef, cursor: PageToken | None, limit: int, *, recent: bool
    ) -> TranscriptPage:
        path = self._resolve(session)
        try:
            return pi_page(path, self._branches.spans(path), cursor, limit, recent=recent)
        except OSError as error:
            raise TranscriptStoreError(f"Cannot read Pi transcript: {error}") from error

    def status(self, session: SessionRef) -> tuple[bool | None, str | None]:
        path = self._resolve(session)
        busy, model = None, None
        try:
            spans = self._branches.spans(path)
            with path.open("rb") as handle:
                for span in reversed(spans[-MAX_STATUS_RECORDS:]):
                    if span.size > MAX_STATUS_RECORD_BYTES:
                        break
                    handle.seek(span.start)
                    line = handle.read(span.size).decode("utf-8")
                    if busy is None:
                        busy = native_turn_busy(session.harness, [line])
                    if model is None:
                        model = native_model([line])
                    if busy is not None and model is not None:
                        break
        except OSError as error:
            raise TranscriptStoreError(f"Cannot inspect Pi transcript: {error}") from error
        return busy, model

    def record(self, session: SessionRef, reference: PageToken) -> RecordDownload:
        return open_record(self._resolve(session), reference)

    def revision(self, session: SessionRef) -> tuple[str, int, int, str | None]:
        path = self._resolve(session)
        stat = path.stat()
        leaf = read_pi_leaf(
            path, str(session.native_id), (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        )
        modified = max(stat.st_mtime_ns, leaf.modified_at) if leaf else stat.st_mtime_ns
        return str(path), modified, stat.st_size, leaf.leaf_id if leaf else None
