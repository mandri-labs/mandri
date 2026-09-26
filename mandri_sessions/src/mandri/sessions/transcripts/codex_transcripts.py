"""Codex rollout JSONL transcript reader over the codex sessions store."""

import sqlite3
from pathlib import Path

from mandri.core.ids import PageToken
from mandri.core.ports.transcripts import SessionRef, TranscriptPage
from mandri.sessions.transcripts.errors import TranscriptNotFoundError, TranscriptStoreError
from mandri.sessions.transcripts.jsonl import page_from_jsonl
from mandri.sessions.transcripts.recent import recent_jsonl
from mandri.sessions.transcripts.record_download import RecordDownload, open_record
from mandri.sessions.transcripts.status import jsonl_status


class CodexTranscriptReader:
    def __init__(self, sessions_root: Path, state_db: Path | None = None) -> None:
        self._sessions_root = Path(sessions_root)
        self._state_db = state_db or self._sessions_root.parent / "state_5.sqlite"

    def writer_lock_path(self, session: SessionRef) -> Path | None:
        native_id = str(session.native_id)
        if not native_id or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for char in native_id
        ):
            return None
        if not self._sessions_root.is_dir():
            return None
        return self._sessions_root.parent / "thread-writer-locks" / f"{native_id}.lock"

    def page(self, session: SessionRef, cursor: PageToken | None, limit: int) -> TranscriptPage:
        return page_from_jsonl(self._resolve(session), cursor, limit)

    def _resolve(self, session: SessionRef) -> Path:
        native_id = str(session.native_id)
        if session.transcript_path is not None:
            return Path(str(session.transcript_path))
        canonical = self._canonical_path(native_id)
        if canonical is not None:
            return canonical
        if not self._sessions_root.is_dir():
            raise TranscriptNotFoundError(f"codex sessions root missing: {self._sessions_root}")
        matches = [
            path
            for path in self._sessions_root.rglob(f"rollout-*{native_id}*.jsonl")
            if path.stem.endswith(native_id) or f"{native_id}_" in path.stem
        ]
        if not matches:
            raise TranscriptNotFoundError(f"no codex rollout for {native_id}")
        return max(matches, key=lambda path: (path.stat().st_mtime_ns, path.name))

    def recent(self, session: SessionRef, cursor: PageToken | None, limit: int) -> TranscriptPage:
        return recent_jsonl(self._resolve(session), cursor, limit)

    def status(self, session: SessionRef) -> tuple[bool | None, str | None]:
        return jsonl_status(self._resolve(session), session.harness)

    def record(self, session: SessionRef, reference: PageToken) -> RecordDownload:
        return open_record(self._resolve(session), reference)

    def revision(self, session: SessionRef) -> tuple[str, int, int]:
        path = self._resolve(session)
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_size

    def _canonical_path(self, native_id: str) -> Path | None:
        if not self._state_db.is_file():
            return None
        try:
            connection = sqlite3.connect(
                f"{self._state_db.resolve().as_uri()}?mode=ro", uri=True, timeout=2.0
            )
            try:
                columns = {row[1] for row in connection.execute("PRAGMA table_info(threads)")}
                if "rollout_path" not in columns:
                    return None
                row = connection.execute(
                    "SELECT rollout_path FROM threads WHERE id = ?", (native_id,)
                ).fetchone()
            finally:
                connection.close()
        except sqlite3.Error as error:
            raise TranscriptStoreError(f"cannot resolve codex rollout: {error}") from error
        return Path(row[0]) if row and row[0] else None
