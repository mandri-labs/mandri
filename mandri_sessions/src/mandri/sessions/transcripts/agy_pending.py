import sqlite3
from contextlib import closing
from pathlib import Path

from mandri.core.ports.transcripts import SessionRef
from mandri.sessions.agy_profiles import validate_agy_id
from mandri.sessions.agy_store import agy_metadata, agy_roots
from mandri.sessions.ownership.file_lock import writer_locked


def agy_history_pending(root: Path, profiles_root: Path | None, session: SessionRef) -> bool:
    if session.transcript_path is not None:
        return False
    native_id = validate_agy_id(str(session.native_id))
    metadata = agy_metadata(root, profiles_root).get(native_id, {})
    if metadata.get("is_mandri_root") is not True or metadata.get("history_pending") is not True:
        return False
    lock = root / "antigravity-cli/.mandri-locks" / f"{native_id}.lock"
    if writer_locked(lock) is not True:
        return False
    for source in agy_roots(root, profiles_root):
        database = source / "antigravity-cli/conversations" / f"{native_id}.db"
        if not database.is_file():
            continue
        try:
            with closing(
                sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
            ) as connection:
                if connection.execute("SELECT 1 FROM steps LIMIT 1").fetchone() is not None:
                    return False
        except sqlite3.Error:
            return False
    return True
