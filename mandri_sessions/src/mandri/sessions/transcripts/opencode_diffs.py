import contextlib
import functools
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

_TREE_ID = re.compile(r"[0-9a-f]{40,64}\Z")


def turn_files(database: Path, session_id: str, message_id: str) -> dict[str, Any] | None:
    uri = f"{database.resolve().as_uri()}?mode=ro"
    with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=2)) as db:
        boundary = db.execute(
            "SELECT seq FROM session_message WHERE id=? AND session_id=? AND type='idle'",
            (message_id, session_id),
        ).fetchone()
        if boundary is None:
            return None
        previous = db.execute(
            "SELECT coalesce(max(seq),-1) FROM session_message"
            " WHERE session_id=? AND type='idle' AND seq<?",
            (session_id, boundary[0]),
        ).fetchone()[0]
        user = db.execute(
            "SELECT id FROM session_message WHERE session_id=? AND type='user'"
            " AND seq>? AND seq<? ORDER BY seq LIMIT 1",
            (session_id, previous, boundary[0]),
        ).fetchone()
        project = db.execute(
            "SELECT project_id FROM session_v2 WHERE id=?", (session_id,)
        ).fetchone()
        project_id = project[0] if project and isinstance(project[0], str) else None
        if (
            user is None
            or not project_id
            or project_id in {".", ".."}
            or Path(project_id).name != project_id
        ):
            return None
        if db.execute(
            "SELECT 1 FROM session_message WHERE session_id=? AND type='location-switched'"
            " AND seq>? AND seq<? LIMIT 1",
            (session_id, previous, boundary[0]),
        ).fetchone():
            return None
        snapshots = db.execute(
            "SELECT json_extract(data,'$.snapshot.start'),json_extract(data,'$.snapshot.end')"
            " FROM session_message WHERE session_id=? AND type='assistant' AND seq>? AND seq<?"
            " ORDER BY seq",
            (session_id, previous, boundary[0]),
        ).fetchall()
    start = next((row[0] for row in snapshots if row[0]), None)
    end = next((row[1] for row in reversed(snapshots) if row[1]), None)
    if not isinstance(start, str) or not isinstance(end, str):
        return None
    if not _TREE_ID.fullmatch(start) or not _TREE_ID.fullmatch(end):
        return None
    root = database.parent / "snapshot" / project_id
    if not root.is_dir():
        return None
    for repository in root.iterdir():
        if not (repository / "HEAD").is_file():
            continue
        files = snapshot_files(repository, start, end)
        if files is not None:
            return {"type": "file_snapshot", "turnID": user[0], "files": files}
    return None


def _git(repository: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "--git-dir", str(repository), "-c", "core.fsmonitor=false", *arguments],
        capture_output=True,
        check=True,
        timeout=5,
    )
    return result.stdout


def snapshot_files(repository: Path, start: str, end: str) -> list[dict[str, Any]] | None:
    try:
        return _snapshot_files(repository, start, end)
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None


@functools.lru_cache(maxsize=128)
def _snapshot_files(repository: Path, start: str, end: str) -> list[dict[str, Any]]:
    entries = _git(
        repository,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--name-status",
        "-z",
        start,
        end,
    ).split(b"\0")
    files = []
    for index in range(0, len(entries) - 1, 2):
        status = entries[index].decode("ascii")
        path = entries[index + 1].decode("utf-8", errors="replace")
        patch = _git(
            repository,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--no-color",
            "--unified=3",
            start,
            end,
            "--",
            path,
        ).decode("utf-8", errors="replace")
        lines = patch.splitlines()
        files.append(
            {
                "file": path,
                "patch": patch,
                "status": {"A": "added", "D": "deleted"}.get(status, "modified"),
                "additions": sum(
                    line.startswith("+") and not line.startswith("+++") for line in lines
                ),
                "deletions": sum(
                    line.startswith("-") and not line.startswith("---") for line in lines
                ),
            }
        )
    return files
