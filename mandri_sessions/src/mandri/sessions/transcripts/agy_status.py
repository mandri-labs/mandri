import json
from pathlib import Path

from mandri.sessions.agy_store import agy_roots
from mandri.sessions.transcripts.records import ReverseRecords
from mandri.sessions.transcripts.status import MAX_STATUS_RECORD_BYTES, MAX_STATUS_SCAN_BYTES


def agy_journal_status(
    root: Path, profiles_root: Path | None, native_id: str, transcript_modified_ns: int
) -> bool | None:
    newest = 0
    busy = None
    for profile in agy_roots(root, profiles_root):
        path = profile / "mandri-events.jsonl"
        try:
            stat = path.stat()
            if stat.st_mtime_ns < transcript_modified_ns:
                continue
            with path.open("rb") as handle:
                records = ReverseRecords(handle, stat.st_size, MAX_STATUS_SCAN_BYTES)
                for record in records:
                    if record.size > MAX_STATUS_RECORD_BYTES:
                        break
                    handle.seek(record.start)
                    event = json.loads(handle.read(record.size))
                    if not isinstance(event, dict):
                        break
                    stamp = event.get("_mandri_recorded_at_ns")
                    if not isinstance(stamp, int) or stamp < max(newest, transcript_modified_ns):
                        break
                    if event.get("event") == "init" and event.get("conversation_id") == native_id:
                        newest, busy = stamp, None
                        break
                    hook = event.get("hook") if event.get("event") == "hook" else event.get("event")
                    data = event.get("data") if event.get("event") == "hook" else event
                    if not isinstance(data, dict) or data.get("conversationId") != native_id:
                        continue
                    if hook == "Stop" and isinstance(data.get("fullyIdle"), bool):
                        newest, busy = stamp, not data["fullyIdle"]
                        break
                    if hook in ("PreInvocation", "PreToolUse"):
                        newest, busy = stamp, True
                        break
        except (OSError, ValueError):
            continue
    return busy
