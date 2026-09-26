import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from mandri.sessions.agy_profiles import read_agy_json, validate_agy_id
from mandri.sessions.errors import DatabaseAccessError


def agy_roots(root: Path, profiles_root: Path | None = None) -> list[Path]:
    profiles = sorted(profiles_root.iterdir()) if profiles_root and profiles_root.is_dir() else []
    return [root, *(path for path in profiles if path.is_dir())]


def agy_summary_rows(root: Path) -> list[dict[str, Any]]:
    database = root / "antigravity-cli/conversation_summaries.db"
    if not database.is_file():
        return []
    try:
        with closing(
            sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
        ) as db:
            db.row_factory = sqlite3.Row
            tables = db.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            for row in tables:
                table = '"' + row[0].replace('"', '""') + '"'
                columns = {column[1] for column in db.execute(f"PRAGMA table_info({table})")}
                if "conversation_id" in columns and "last_modified_time" in columns:
                    selected = columns & {
                        "conversation_id",
                        "last_modified_time",
                        "title",
                        "preview",
                        "workspace_uris",
                        "parent_conversation_id",
                        "nesting_depth",
                        "status",
                        "agent_name",
                        "not_fully_idle",
                        "killed",
                        "created_at",
                    }
                    projection = ", ".join(sorted(selected))
                    return [dict(item) for item in db.execute(f"SELECT {projection} FROM {table}")]
    except sqlite3.Error as error:
        raise DatabaseAccessError("Cannot read Antigravity conversation summaries") from error
    return []


def agy_metadata(root: Path, profiles_root: Path | None = None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in agy_roots(root, profiles_root):
        try:
            summaries = agy_summary_rows(source)
        except DatabaseAccessError:
            summaries = []
        for row in summaries:
            native_id = row.get("conversation_id")
            if isinstance(native_id, str) and _valid_id(native_id):
                previous = result.setdefault(native_id, {})
                if epoch_ms(row.get("last_modified_time")) >= epoch_ms(
                    previous.get("last_modified_time")
                ):
                    previous.update({key: value for key, value in row.items() if value is not None})
        _merge_cache(source, result)
        _merge_binding(source, result)
    return result


def _merge_cache(root: Path, result: dict[str, dict[str, Any]]) -> None:
    cache = root / "antigravity-cli/cache"
    try:
        metadata = read_agy_json(cache / "conversation_metadata.json")
        last = read_agy_json(cache / "last_conversations.json")
    except (OSError, ValueError):
        return
    for native_id, value in metadata.items():
        if _valid_id(native_id) and isinstance(value, dict):
            row = result.setdefault(native_id, {})
            for key, item in value.items():
                row.setdefault(key, item)
    for workspace, native_id in last.items():
        if isinstance(native_id, str) and _valid_id(native_id):
            result.setdefault(native_id, {}).setdefault("cwd", workspace)


def _merge_binding(root: Path, result: dict[str, dict[str, Any]]) -> None:
    paths = [*(root / "bindings").glob("*.json"), root / "mandri-session.json"]
    for path in paths:
        _read_binding(path, result)


def _read_binding(path: Path, result: dict[str, dict[str, Any]]) -> None:
    try:
        binding = read_agy_json(path)
    except (OSError, ValueError):
        return
    native_id = binding.get("native_id")
    if isinstance(native_id, str) and _valid_id(native_id):
        row = result.setdefault(native_id, {})
        modified = path.stat().st_mtime_ns
        if modified >= row.get("_binding_mtime_ns", 0):
            row.update(binding)
            row["_binding_mtime_ns"] = modified


def _valid_id(value: str) -> bool:
    try:
        validate_agy_id(value)
    except ValueError:
        return False
    return True


def epoch_ms(value: Any) -> int:
    if isinstance(value, (float, int)):
        return int(value * 1000 if 0 < value < 100_000_000_000 else value)
    if isinstance(value, str) and value:
        try:
            return epoch_ms(float(value))
        except ValueError:
            try:
                return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
            except ValueError:
                return 0
    if isinstance(value, dict):
        return int(value.get("seconds", 0)) * 1000 + int(value.get("nanos", 0)) // 1_000_000
    return 0


def workspace_path(row: dict[str, Any]) -> str:
    raw = row.get("cwd") or row.get("workspace_uris") or row.get("workspacePaths") or []
    if isinstance(raw, str) and raw.startswith("["):
        try:
            raw = json.loads(raw)
        except ValueError:
            return ""
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    if not isinstance(raw, str):
        return ""
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        path = unquote(parsed.path)
        if parsed.netloc and parsed.netloc != "localhost":
            return "//" + parsed.netloc + path
        return path[1:] if len(path) > 2 and path[2] == ":" else path
    return raw


def transcript_path(root: Path, native_id: str) -> Path | None:
    logs = root / "antigravity-cli/brain" / validate_agy_id(native_id) / ".system_generated/logs"
    for name in ("transcript_full.jsonl", "transcript.jsonl"):
        candidate = logs / name
        if candidate.is_file():
            return candidate
    return None


def transcript_head(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    result: dict[str, Any] = {}
    try:
        with path.open("rb") as handle:
            for _ in range(32):
                line = handle.readline(262145)
                if not line or not line.endswith(b"\n"):
                    break
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(item, dict):
                    continue
                if "created_at" in item:
                    result.setdefault("created_at", item["created_at"])
                if item.get("type") == "USER_INPUT" and isinstance(item.get("content"), str):
                    text = item["content"]
                    if text.startswith("<USER_REQUEST>"):
                        text = text.removeprefix("<USER_REQUEST>").split("</USER_REQUEST>", 1)[0]
                    result["title"] = text.strip().split("\n", 1)[0][:120]
                    break
    except OSError:
        return result
    return result
