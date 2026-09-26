import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import psutil
from mandri.core.fs.paths import normalize_fs_path
from mandri.core.ids import HarnessKind
from mandri.sessions.ownership.file_lock import linux_lock_pids, writer_locked
from mandri.sessions.ownership.windows_lock import locking_processes


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    created_at: float
    dedicated: bool


def is_harness(name: str, command: list[str], harness: HarnessKind) -> bool:
    executable = os.path.basename(name).lower()
    if executable in {harness.value, f"{harness.value}.exe"}:
        return True
    if harness is HarnessKind.AGY:
        return executable in {"antigravity", "antigravity.exe"}
    if harness is HarnessKind.PI:
        return (
            executable in {"node", "node.exe", "bun", "bun.exe"}
            and len(command) > 1
            and command[1].replace("\\", "/").rsplit("/", 1)[-1] in {"pi", "pi.js"}
        ) or any(
            package in arg.replace("\\", "/").lower()
            for arg in command
            for package in ("/@mariozechner/pi-coding-agent/", "/@earendil-works/pi-coding-agent/")
        )
    package = {
        HarnessKind.CODEX: "/@openai/codex/",
        HarnessKind.CLAUDE: "/@anthropic-ai/claude-code/",
        HarnessKind.OPENCODE: "/opencode-ai/",
    }[harness]
    return any(package in arg.replace("\\", "/").lower() for arg in command)


def dedicated_command(command: list[str], harness: HarnessKind, native_id: str) -> bool:
    if "app-server" in command or "serve" in command or "web" in command:
        return False
    flags = {
        HarnessKind.CODEX: {"resume"},
        HarnessKind.CLAUDE: {"--resume", "-r", "--session-id"},
        HarnessKind.OPENCODE: {"--session", "-s"},
        HarnessKind.AGY: {"--conversation"},
        HarnessKind.PI: {"--session", "--session-id"},
    }[harness]
    return any(arg in flags and value == native_id for arg, value in pairwise(command)) or any(
        arg == f"{flag}={native_id}" for flag in flags for arg in command
    )


def lock_owners(path: Path) -> dict[int, float]:
    if sys.platform == "win32":
        return locking_processes(path)
    if sys.platform.startswith("linux"):
        return {pid: psutil.Process(pid).create_time() for pid in linux_lock_pids(path)}
    return {}


def codex_writer(path: Path, native_id: str) -> ProcessIdentity | None:
    try:
        owners = lock_owners(path)
        if len(owners) != 1:
            return None
        pid, started = next(iter(owners.items()))
        process = psutil.Process(pid)
        created_at = process.create_time()
        if abs(created_at - started) > 0.01:
            return None
        command = process.cmdline()
        dedicated = is_harness(process.name(), command, HarnessKind.CODEX) and dedicated_command(
            command, HarnessKind.CODEX, native_id
        )
        if dedicated:
            dedicated = _only_owned_thread(path, pid)
        return ProcessIdentity(pid, created_at, dedicated)
    except (OSError, psutil.Error):
        return None


def _only_owned_thread(path: Path, pid: int) -> bool:
    for other in path.parent.glob("*.lock"):
        if other == path or other.name.startswith("."):
            continue
        locked = writer_locked(other)
        if locked is None or (locked and pid in lock_owners(other)):
            return False
    return True


def session_processes(
    harness: HarnessKind,
    native_id: str,
    project_path: str,
    pi_processes: Mapping[int, str | None] | None = None,
    ignored_pids: frozenset[int] = frozenset(),
) -> tuple[list[ProcessIdentity], bool]:
    matches = []
    uncertain = False
    for process in psutil.process_iter(["name", "cmdline", "cwd", "create_time"], ad_value=None):
        if process.pid in ignored_pids:
            continue
        info = process.info
        command = info["cmdline"] or []
        if not is_harness(info["name"] or "", command, harness):
            continue
        if harness is HarnessKind.PI:
            known_id = (pi_processes or {}).get(process.pid)
            if known_id is not None:
                if known_id == native_id:
                    if info["create_time"] is None:
                        uncertain = True
                    else:
                        matches.append(ProcessIdentity(process.pid, info["create_time"], False))
                continue
            selected = pi_selected_session(command)
            if selected is not None and pi_session_matches(selected, native_id):
                if info["create_time"] is None:
                    uncertain = True
                else:
                    matches.append(ProcessIdentity(process.pid, info["create_time"], False))
                continue
        if harness is HarnessKind.AGY and dedicated_command(command, harness, native_id):
            if info["create_time"] is None:
                uncertain = True
            else:
                matches.append(ProcessIdentity(process.pid, info["create_time"], True))
            continue
        cwd = info["cwd"]
        if cwd is not None and normalize_fs_path(cwd) != normalize_fs_path(project_path):
            continue
        if info["create_time"] is None or not command:
            uncertain = True
            continue
        if harness is HarnessKind.PI:
            if "--no-session" not in command:
                uncertain = True
            continue
        if dedicated_command(command, harness, native_id):
            matches.append(ProcessIdentity(process.pid, info["create_time"], True))
        elif any(
            arg.split("=", 1)[0]
            in {"--resume", "-r", "--session", "-s", "--session-id", "--conversation"}
            for arg in command
        ):
            continue
        else:
            uncertain = True
    return matches, uncertain


def pi_selected_session(command: list[str]) -> str | None:
    for arg, value in pairwise(command):
        if arg in {"--session", "--session-id"}:
            return value
    for arg in command:
        if arg.startswith(("--session=", "--session-id=")):
            return arg.split("=", 1)[1]
    return None


def pi_session_matches(selected: str, native_id: str) -> bool:
    name = selected.replace("\\", "/").rsplit("/", 1)[-1]
    if (
        selected == native_id
        or name == f"{native_id}.jsonl"
        or name.endswith(f"_{native_id}.jsonl")
    ):
        return True
    if "/" not in selected and "\\" not in selected and not selected.endswith(".jsonl"):
        return native_id.startswith(selected)
    try:
        with Path(selected).expanduser().open("rb") as handle:
            entry = json.loads(handle.readline(1024 * 1024))
        return (
            isinstance(entry, dict)
            and entry.get("type") == "session"
            and entry.get("id") == native_id
        )
    except (OSError, ValueError, RecursionError):
        return False
