import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from mandri.core.ids import SessionId
from mandri.runtime.control.agy_bridge import AgyBridge
from mandri.runtime.control.agy_policy import AgyPolicy
from mandri.runtime.control.modes import agy_launch_args
from mandri.runtime.errors import SessionNotResumableError
from mandri.runtime.launch_preparation import PreparedLaunch
from mandri.sessions.adapters.agy_sessions import AgySessionsAdapter
from mandri.sessions.agy_lease import AgyConversationLease
from mandri.sessions.agy_profiles import (
    configure_agy_model,
    default_agy_root,
    merge_agy_hooks,
    prepare_agy_profile,
    read_agy_json,
    write_agy_json,
)
from mandri.sessions.agy_store import agy_metadata
from mandri.sessions.errors import SessionConflictError


class AgyLaunch:
    def __init__(
        self,
        profile: Path,
        bridge: AgyBridge,
        metadata: dict[str, Any],
        canonical_root: Path,
        lease: AgyConversationLease | None = None,
    ) -> None:
        self.profile = profile
        self.bridge = bridge
        self.metadata = metadata
        self.canonical_root = canonical_root
        self._lease = lease

    def bind(self, native_id: str) -> None:
        expected = self.metadata.get("native_id")
        if expected is not None and expected != native_id:
            raise SessionConflictError("Antigravity resumed a different conversation")
        if self._lease is None:
            lease = AgyConversationLease(self.canonical_root, native_id)
            lease.acquire()
            self._lease = lease
        self.metadata["native_id"] = native_id
        write_agy_json(self.profile / "mandri-session.json", self.metadata)

    def record(self, event: dict[str, Any]) -> None:
        with (self.profile / "mandri-events.jsonl").open("a", encoding="utf-8") as journal:
            record = {**event, "_mandri_recorded_at_ns": time.time_ns()}
            journal.write(json.dumps(record, ensure_ascii=False) + "\n")
        step = event.get("step_update")
        if (
            self.metadata.get("history_pending") is True
            and self.metadata.get("native_id") is not None
            and event.get("event") == "step_update"
            and isinstance(step, dict)
            and (step.get("conversation_id") or event.get("conversation_id"))
            in (None, self.metadata["native_id"])
        ):
            self.metadata["history_pending"] = False
            write_agy_json(self.profile / "mandri-session.json", self.metadata)

    async def aclose(self) -> None:
        try:
            await self.bridge.aclose()
            path = self.profile / "config/hooks.json"
            hooks = read_agy_json(path)
            hooks.pop("mandri", None)
            write_agy_json(path, hooks)
        finally:
            if self._lease is not None:
                self._lease.release()
                self._lease = None


def prepare_agy_launch(
    prepared: PreparedLaunch,
    session_id: str,
    cwd: Path,
    profiles: Path,
    canonical: Path | None,
    native: bool,
    model: str,
    mode: str,
    gateway_port: int,
    timeout: int,
    publish: Callable[[dict[str, Any]], Awaitable[None]],
) -> tuple[PreparedLaunch, AgyLaunch]:
    canonical = canonical or default_agy_root()
    native_id = _conversation_id(prepared.argv)
    lease = AgyConversationLease(canonical, native_id) if native_id else None
    if lease:
        lease.acquire()
    try:
        if native_id and not AgySessionsAdapter(canonical, profiles).exists(SessionId(native_id)):
            raise SessionNotResumableError("Antigravity conversation no longer exists")
        return _prepare_profile(
            prepared,
            session_id,
            cwd,
            profiles,
            canonical,
            native,
            model,
            mode,
            gateway_port,
            timeout,
            publish,
            lease,
            native_id,
        )
    except BaseException:
        if lease:
            lease.release()
        raise


def _prepare_profile(
    prepared: PreparedLaunch,
    session_id: str,
    cwd: Path,
    profiles: Path,
    canonical: Path,
    native: bool,
    model: str,
    mode: str,
    gateway_port: int,
    timeout: int,
    publish: Callable[[dict[str, Any]], Awaitable[None]],
    lease: AgyConversationLease | None,
    native_id: str | None,
) -> tuple[PreparedLaunch, AgyLaunch]:
    profile = prepare_agy_profile(profiles, session_id, native=native, canonical_root=canonical)
    if not native:
        configure_agy_model(profile, prepared.env)
    settings = read_agy_json(profile / "antigravity-cli/settings.json")
    permissions = settings.get("permissions", {})
    if not isinstance(permissions, dict):
        raise ValueError("Antigravity permissions must be an object")

    async def publish_record(event: dict[str, Any]) -> None:
        resource.record(event)
        await publish(event)

    bridge = AgyBridge(AgyPolicy(mode, cwd, permissions), publish_record, timeout)
    hooks: dict[str, Any] = {}
    for event in ("PreToolUse", "PostToolUse", "PreInvocation", "PostInvocation", "Stop"):
        argv = [sys.executable, "-m", "mandri.runtime.agy_hook", event]
        command = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)
        handler = {"type": "command", "command": command, "timeout": timeout + 10}
        hooks[event] = (
            [{"matcher": "*", "hooks": [handler]}]
            if event in {"PreToolUse", "PostToolUse"}
            else [handler]
        )
    merge_agy_hooks(profile, hooks)
    env = {
        **prepared.env,
        "MANDRI_AGY_HOOK_URL": f"http://127.0.0.1:{gateway_port}/v1/runtime/agy/{session_id}/hook",
        "MANDRI_AGY_HOOK_TOKEN": bridge.token,
        "MANDRI_AGY_HOOK_TIMEOUT": str(timeout + 5),
    }
    argv = [
        *prepared.argv,
        "--gemini_dir",
        str(profile),
        "--add-dir",
        str(cwd),
        "--dangerously-skip-permissions",
        *agy_launch_args(mode),
    ]
    previous = agy_metadata(canonical, profiles).get(native_id, {}) if native_id else {}
    metadata: dict[str, Any] = {
        "cwd": str(cwd),
        "model": model,
        "model_source": "native" if native else "gateway",
        "is_mandri_root": native_id is None or previous.get("is_mandri_root") is True,
        "history_pending": native_id is None,
    }
    if native_id:
        metadata["native_id"] = native_id
    write_agy_json(profile / "mandri-session.json", metadata)
    resource = AgyLaunch(profile, bridge, metadata, canonical, lease)
    return PreparedLaunch(argv, env, prepared.listen_port), resource


def _conversation_id(argv: list[str]) -> str | None:
    for index, argument in enumerate(argv):
        if argument.startswith("--conversation="):
            return argument.split("=", 1)[1]
        if argument == "--conversation" and index + 1 < len(argv):
            return argv[index + 1]
    return None
