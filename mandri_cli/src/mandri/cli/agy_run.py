import os
import shlex
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mandri.cli.run_errors import RunError
from mandri.cli.types import RunSpec
from mandri.config.toml_adapter import TomlConfigAdapter
from mandri.core.ids import HarnessKind, SessionId
from mandri.core.types.availability import SessionOwner
from mandri.sessions.adapters.agy_sessions import AgySessionsAdapter
from mandri.sessions.agy_lease import AgyConversationLease
from mandri.sessions.agy_profiles import (
    default_agy_root,
    merge_agy_hooks,
    prepare_agy_profile,
    read_agy_json,
    validate_agy_id,
    write_agy_json,
)
from mandri.sessions.agy_store import agy_metadata
from mandri.sessions.errors import SessionRunningError
from mandri.sessions.ownership.service import inspect_owner


@dataclass(frozen=True)
class AgyRunProfile:
    root: Path
    args: tuple[str, ...]
    canonical_root: Path
    leases: dict[str, AgyConversationLease] = field(default_factory=dict)


def close_agy_run(profile: AgyRunProfile) -> None:
    try:
        hooks_path = profile.root / "config/hooks.json"
        hooks = read_agy_json(hooks_path)
        hooks.pop("mandri", None)
        write_agy_json(hooks_path, hooks)
    finally:
        for lease in profile.leases.values():
            lease.release()


def wait_agy_process(process: subprocess.Popen[bytes], profile: AgyRunProfile) -> int:
    try:
        while True:
            binding = read_agy_json(profile.root / "mandri-session.json")
            native_id = binding.get("native_id")
            if isinstance(native_id, str) and native_id not in profile.leases:
                lease = _claim_conversation(profile.canonical_root, native_id)
                profile.leases[native_id] = lease
            try:
                return process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise


def prepare_agy_run(spec: RunSpec) -> AgyRunProfile:
    config = TomlConfigAdapter(spec.base_dir).load().sessions
    canonical = Path(config.agy_home) if config.agy_home else default_agy_root()
    profiles = (
        Path(config.agy_profiles_dir) if config.agy_profiles_dir else spec.base_dir / "agy-profiles"
    )
    cwd = str((spec.cwd or Path.cwd()).resolve())
    native_id, passthrough = _resume_arguments(spec.passthrough_args, canonical, cwd)
    lease = _claim_conversation(canonical, native_id) if native_id is not None else None
    try:
        if native_id is not None:
            if not AgySessionsAdapter(canonical, profiles).exists(SessionId(native_id)):
                raise RunError(f"Antigravity conversation not found: {native_id}")
            owner = inspect_owner(HarnessKind.AGY, native_id, cwd)
            if owner.owner is not SessionOwner.UNOWNED:
                raise RunError(
                    "Antigravity conversation already has a writer or ownership is uncertain"
                )
        result = _configure_profile(spec, canonical, profiles, cwd, native_id, passthrough)
    except BaseException:
        if lease:
            lease.release()
        raise
    if lease and native_id:
        result.leases[native_id] = lease
    return result


def _claim_conversation(canonical: Path, native_id: str) -> AgyConversationLease:
    lease = AgyConversationLease(canonical, native_id)
    try:
        lease.acquire()
    except SessionRunningError as error:
        raise RunError(str(error)) from error
    return lease


def _configure_profile(
    spec: RunSpec,
    canonical: Path,
    profiles: Path,
    cwd: str,
    native_id: str | None,
    passthrough: tuple[str, ...],
) -> AgyRunProfile:
    profile = prepare_agy_profile(
        profiles, native_id or str(uuid.uuid4()), native=False, canonical_root=canonical
    )
    previous = agy_metadata(canonical, profiles).get(native_id, {}) if native_id else {}
    context: dict[str, Any] = {
        "cwd": cwd,
        "model": spec.model_arg,
        "model_source": "gateway",
        "new_conversation": native_id is None,
    }
    if previous.get("is_mandri_root") is True:
        context["is_mandri_root"] = True
    if native_id is not None:
        context["native_id"] = native_id
    write_agy_json(profile / "mandri-launch.json", context)
    if native_id is not None:
        write_agy_json(profile / "mandri-session.json", context)
    hooks = {}
    for event in ("PreInvocation", "PostInvocation", "Stop"):
        command = [sys.executable, "-m", "mandri.sessions.agy_observe", str(profile), event]
        serialized = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
        hooks[event] = [{"type": "command", "command": serialized, "timeout": 5}]
    merge_agy_hooks(profile, hooks)
    return AgyRunProfile(
        profile, ("--gemini_dir", str(profile), "--add-dir", cwd, *passthrough), canonical
    )


def _resume_arguments(
    args: tuple[str, ...], canonical: Path, cwd: str
) -> tuple[str | None, tuple[str, ...]]:
    native_id = None
    passthrough = []
    iterator = iter(args)
    for arg in iterator:
        flag, equals, value = arg.partition("=")
        if flag in ("--gemini_dir", "--model"):
            raise RunError(f"{flag} is managed by Mandri; select the model with mandri run --model")
        if flag == "--conversation":
            native_id = value if equals else next(iterator, "")
            try:
                validate_agy_id(native_id)
            except ValueError as error:
                raise RunError(str(error)) from error
        elif flag in ("--continue", "-c"):
            last = read_agy_json(canonical / "antigravity-cli/cache/last_conversations.json")
            candidate = last.get(cwd)
            if not isinstance(candidate, str):
                raise RunError("No Antigravity conversation to continue; use --conversation <id>")
            native_id = validate_agy_id(candidate)
        else:
            passthrough.append(arg)
    if native_id:
        passthrough.extend(("--conversation", native_id))
    return native_id, tuple(passthrough)
