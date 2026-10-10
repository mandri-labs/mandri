from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from mandri.runtime.agy_launch import AgyLaunch
from mandri.runtime.control.agy_bridge import AgyBridge
from mandri.runtime.control.agy_policy import AgyPolicy
from mandri.runtime.control.modes import agy_launch_args
from mandri.runtime.docker_config import NATIVE_HOME, WORKSPACE_ROOT
from mandri.runtime.docker_ingress import WorkerIngress
from mandri.runtime.launch_preparation import PreparedLaunch
from mandri.sessions.agy_profiles import (
    configure_agy_model,
    merge_agy_hooks,
    read_agy_json,
    write_agy_json,
)


def prepare_docker_agy(
    prepared: PreparedLaunch,
    session_id: str,
    workspace: Path,
    state: Path,
    model: str,
    mode: str,
    timeout: int,
    ingress: WorkerIngress,
    ingress_url: str,
    publish: Callable[[dict[str, Any]], Awaitable[None]],
    native_id: str | None = None,
) -> tuple[PreparedLaunch, AgyLaunch]:
    profile = state / ".gemini"
    settings = read_agy_json(profile / "antigravity-cli/settings.json")
    settings.update(enableTelemetry=False, useG1Credits=False, modelProvider="gemini")
    custom = settings.setdefault("customModelsConfig", {}).setdefault("customModels", {})
    custom["mandri"] = {**custom.get("mandri", {}), "modelName": "mandri-route"}
    write_agy_json(profile / "antigravity-cli/settings.json", settings)
    configure_agy_model(profile, prepared.env)
    for name in ("conversations", "brain"):
        (profile / "antigravity-cli" / name).mkdir(parents=True, exist_ok=True)

    async def publish_record(event: dict[str, Any]) -> None:
        resource.record(event)
        await publish(event)

    policy = AgyPolicy(
        mode, workspace, settings.get("permissions", {}), runtime_root=Path(WORKSPACE_ROOT)
    )
    bridge = AgyBridge(policy, publish_record, timeout)
    hooks: dict[str, Any] = {}
    for event in ("PreToolUse", "PostToolUse", "PreInvocation", "PostInvocation", "Stop"):
        handler = {
            "type": "command",
            "command": f"python3 /opt/mandri/agy_hook.py {event}",
            "timeout": timeout + 10,
        }
        hooks[event] = (
            [{"matcher": "*", "hooks": [handler]}]
            if event in {"PreToolUse", "PostToolUse"}
            else [handler]
        )
    merge_agy_hooks(profile, hooks)
    ingress.set_hook(session_id, bridge.token)
    metadata: dict[str, Any] = {
        "cwd": str(workspace),
        "model": model,
        "model_source": "gateway",
        "is_mandri_root": True,
        "history_pending": native_id is None,
    }
    if native_id:
        metadata["native_id"] = native_id
    write_agy_json(profile / "mandri-session.json", metadata)
    resource = AgyLaunch(profile, bridge, metadata, profile)
    env = {
        **prepared.env,
        "MANDRI_AGY_HOOK_URL": f"{ingress_url}/v1/runtime/agy/{session_id}/hook",
        "MANDRI_AGY_HOOK_TOKEN": bridge.token,
        "MANDRI_AGY_HOOK_TIMEOUT": str(timeout + 5),
    }
    argv = [
        *prepared.argv,
        "--gemini_dir",
        f"{NATIVE_HOME}/.gemini",
        "--add-dir",
        WORKSPACE_ROOT,
        "--dangerously-skip-permissions",
        *agy_launch_args(mode),
    ]
    return PreparedLaunch(argv, env, prepared.listen_port), resource
