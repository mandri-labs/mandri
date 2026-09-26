import os
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from mandri.core.ids import HarnessKind
from mandri.core.protocol.commands import CommandCatalogParams
from mandri.core.types.config import SessionsConfig
from mandri.runtime import wiring
from mandri.runtime.command_catalogs import CommandCatalogCache
from mandri.runtime.control.errors import ControlError
from mandri.runtime.docker_backend import DockerBackend
from mandri.runtime.docker_command_catalog import DockerCommandDiscovery
from mandri.runtime.native_command_catalog import discover_commands
from mandri.runtime.native_launch import native_launch
from mandri.runtime.process import ManagedProcess
from mandri.sessions.agy_profiles import prepare_agy_profile


def create_command_catalog_cache(
    commands: Mapping[str, list[str]],
    sessions: SessionsConfig,
    spawn: Callable[..., Awaitable[ManagedProcess]],
    *,
    default_cwd: str,
    profiles_dir: Path,
    docker_backend: DockerBackend | None = None,
    docker_commands: Mapping[str, list[str]] | None = None,
) -> CommandCatalogCache:
    docker_probe = DockerCommandDiscovery(docker_backend) if docker_backend is not None else None

    async def discover(scope: CommandCatalogParams) -> list[dict[str, Any]]:
        if scope.execution_backend == "docker":
            if docker_probe is None:
                raise ControlError("Docker command discovery is not configured")
            command = list((docker_commands or {}).get(scope.harness, []))
            return await docker_probe.discover(
                HarnessKind(scope.harness), command, scope.cwd or default_cwd
            )
        command = list(commands.get(scope.harness, []))
        if not command:
            raise ControlError("This harness is not installed on the daemon host")
        kind = HarnessKind(scope.harness)
        env = dict(os.environ)
        if kind is not HarnessKind.OPENCODE:
            plan = native_launch(kind)
            command.extend(plan.args)
            env = wiring.merged_env(env, plan, None)
        if kind is HarnessKind.CODEX and sessions.codex_home:
            env["CODEX_HOME"] = sessions.codex_home
        if kind is HarnessKind.CLAUDE and sessions.claude_config_dir:
            env["CLAUDE_CONFIG_DIR"] = sessions.claude_config_dir
        if kind is HarnessKind.AGY and not any(
            argument == "--gemini_dir" or argument.startswith("--gemini_dir=")
            for argument in command
        ):
            profile = prepare_agy_profile(
                profiles_dir,
                "command-catalog",
                native=True,
                canonical_root=Path(sessions.agy_home) if sessions.agy_home else None,
            )
            command.extend(("--gemini_dir", str(profile)))
        return await discover_commands(kind, command, scope.cwd or default_cwd, env, spawn)

    return CommandCatalogCache(
        discover,
        default_cwd=default_cwd,
        supported_backends=("host", "docker") if docker_probe is not None else ("host",),
    )
