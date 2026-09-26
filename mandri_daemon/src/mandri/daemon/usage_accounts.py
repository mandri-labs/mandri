import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

from mandri.core.ids import HarnessKind
from mandri.core.launch import merged_env
from mandri.core.types.config import SessionsConfig
from mandri.core.types.usage import UsageAccount
from mandri.runtime.native_launch import native_launch
from mandri.runtime.native_usage import read_native_usage
from mandri.runtime.process import ManagedProcess, spawn
from mandri.runtime.usage_profiles import usage_profile_id
from mandri.sessions.agy_profiles import default_agy_root

logger = logging.getLogger(__name__)
NATIVE_QUOTA_HARNESSES = ("codex", "claude", "agy", "pi")


class NativeQuotaSync:
    def __init__(
        self,
        commands: Mapping[str, list[str]],
        sessions: SessionsConfig,
        sink: Callable[[UsageAccount], Awaitable[object]],
        *,
        spawn_process: Callable[..., Awaitable[ManagedProcess]] = spawn,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._commands = commands
        self._sessions = sessions
        self._sink = sink
        self._spawn = spawn_process
        self._env = dict(os.environ if env is None else env)
        self._lock = asyncio.Lock()

    async def refresh(self) -> None:
        if self._lock.locked():
            return
        pending = []
        async with self._lock, asyncio.TaskGroup() as tasks:
            for harness in NATIVE_QUOTA_HARNESSES:
                command = self._commands.get(harness)
                if command:
                    pending.append(
                        tasks.create_task(self._refresh(HarnessKind(harness), list(command)))
                    )
        if not any(task.result() for task in pending):
            raise RuntimeError("No native account quotas could be refreshed")

    async def _refresh(self, harness: HarnessKind, command: list[str]) -> bool:
        try:
            plan = native_launch(harness)
            env = merged_env(self._env, plan)
            if harness is HarnessKind.CODEX and self._sessions.codex_home:
                env["CODEX_HOME"] = self._sessions.codex_home
            if harness is HarnessKind.CLAUDE and self._sessions.claude_config_dir:
                env["CLAUDE_CONFIG_DIR"] = self._sessions.claude_config_dir
            agy_root = (
                Path(self._sessions.agy_home) if self._sessions.agy_home else default_agy_root()
            )
            profile_id = usage_profile_id(harness, env, agy_root=agy_root)
            if harness is HarnessKind.AGY:
                command = [
                    command[0],
                    "--gemini_dir",
                    str(agy_root),
                    "-p",
                    "/usage",
                    "--output-format",
                    "json",
                ]
            else:
                command.extend(plan.args)
            if harness is HarnessKind.CLAUDE:
                command.extend(
                    (
                        "--no-session-persistence",
                        "--strict-mcp-config",
                        "--mcp-config",
                        '{"mcpServers":{}}',
                    )
                )
            if harness is HarnessKind.PI:
                extension = str(files("mandri.core").joinpath("resources/pi_usage.ts"))
                command.extend(
                    (
                        "--no-session",
                        "--no-extensions",
                        "--no-skills",
                        "--no-prompt-templates",
                        "--extension",
                        extension,
                    )
                )
            with TemporaryDirectory(prefix="mandri-quota-") as directory:
                accounts = await read_native_usage(
                    harness, profile_id, command, Path(directory), env, self._spawn
                )
            for account in accounts:
                if account.has_account_data:
                    await self._sink(account)
            return any(account.has_account_data for account in accounts)
        except PermissionError:
            logger.warning(
                "Native quota refresh failed for %s: authentication required; "
                "sign in to the native CLI with the daemon's user profile",
                harness,
            )
            return False
        except Exception as error:
            logger.warning(
                "Native quota refresh failed for %s (%s); previous readings retained",
                harness,
                type(error).__name__,
            )
            return False
