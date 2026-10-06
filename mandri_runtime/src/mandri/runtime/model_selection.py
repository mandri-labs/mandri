import os
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path

from mandri.core.ids import HarnessKind, SessionId
from mandri.core.types.model_selection import ModelSource, model_capabilities
from mandri.providers.errors import ProviderInvalidError
from mandri.runtime import wiring
from mandri.runtime.errors import HarnessNotInstalledError
from mandri.runtime.native_catalog import NativeModel, discover_models
from mandri.runtime.native_launch import native_launch
from mandri.runtime.process import ManagedProcess
from mandri.runtime.session_state import RuntimeStates
from mandri.sessions.agy_profiles import prepare_agy_profile
from mandri.sessions.errors import SessionConflictError
from mandri.sessions.service import SessionsService


class ModelSelectionService:
    def __init__(
        self,
        commands: Mapping[str, list[str]],
        states: RuntimeStates,
        sessions: SessionsService | None,
        agy_home: Path | None = None,
        agy_profiles_dir: Path | None = None,
    ) -> None:
        self._harness_commands = commands
        self._session_state = states.session
        self._sessions = sessions
        self._agy_home = agy_home
        self._agy_profiles = agy_profiles_dir or Path.home() / ".mandri" / "agy-profiles"
        self._native_catalogs: dict[tuple[str, str], tuple[float, list[NativeModel]]] = {}

    @staticmethod
    def validate_model_selection(harness: str, model: str, source: ModelSource) -> None:
        if source not in model_capabilities(harness).sources:
            raise ProviderInvalidError(
                f"Harness {harness!r} does not support {source.value} models"
            )
        if source is ModelSource.NATIVE and not model.strip():
            raise ProviderInvalidError("Native model name must not be blank")

    async def native_models(
        self, harness: str, cwd: str | None, spawn_process: Callable[..., Awaitable[ManagedProcess]]
    ) -> list[NativeModel]:
        self.validate_model_selection(harness, "default", ModelSource.NATIVE)
        command = self._harness_commands.get(harness)
        if command is None:
            raise HarnessNotInstalledError(f"no command configured for harness {harness!r}")
        directory = cwd or str(Path.cwd())
        key = (harness, directory)
        cached = self._native_catalogs.get(key)
        if cached is not None and time.monotonic() - cached[0] < 60:
            return cached[1]
        kind = HarnessKind(harness)
        plan = native_launch(kind)
        if kind is HarnessKind.AGY:
            profile = prepare_agy_profile(
                self._agy_profiles, "catalog", native=True, canonical_root=self._agy_home
            )
            command = [command[0], "--gemini_dir", str(profile), "models"]
        rows = await discover_models(
            kind,
            [*command, *plan.args],
            directory,
            wiring.merged_env(os.environ, plan, None),
            spawn_process,
        )
        self._native_catalogs[key] = (time.monotonic(), rows)
        return rows

    async def needs_restart(self, session_id: str) -> bool:
        if self._sessions is None:
            return False
        record = await self._sessions.get_session(SessionId(session_id))
        selection = (record.model_source, record.model or "default", record.reasoning_effort)
        previous = self._session_state(session_id).launched_model
        if previous is None or previous == selection:
            return False
        if previous[0] is ModelSource.GATEWAY and selection[0] is ModelSource.GATEWAY:
            capabilities = model_capabilities(record.harness)
            return (
                capabilities.gateway_model_requires_restart and previous[1] != selection[1]
            ) or (capabilities.gateway_effort_requires_restart and previous[2] != selection[2])
        if record.native_id is None:
            raise SessionConflictError(
                "Wait for the first harness response before changing native model settings"
            )
        return True
