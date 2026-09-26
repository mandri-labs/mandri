import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from mandri.core.ids import HarnessKind
from mandri.core.protocol.commands import (
    CommandCatalogParams,
    CommandCatalogsSnapshot,
    CommandDescriptor,
    ScopedCommandCatalog,
)
from mandri.runtime.control.errors import ControlError

CommandDiscovery = Callable[[CommandCatalogParams], Awaitable[list[dict[str, Any]]]]
CatalogKey = tuple[str, str, str | None, str, str]


class CommandCatalogCache:
    def __init__(
        self,
        discover: CommandDiscovery | None = None,
        *,
        default_cwd: str | None = None,
        timeout_seconds: float = 25,
        supported_backends: tuple[Literal["host", "docker"], ...] = ("host",),
    ) -> None:
        self._discover = discover
        self._backends = supported_backends
        self._cwd = str(Path(default_cwd or Path.cwd()).resolve())
        self._timeout = timeout_seconds
        self._entries: dict[CatalogKey, ScopedCommandCatalog] = {}
        self._pending: dict[CatalogKey, asyncio.Task[ScopedCommandCatalog]] = {}
        self._startup: asyncio.Task[None] | None = None
        self._projects: asyncio.Task[None] | None = None
        self._semaphore = asyncio.Semaphore(4)

    def start(self, projects: Sequence[CommandCatalogParams] = ()) -> None:
        if self._startup is None:
            self._startup = asyncio.create_task(self._warm_defaults(), name="command-catalogs")
            if projects:
                self._projects = asyncio.create_task(
                    self._warm_projects(projects), name="command-project-catalogs"
                )

    async def _warm_defaults(self) -> None:
        await asyncio.gather(
            *(
                self.get(CommandCatalogParams(harness=kind.value, execution_backend=backend))
                for backend in self._backends
                for kind in HarnessKind
            )
        )

    async def _warm_projects(self, projects: Sequence[CommandCatalogParams]) -> None:
        if self._startup is not None:
            await asyncio.shield(self._startup)
        await asyncio.gather(*(self.get(scope) for scope in projects))

    async def snapshot(self) -> CommandCatalogsSnapshot:
        self.start()
        assert self._startup is not None
        await asyncio.shield(self._startup)
        return CommandCatalogsSnapshot(
            default_cwd=self._cwd,
            catalogs=[row.model_copy(deep=True) for row in self._entries.values()],
        )

    async def get(self, params: CommandCatalogParams) -> ScopedCommandCatalog:
        directory = str(Path(params.cwd or self._cwd).expanduser().resolve())
        scope = params.model_copy(update={"cwd": directory})
        key = (
            scope.harness,
            directory,
            scope.profile_id,
            scope.execution_backend,
            scope.privacy_mode,
        )
        cached = self._entries.get(key)
        if cached is not None and not params.force_refresh:
            return cached.model_copy(deep=True)
        task = self._pending.get(key)
        if task is None:
            task = asyncio.create_task(
                self._load_and_store(key, scope), name=f"command-catalog:{scope.harness}"
            )
            self._pending[key] = task
        result = await asyncio.shield(task)
        return result.model_copy(deep=True)

    async def _load_and_store(
        self, key: CatalogKey, scope: CommandCatalogParams
    ) -> ScopedCommandCatalog:
        try:
            result = await self._load(scope)
            self._entries[key] = result
            return result
        finally:
            self._pending.pop(key, None)

    async def _load(self, scope: CommandCatalogParams) -> ScopedCommandCatalog:
        fields = scope.model_dump(exclude={"force_refresh"})
        if scope.execution_backend not in self._backends:
            return ScopedCommandCatalog(
                **fields,
                state="unavailable",
                commands=[],
                reason="A matching Docker command probe is not available",
            )
        if scope.privacy_mode != "none":
            return ScopedCommandCatalog(
                **fields,
                state="unavailable",
                commands=[],
                reason="Private command discovery requires a matching protected workspace probe",
            )
        if scope.profile_id is not None:
            return ScopedCommandCatalog(
                **fields,
                state="unavailable",
                commands=[],
                reason="This profile has no standalone command probe",
            )
        if self._discover is None:
            return ScopedCommandCatalog(
                **fields,
                state="unavailable",
                commands=[],
                reason="Native command discovery is not configured",
            )
        try:
            async with self._semaphore, asyncio.timeout(self._timeout):
                rows = await self._discover(scope)
            commands = {row["id"]: CommandDescriptor.model_validate(row) for row in rows}
            return ScopedCommandCatalog(**fields, state="ready", commands=list(commands.values()))
        except TimeoutError:
            reason = "Native command discovery timed out"
        except ControlError as exc:
            reason = str(exc)
        except Exception:
            reason = "Native command discovery is unavailable in this scope"
        return ScopedCommandCatalog(**fields, state="unavailable", commands=[], reason=reason)

    async def aclose(self) -> None:
        tasks = [
            *self._pending.values(),
            *[task for task in (self._startup, self._projects) if task is not None],
        ]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
