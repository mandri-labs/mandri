import asyncio
from collections.abc import Callable
from typing import Any, Protocol, cast

from mandri.core.protocol.commands import (
    CommandCatalog,
    CommandDescriptor,
    CommandInvocation,
    CommandResult,
)
from mandri.core.protocol.errors import ProtocolError, ProtocolErrorCode
from mandri.runtime.command_catalogs import CommandCatalogCache
from mandri.runtime.control.errors import ControlError, ControlTransportError


class CommandControl(Protocol):
    async def list_commands(self) -> list[dict[str, Any]]: ...

    async def execute_command(self, command_id: str, arguments: str) -> dict[str, Any]: ...

    async def interrupt(self) -> bool: ...


class CommandService:
    def __init__(
        self,
        control: Callable[[str], object | None],
        live: Callable[[str], bool],
        busy: Callable[[str], bool],
    ) -> None:
        self.catalogs = CommandCatalogCache()
        self._control = control
        self._live = live
        self._busy = busy
        self._records: dict[tuple[str, str], CommandInvocation] = {}
        self._arguments: dict[tuple[str, str], str] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._owners: dict[tuple[str, str], CommandControl] = {}
        self._running: dict[str, CommandInvocation] = {}
        self._uncertain: dict[str, CommandControl] = {}

    def active(self, session_id: str) -> bool:
        control = self._control(session_id)
        if control is None:
            return False
        if self._uncertain.get(session_id) is control:
            return True
        task = self._tasks.get(session_id)
        record = self._running.get(session_id)
        return (
            task is not None
            and not task.done()
            and record is not None
            and self._owners.get((session_id, record.invocation_id)) is control
        )

    def _require(self, session_id: str) -> CommandControl:
        control = self._control(session_id)
        if not self._live(session_id) or control is None:
            raise ProtocolError(ProtocolErrorCode.SESSION_NOT_RUNNING, "Resume this session first")
        if not callable(getattr(control, "list_commands", None)):
            raise ProtocolError(
                ProtocolErrorCode.STEER_UNSUPPORTED,
                "This harness does not expose a native command catalog",
            )
        return cast(CommandControl, control)

    async def catalog(self, session_id: str) -> CommandCatalog:
        control = self._require(session_id)
        try:
            async with asyncio.timeout(20):
                rows = await control.list_commands()
        except (ControlError, TimeoutError) as exc:
            raise ProtocolError(
                ProtocolErrorCode.CONTROL_DELIVERY_FAILED, "Native command discovery failed"
            ) from exc
        unique = {row["id"]: CommandDescriptor.model_validate(row) for row in rows}
        return CommandCatalog(commands=list(unique.values()))

    def list(self, session_id: str) -> list[CommandInvocation]:
        return [record for (sid, _), record in self._records.items() if sid == session_id]

    def get(self, session_id: str, invocation_id: str) -> CommandInvocation:
        record = self._records.get((session_id, invocation_id))
        if record is None:
            raise ProtocolError(
                ProtocolErrorCode.INVALID_PARAMS,
                "Command state is unavailable; it has not been replayed",
            )
        return record

    async def invoke(
        self, session_id: str, invocation_id: str, command_id: str, arguments: str
    ) -> CommandInvocation:
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            key = (session_id, invocation_id)
            existing = self._records.get(key)
            if existing is not None:
                if existing.command.id != command_id or self._arguments[key] != arguments:
                    raise ProtocolError(
                        ProtocolErrorCode.INVALID_PARAMS, "Invocation identity already used"
                    )
                return existing
            if len(self._records) >= 2048:
                raise ProtocolError(
                    ProtocolErrorCode.SESSION_CONFLICT, "Command history capacity reached"
                )
            control = self._require(session_id)
            if self.active(session_id) or self._busy(session_id):
                raise ProtocolError(
                    ProtocolErrorCode.SESSION_CONFLICT, "Wait for the active operation to finish"
                )
            catalog = await self.catalog(session_id)
            command = next((row for row in catalog.commands if row.id == command_id), None)
            if command is None or not command.available:
                raise ProtocolError(
                    ProtocolErrorCode.INVALID_PARAMS,
                    "This command is no longer available; refresh the catalog",
                )
            if arguments and not command.accepts_arguments:
                raise ProtocolError(
                    ProtocolErrorCode.INVALID_PARAMS,
                    "This native command does not accept arguments",
                )
            if self._control(session_id) is not control or not self._live(session_id):
                raise ProtocolError(ProtocolErrorCode.SESSION_NOT_RUNNING, "Session changed")
            if self._busy(session_id):
                raise ProtocolError(ProtocolErrorCode.SESSION_CONFLICT, "Session became busy")
            record = CommandInvocation(
                invocation_id=invocation_id,
                session_id=session_id,
                command=command,
                arguments=arguments,
                state="running",
                cancellable=bool(getattr(control, "commands_cancellable", False)),
            )
            self._records[key] = record
            self._arguments[key] = arguments
            self._owners[key] = control
            self._running[session_id] = record
            self._tasks[session_id] = asyncio.create_task(
                self._execute(record, control, arguments), name=f"command:{invocation_id}"
            )
            return record

    async def _execute(
        self, record: CommandInvocation, control: CommandControl, arguments: str
    ) -> None:
        try:
            async with asyncio.timeout(300):
                raw = await control.execute_command(record.command.id, arguments)
            if record.state != "running":
                return
            if self._control(record.session_id) is not control or not self._live(record.session_id):
                record.state = "unknown"
                record.error = "The native session changed before completion was confirmed"
                return
            record.result = CommandResult.model_validate(raw)
            record.state = "succeeded"
        except asyncio.CancelledError:
            if record.state == "running":
                record.state = "unknown"
                record.error = "The connection closed before completion was confirmed"
        except TimeoutError:
            record.state = "unknown"
            record.error = "Completion was not confirmed. The command has not been replayed"
        except ControlTransportError as exc:
            record.state = "unknown"
            record.error = str(exc)
        except ControlError as exc:
            record.state = "failed"
            record.error = str(exc)
        except Exception:
            record.state = "unknown"
            record.error = (
                "The native result could not be interpreted. Check the session before retrying"
            )
        finally:
            record.cancellable = False
            if record.state == "unknown" and self._control(record.session_id) is control:
                self._uncertain[record.session_id] = control

    async def cancel(self, session_id: str, invocation_id: str) -> CommandInvocation:
        record = self.get(session_id, invocation_id)
        if record.state != "running":
            return record
        if not record.cancellable:
            raise ProtocolError(
                ProtocolErrorCode.STEER_UNSUPPORTED, "Native cancellation is unavailable"
            )
        control = self._require(session_id)
        owner = self._owners.get((session_id, invocation_id))
        if control is not owner:
            record.state = "unknown"
            record.cancellable = False
            record.error = "The native session changed before interruption"
            return record
        if not await control.interrupt():
            raise ProtocolError(
                ProtocolErrorCode.CONTROL_DELIVERY_FAILED, "Interruption was not acknowledged"
            )
        if record.state == "running":
            record.state = "unknown"
            record.error = "Interruption requested; final native state is not yet confirmed"
        record.cancellable = False
        if record.state == "unknown":
            self._uncertain[session_id] = control
        return record

    def disconnected(self, session_id: str) -> None:
        record = self._running.get(session_id)
        if record is not None and record.state == "running":
            record.state = "unknown"
            record.error = "The connection closed before completion was confirmed"
            record.cancellable = False
            owner = self._owners.get((session_id, record.invocation_id))
            if owner is not None:
                self._uncertain[session_id] = owner
        task = self._tasks.get(session_id)
        if task is not None:
            task.cancel()
