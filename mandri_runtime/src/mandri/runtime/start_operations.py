import asyncio
import contextlib
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from mandri.core.types.execution import ProtectionError

_CURRENT_OPERATION: ContextVar[str | None] = ContextVar("mandri_start_operation", default=None)


@dataclass(frozen=True)
class StartOperation[T]:
    fingerprint: str
    task: asyncio.Task[T]


class StartOperations[T]:
    def __init__(self) -> None:
        self._operations: dict[str, StartOperation[T]] = {}
        self._cancelled: set[str] = set()
        self._cancellations: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    @property
    def current_id(self) -> str | None:
        return _CURRENT_OPERATION.get()

    async def _invoke(self, operation_id: str, factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
        token = _CURRENT_OPERATION.set(operation_id)
        try:
            return await factory()
        finally:
            _CURRENT_OPERATION.reset(token)

    @staticmethod
    def _validate(operation_id: str) -> None:
        try:
            uuid.UUID(operation_id)
        except ValueError as error:
            raise ProtectionError(
                "validation_error", "Operation identity must be a UUID"
            ) from error

    async def run(
        self, operation_id: str, fingerprint: str, factory: Callable[[], Coroutine[Any, Any, T]]
    ) -> T:
        self._validate(operation_id)
        async with self._lock:
            if operation_id in self._cancelled:
                raise ProtectionError("operation_cancelled", "Session preparation was cancelled")
            operation = self._operations.get(operation_id)
            if operation is None:
                task = asyncio.create_task(
                    self._invoke(operation_id, factory), name=f"session-start:{operation_id}"
                )
                task.add_done_callback(self._consume_exception)
                operation = StartOperation(fingerprint, task)
                self._operations[operation_id] = operation
            elif operation.fingerprint != fingerprint:
                raise ProtectionError(
                    "operation_conflict", "Operation identity belongs to another request"
                )
        try:
            return await asyncio.shield(operation.task)
        except asyncio.CancelledError:
            if operation_id in self._cancelled:
                raise ProtectionError(
                    "operation_cancelled", "Session preparation was cancelled"
                ) from None
            raise

    async def cancel(self, operation_id: str, stop: Callable[[T], Awaitable[None]]) -> None:
        self._validate(operation_id)
        async with self._lock:
            cancellation = self._cancellations.get(operation_id)
            if cancellation is None:
                cancellation = asyncio.create_task(self._cancel(operation_id, stop))
                self._cancellations[operation_id] = cancellation
        await asyncio.shield(cancellation)

    async def _cancel(self, operation_id: str, stop: Callable[[T], Awaitable[None]]) -> None:
        async with self._lock:
            self._cancelled.add(operation_id)
            operation = self._operations.get(operation_id)
            if operation is None:
                return
            if not operation.task.done():
                operation.task.cancel()
        try:
            result = await operation.task
        except asyncio.CancelledError:
            return
        await stop(result)

    @staticmethod
    def _consume_exception(task: asyncio.Task[T]) -> None:
        if not task.cancelled():
            with contextlib.suppress(Exception):
                task.exception()
