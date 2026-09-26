import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from mandri.core.clock import system_now_ms
from mandri.core.types.usage import UsageAccount
from mandri.runtime.process import ManagedProcess, spawn
from mandri.sessions.usage.accounts import codex_account_snapshot

AccountRead = Callable[[], Awaitable[Mapping[str, Any]]]
AccountParser = Callable[[Mapping[str, Any]], UsageAccount]


class NativeAccountReader:
    def __init__(
        self,
        profile_id: str,
        harness: str,
        *,
        clock: Callable[[], int] = system_now_ms,
        cooldown_ms: int = 60000,
    ) -> None:
        self.profile_id = profile_id
        self.harness = harness
        self._clock = clock
        self._cooldown_ms = max(15000, cooldown_ms)
        self._lock = asyncio.Lock()
        self._cached: UsageAccount | None = None

    async def read_codex(self, read: AccountRead, *, qualified: bool) -> UsageAccount:
        if self.harness != "codex" or not qualified:
            return self.unavailable("unsupported")

        async def fetch() -> UsageAccount:
            return codex_account_snapshot(
                self.profile_id,
                await read(),
                observed_at_ms=self._clock(),
            )

        return await self._read(fetch)

    async def read_agy(
        self,
        binary: Path,
        *,
        cwd: Path,
        env: Mapping[str, str],
        qualified_version: str | None,
        parser: AccountParser | None = None,
    ) -> UsageAccount:
        if self.harness != "agy" or qualified_version != "1.2.1" or parser is None:
            return self.unavailable("unsupported")

        async def fetch() -> UsageAccount:
            payload = await _agy_usage(binary, cwd, env)
            parsed = parser(payload)
            return replace(
                parsed,
                account_id=self.profile_id,
                harness="agy",
                observed_at=self._clock(),
            )

        return await self._read(fetch)

    def unavailable(self, status: str = "unavailable") -> UsageAccount:
        return UsageAccount(
            account_id=self.profile_id,
            harness=self.harness,
            observed_at=self._clock(),
            status=status,
        )

    async def _read(self, fetch: Callable[[], Awaitable[UsageAccount]]) -> UsageAccount:
        async with self._lock:
            if self._cached and self._clock() - self._cached.observed_at < self._cooldown_ms:
                return self._cached
            try:
                async with asyncio.timeout(10):
                    result = await fetch()
            except PermissionError:
                result = self.unavailable("authentication_required")
            except Exception:
                result = self.unavailable()
            self._cached = result
            return result


async def _agy_usage(binary: Path, cwd: Path, env: Mapping[str, str]) -> Mapping[str, Any]:
    process: ManagedProcess | None = None
    tasks: list[asyncio.Task[bytes]] = []
    try:
        process = await spawn(
            [str(binary), "-p", "/usage", "--output-format", "json"],
            cwd=cwd,
            env={**env, "AGY_CLI_DISABLE_AUTO_UPDATE": "true"},
            line_limit=256 * 1024,
        )
        tasks = [
            asyncio.create_task(_bounded_output(process.process.stdout)),
            asyncio.create_task(_bounded_output(process.process.stderr)),
        ]
        await asyncio.gather(*tasks, process.wait())
        stdout = tasks[0].result()
        code = process.returncode
        if code != 0:
            raise ValueError("Native account read failed")
        payload = json.loads(stdout)
        if not isinstance(payload, dict):
            raise ValueError("Unsupported native account response")
        return payload
    finally:
        if process is not None:
            await process.stop(grace=0.2)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _bounded_output(stream: asyncio.StreamReader | None) -> bytes:
    if stream is None:
        raise ValueError("Missing native account stream")
    output = bytearray()
    while data := await stream.read(8192):
        output.extend(data)
        if len(output) > 256 * 1024:
            raise ValueError("Native account response exceeded limit")
    return bytes(output)
