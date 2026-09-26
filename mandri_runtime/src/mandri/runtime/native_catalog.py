import asyncio
import contextlib
import dataclasses
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from mandri.core.ids import HarnessKind
from mandri.runtime.control.errors import ControlTransportError
from mandri.runtime.control.pi_rpc import PiRpcConnection
from mandri.runtime.control.stdio_rpc import StdioRpcConnection
from mandri.runtime.process import ManagedProcess


@dataclasses.dataclass(frozen=True)
class NativeModel:
    id: str
    display_name: str
    reasoning_efforts: tuple[str, ...] = ()
    default_effort: str | None = None


def parse_models(harness: HarnessKind, rows: Sequence[Any]) -> list[NativeModel]:
    models: dict[str, NativeModel] = {"default": NativeModel("default", "Default")}
    for row in rows:
        if not isinstance(row, dict) or row.get("hidden"):
            continue
        if harness is HarnessKind.PI:
            provider = row.get("provider")
            identifier = row.get("id")
            if not isinstance(provider, str) or not isinstance(identifier, str):
                continue
            pi_model = f"{provider}/{identifier}"
            mapping = row.get("thinkingLevelMap")
            mapping = mapping if isinstance(mapping, dict) else {}
            pi_efforts = tuple(
                level
                for level in ("off", "minimal", "low", "medium", "high", "xhigh", "max")
                if mapping.get(level, level) is not None
                and (level not in {"xhigh", "max"} or level in mapping)
            ) if row.get("reasoning") else ()
            models[pi_model] = NativeModel(
                pi_model, str(row.get("name") or identifier), pi_efforts
            )
            continue
        model = (
            row.get("model", row.get("id")) if harness is HarnessKind.CODEX else row.get("value")
        )
        if not isinstance(model, str) or not model.strip():
            continue
        if harness is HarnessKind.CODEX:
            efforts = [
                item.get("reasoningEffort")
                for item in (row.get("supportedReasoningEfforts") or [])
                if isinstance(item, dict)
            ]
        else:
            efforts = row.get("supportedEffortLevels") or []
        default = row.get("defaultReasoningEffort", row.get("defaultEffort"))
        models[model] = NativeModel(
            model,
            str(row.get("displayName") or model),
            tuple(dict.fromkeys(value for value in efforts if isinstance(value, str))),
            default if isinstance(default, str) else None,
        )
        if row.get("isDefault"):
            models["default"] = dataclasses.replace(
                models[model], id="default", display_name="Default"
            )
    return list(models.values())


async def _drain_stderr(process: ManagedProcess) -> None:
    while await process.read_stderr_line():
        pass


async def discover_models(
    harness: HarnessKind,
    command: list[str],
    cwd: str | Path,
    env: dict[str, str],
    spawn: Callable[..., Awaitable[ManagedProcess]],
) -> list[NativeModel]:
    if harness is HarnessKind.PI and "--no-session" not in command:
        command = [*command, "--no-session"]
    process = await spawn(command, cwd=cwd, env=env)
    stderr = asyncio.create_task(_drain_stderr(process))
    connection = StdioRpcConnection(process)
    try:
        async with asyncio.timeout(20):
            if harness is HarnessKind.PI:
                result = await PiRpcConnection(process).call("get_available_models", {})
                rows = result.get("models")
                if not isinstance(rows, list):
                    raise ControlTransportError("Pi did not return a native model catalog")
                return parse_models(harness, rows)
            if harness is HarnessKind.AGY:
                models: list[NativeModel] = [NativeModel("default", "Default")]
                while line := await process.read_stdout_line():
                    slug, separator, label = line.strip().partition("\t")
                    if separator and slug and label:
                        models.append(NativeModel(slug, label))
                code = await process.wait()
                if code != 0 or len(models) == 1:
                    raise ControlTransportError(
                        "Antigravity native catalog unavailable; check Google sign-in"
                    )
                return models
            if harness is HarnessKind.CLAUDE:
                result = await connection.call("initialize", {}, claude=True)
                rows = result.get("models")
                if not isinstance(rows, list):
                    raise ControlTransportError("Claude did not return a native model catalog")
                return parse_models(harness, rows)
            await connection.call(
                "initialize",
                {
                    "clientInfo": {"name": "mandri", "version": "0.1.0"},
                    "capabilities": {},
                },
            )
            await connection.send({"method": "initialized"})
            rows = []
            cursor = None
            while True:
                result = await connection.call(
                    "model/list",
                    {
                        "limit": 100,
                        "includeHidden": False,
                        "cursor": cursor,
                    },
                )
                rows.extend(result.get("data", []))
                cursor = result.get("nextCursor")
                if not cursor:
                    return parse_models(harness, rows)
    finally:
        try:
            await process.stop(grace=1)
        finally:
            stderr.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr
