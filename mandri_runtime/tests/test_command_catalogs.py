import asyncio
from typing import Any

import pytest
from mandri.core.protocol.commands import CommandCatalogParams
from mandri.runtime.command_catalogs import CommandCatalogCache
from mandri.runtime.control.errors import ControlError


async def test_startup_snapshot_waits_for_all_harnesses_without_sessions(tmp_path) -> None:
    called: list[CommandCatalogParams] = []
    ready = asyncio.Event()

    async def discover(scope: CommandCatalogParams) -> list[dict[str, Any]]:
        called.append(scope)
        await ready.wait()
        return [{"id": scope.harness, "name": scope.harness}]

    cache = CommandCatalogCache(discover, default_cwd=str(tmp_path))
    cache.start()
    snapshot = asyncio.create_task(cache.snapshot())
    await asyncio.sleep(0)
    assert not snapshot.done()
    ready.set()
    result = await snapshot
    assert result.default_cwd == str(tmp_path)
    assert {item.harness for item in result.catalogs} == {
        "claude",
        "codex",
        "agy",
        "opencode",
        "pi",
    }
    assert all(item.state == "ready" for item in result.catalogs)
    assert len(called) == 5
    await cache.snapshot()
    assert len(called) == 5
    await cache.aclose()


async def test_scope_isolation_and_inflight_deduplication(tmp_path) -> None:
    calls: list[str | None] = []

    async def discover(scope: CommandCatalogParams) -> list[dict[str, Any]]:
        calls.append(scope.cwd)
        await asyncio.sleep(0)
        return [{"id": "custom", "name": "custom"}]

    cache = CommandCatalogCache(discover, default_cwd=str(tmp_path))
    scope = CommandCatalogParams(harness="claude")
    first, second = await asyncio.gather(cache.get(scope), cache.get(scope))
    first.commands.clear()
    assert len(second.commands) == 1
    assert len((await cache.get(scope)).commands) == 1
    assert len(calls) == 1
    await cache.get(scope.model_copy(update={"cwd": str(tmp_path / "other")}))
    assert len(calls) == 2
    assert calls[0] != calls[1]
    await cache.aclose()


@pytest.mark.parametrize(
    "extra",
    [{"profile_id": "other"}, {"execution_backend": "docker"}, {"privacy_mode": "surrogate"}],
)
async def test_unmatched_execution_scope_never_uses_host_catalog(tmp_path, extra) -> None:
    async def discover(scope: CommandCatalogParams) -> list[dict[str, Any]]:
        raise AssertionError("Must not discover using a different execution scope")

    cache = CommandCatalogCache(discover, default_cwd=str(tmp_path))
    catalog = await cache.get(CommandCatalogParams(harness="claude", **extra))
    assert catalog.state == "unavailable"
    assert catalog.commands == []
    assert catalog.reason


async def test_failures_are_cached_and_do_not_poison_other_harnesses(tmp_path) -> None:
    calls = 0

    async def discover(scope: CommandCatalogParams) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        if scope.harness == "claude":
            raise ControlError("Native discovery unavailable")
        return []

    cache = CommandCatalogCache(discover, default_cwd=str(tmp_path))
    failed = await cache.get(CommandCatalogParams(harness="claude"))
    assert failed.state == "unavailable"
    assert failed.reason == "Native discovery unavailable"
    await cache.get(CommandCatalogParams(harness="claude"))
    assert calls == 1
    ready = await cache.get(CommandCatalogParams(harness="agy"))
    assert ready.state == "ready"
    assert ready.commands == []


async def test_timeout_cancels_probe_and_returns_readable_unavailable(tmp_path) -> None:
    cancelled = asyncio.Event()

    async def discover(scope: CommandCatalogParams) -> list[dict[str, Any]]:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return []

    cache = CommandCatalogCache(discover, default_cwd=str(tmp_path), timeout_seconds=0.001)
    catalog = await cache.get(CommandCatalogParams(harness="codex"))
    assert catalog.state == "unavailable"
    assert "timed out" in (catalog.reason or "")
    assert cancelled.is_set()


async def test_cancelled_client_does_not_cancel_shared_discovery(tmp_path) -> None:
    ready = asyncio.Event()

    async def discover(scope: CommandCatalogParams) -> list[dict[str, Any]]:
        await ready.wait()
        return [{"id": "native", "name": "native"}]

    cache = CommandCatalogCache(discover, default_cwd=str(tmp_path))
    scope = CommandCatalogParams(harness="claude")
    client = asyncio.create_task(cache.get(scope))
    await asyncio.sleep(0)
    client.cancel()
    with pytest.raises(asyncio.CancelledError):
        await client
    ready.set()
    assert (await cache.get(scope)).state == "ready"
    await cache.aclose()


async def test_manual_refresh_retries_unavailable_without_changing_scope(tmp_path) -> None:
    attempts = 0

    async def discover(scope: CommandCatalogParams) -> list[dict[str, Any]]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ControlError("Temporarily unavailable")
        return [{"id": "recovered", "name": "recovered"}]

    cache = CommandCatalogCache(discover, default_cwd=str(tmp_path))
    scope = CommandCatalogParams(harness="claude")
    assert (await cache.get(scope)).state == "unavailable"
    refreshed = await cache.get(scope.model_copy(update={"force_refresh": True}))
    assert refreshed.state == "ready"
    assert "force_refresh" not in refreshed.model_dump()
    assert (await cache.get(scope)).commands[0].id == "recovered"
    assert attempts == 2
    assert not cache._pending


async def test_configured_docker_probe_has_independent_catalog(tmp_path) -> None:
    calls: list[str] = []

    async def discover(scope: CommandCatalogParams) -> list[dict[str, Any]]:
        calls.append(scope.execution_backend)
        return [{"id": scope.execution_backend, "name": scope.execution_backend}]

    cache = CommandCatalogCache(
        discover, default_cwd=str(tmp_path), supported_backends=("host", "docker")
    )
    host = await cache.get(CommandCatalogParams(harness="claude"))
    docker = await cache.get(CommandCatalogParams(harness="claude", execution_backend="docker"))
    assert host.commands[0].id == "host"
    assert docker.commands[0].id == "docker"
    private = await cache.get(
        CommandCatalogParams(harness="claude", execution_backend="docker", privacy_mode="surrogate")
    )
    assert private.state == "unavailable"
    assert calls == ["host", "docker"]
