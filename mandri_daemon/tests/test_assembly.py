"""Assembly tests for the daemon composition root."""

import asyncio
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from mandri.config.toml_adapter import TomlConfigAdapter
from mandri.core.hub import Hub, Topic
from mandri.core.ids import HarnessKind
from mandri.daemon import daemonctl
from mandri.daemon.daemonctl import Address, ProbeStatus
from mandri.daemon.serve import (
    RuntimeResources,
    build_app,
    build_harness_adapters,
    build_harness_commands,
    build_server,
    load_config,
    start_background_tasks,
    stop_background_tasks,
    wire_runtime,
)
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.runtime.adapters import AdapterContext, HarnessAdapters
from mandri.runtime.control.claude import ClaudeControlAdapter
from mandri.runtime.control.opencode import OpencodeControlAdapter
from mandri.runtime.service import RuntimeService
from mandri.sessions.sync import SyncEngine


@pytest.fixture(autouse=True)
def fake_command_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mandri.daemon.command_catalogs.discover_commands", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        "mandri.daemon.command_catalogs.DockerCommandDiscovery.discover", AsyncMock(return_value=[])
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_load_config_creates_default_file(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    adapter = TomlConfigAdapter(tmp_path)
    assert adapter.config_path.is_file()
    assert config == adapter.load()


def test_load_config_persists_overrides(tmp_path: Path) -> None:
    config = load_config(tmp_path, "0.0.0.0", 9999)
    assert config.server.host == "0.0.0.0"
    assert config.server.port == 9999
    assert TomlConfigAdapter(tmp_path).load().server.port == 9999


def test_desktop_connection_overrides_do_not_change_shared_config(tmp_path: Path) -> None:
    original = load_config(tmp_path, "0.0.0.0", 9999)
    path = tmp_path / "config.toml"
    before = path.read_bytes()
    runtime = load_config(tmp_path, "127.0.0.1", 43123, persist_overrides=False)
    assert runtime.server.host == "127.0.0.1"
    assert runtime.server.port == 43123
    assert path.read_bytes() == before
    assert TomlConfigAdapter(tmp_path).load() == original


def test_first_desktop_launch_saves_default_not_temporary_port(tmp_path: Path) -> None:
    runtime = load_config(tmp_path, "127.0.0.1", 43123, persist_overrides=False)
    assert runtime.server.port == 43123
    assert TomlConfigAdapter(tmp_path).load().server.port == 8787


def test_build_harness_commands_resolves_and_overrides() -> None:
    commands = build_harness_commands({"codex": ["codex", "custom"]})
    assert commands["codex"] == ["codex", "custom"]
    for harness, argv in commands.items():
        if harness == "codex":
            continue
        assert Path(argv[0]).is_absolute()


def test_default_claude_launch_enables_partial_messages() -> None:
    commands = build_harness_commands(include_docker=True)
    assert "--include-partial-messages" in commands["claude"]
    for harness in ("codex", "opencode", "agy"):
        assert "--include-partial-messages" not in commands[harness]
    assert build_harness_commands({"claude": ["claude", "custom"]})["claude"] == [
        "claude",
        "custom",
    ]


def test_build_harness_adapters_opencode_requires_listen_context() -> None:
    assert build_harness_adapters(AdapterContext(kind=HarnessKind.OPENCODE)) is None
    adapters = build_harness_adapters(
        AdapterContext(kind=HarnessKind.OPENCODE, listen_port=8123, native_session_id="oc-1")
    )
    assert adapters is not None
    assert isinstance(adapters, HarnessAdapters)
    assert isinstance(adapters.control, OpencodeControlAdapter)
    assert adapters.events is adapters.control
    assert adapters.delivery is not None


async def test_opencode_adapter_sends_gateway_alias() -> None:
    adapters = build_harness_adapters(
        AdapterContext(
            kind=HarnessKind.OPENCODE,
            listen_port=8123,
            native_session_id="oc-1",
            model="provider/new-model",
        )
    )
    assert adapters is not None
    control = adapters.control
    assert isinstance(control, OpencodeControlAdapter)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    await control._client.aclose()
    control._client = httpx.AsyncClient(
        base_url="http://localhost:8123", transport=httpx.MockTransport(respond)
    )
    try:
        await control.send_prompt("continue")
    finally:
        await control.aclose()
    assert requests[0].url.path == "/session/oc-1/prompt_async"
    assert json.loads(requests[0].content)["model"] == {
        "providerID": "mandri",
        "modelID": "mandri_gateway",
    }


def test_build_harness_adapters_claude_stdio() -> None:
    adapters = build_harness_adapters(
        AdapterContext(
            kind=HarnessKind.CLAUDE,
            process=cast(Any, SimpleNamespace()),
            hub=Hub(),
            topic=Topic("sessions:test"),
        )
    )
    assert adapters is not None
    assert isinstance(adapters, HarnessAdapters)
    assert isinstance(adapters.control, ClaudeControlAdapter)


def test_pid_file_roundtrip(tmp_path: Path) -> None:
    daemonctl.write_pid_file(tmp_path)
    record = daemonctl.read_pid_file(tmp_path)
    assert record is not None
    assert record.pid == os.getpid()
    daemonctl.remove_pid_file(tmp_path)
    assert daemonctl.read_pid_file(tmp_path) is None


def test_probe_unreachable() -> None:
    address = Address(host="127.0.0.1", port=_free_port())
    assert daemonctl.probe(address) is ProbeStatus.UNREACHABLE


def test_is_running_without_state(tmp_path: Path) -> None:
    assert daemonctl.is_running(tmp_path) is False


async def _wire(
    tmp_path: Path,
) -> tuple[RuntimeResources, SyncEngine, Hub, httpx.AsyncClient, AiosqliteDatabase]:
    config = load_config(tmp_path)
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "mandri.db")
    await db.migrate()
    hub = Hub()
    http = httpx.AsyncClient()
    resources = RuntimeResources()
    engine = await wire_runtime(resources, tmp_path, config, hub, db, http)
    return resources, engine, hub, http, db


async def test_wire_runtime_wires_all_adapters(tmp_path: Path) -> None:
    resources, _engine, hub, http, db = await _wire(tmp_path)
    try:
        assert resources.db is db
        assert resources.usage_db is not None
        assert resources.usage_db is not db
        assert resources.http is http
        assert resources.hub is hub
        assert resources.sessions is not None
        assert resources.providers is not None
        assert resources.gateway is not None
        assert isinstance(resources.runtime, RuntimeService)
        assert resources.actions is not None
        config = load_config(tmp_path)
        expected = set(
            build_harness_commands(
                config.sessions.launch_args, include_docker=config.docker.image is not None
            )
        )
        assert set(resources.runtime.installed_harnesses()) == expected
        assert cast(Any, resources.providers._routes)._handle[0] is resources.gateway.registry
    finally:
        if resources.runtime is not None:
            await resources.runtime.commands.catalogs.aclose()
        await hub.close_all()
        await http.aclose()
        if resources.usage_db is not None:
            await resources.usage_db.close()
        await db.close()


async def test_build_app_lifespan_carries_resources(tmp_path: Path) -> None:
    resources, _engine, hub, http, db = await _wire(tmp_path)
    try:
        app = build_app(resources)
        async with app.router.lifespan_context(app):
            state = app.state.lifespan
            assert state.db is resources.db
            assert state.sessions is resources.sessions
            assert state.providers is resources.providers
            assert state.gateway is resources.gateway
            assert state.runtime is resources.runtime
            assert state.lifetime is resources.runtime
            assert state.actions is resources.actions
    finally:
        if resources.runtime is not None:
            await resources.runtime.commands.catalogs.aclose()
        await hub.close_all()
        await http.aclose()
        if resources.usage_db is not None:
            await resources.usage_db.close()
        await db.close()


async def test_background_tasks_start_and_stop(tmp_path: Path) -> None:
    resources, engine, hub, http, db = await _wire(tmp_path)
    server, _fresh = build_server("127.0.0.1", _free_port())
    config = load_config(tmp_path)
    runtime = resources.runtime
    assert runtime is not None
    tasks = start_background_tasks(engine, runtime, config, server, tmp_path)
    try:
        await asyncio.sleep(0.1)
        assert not tasks.sync.done()
        assert not tasks.reconcile.done()
        assert not tasks.approvals.done()
        assert not tasks.pid_watch.done()
    finally:
        await stop_background_tasks(tasks, resources, hub, http, db, tmp_path)
    assert tasks.sync.done()
    assert tasks.reconcile.done()
    assert tasks.approvals.done()
    assert tasks.pid_watch.done()
