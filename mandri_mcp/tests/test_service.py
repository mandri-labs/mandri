import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx2
import pytest
from mandri.core.ids import HarnessKind
from mandri.core.types.mcp import McpServerConfig
from mandri.mcp.registry import Registry
from mandri.mcp.service import McpService
from mcp import Client, types
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import Server

TIMEOUT = 3


async def ready(service: McpService) -> None:
    async with asyncio.timeout(TIMEOUT):
        while not any(row.state == "ready" for row in service.supervisors.values()):
            await asyncio.sleep(0)


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch):
    connections = []
    calls = []
    barrier = asyncio.Event()
    barrier.set()

    async def tools(ctx, params):
        return types.ListToolsResult(
            tools=[
                types.Tool(name="echo", input_schema={"type": "object"}, description="Echo"),
            ]
        )

    async def call(ctx, params):
        calls.append(params.arguments)
        await ctx.session.report_progress(1, 2, "Working")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=str(params.arguments))],
            structured_content=params.arguments,
        )

    async def resources(ctx, params):
        return types.ListResourcesResult(
            resources=[
                types.Resource(uri="fixture://resource", name="resource", mime_type="text/plain")
            ]
        )

    async def templates(ctx, params):
        return types.ListResourceTemplatesResult(
            resource_templates=[
                types.ResourceTemplate(uri_template="fixture://{key}", name="template")
            ]
        )

    async def read(ctx, params):
        return types.ReadResourceResult(
            contents=[
                types.TextResourceContents(
                    uri=params.uri, text="resource contents", mime_type="text/plain"
                )
            ]
        )

    async def prompts(ctx, params):
        return types.ListPromptsResult(prompts=[types.Prompt(name="greeting")])

    async def prompt(ctx, params):
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user", content=types.TextContent(type="text", text="Hello")
                )
            ]
        )

    server = Server(
        "upstream",
        on_list_tools=tools,
        on_call_tool=call,
        on_list_resources=resources,
        on_list_resource_templates=templates,
        on_read_resource=read,
        on_list_prompts=prompts,
        on_get_prompt=prompt,
    )

    @asynccontextmanager
    async def connect(config, base_dir, auth=None, message_handler=None):
        connections.append(config.name)
        if config.name == "broken":
            raise FileNotFoundError()
        await barrier.wait()
        async with Client(server) as client:
            yield client

    monkeypatch.setattr("mandri.mcp.supervisor.connect", connect)
    return connections, calls, barrier


async def test_daemon_owns_upstream_and_bindings_never_wait(tmp_path, upstream):
    connections, _, barrier = upstream
    service = McpService(tmp_path, 9000)
    service.start()
    await service.add(McpServerConfig(name="echo", command="fixture"))
    await ready(service)
    first = service.bind("first", HarnessKind.CODEX)
    second = service.bind("second", HarnessKind.CLAUDE)
    assert len(first.servers) == len(second.servers) == 1
    assert connections == ["echo"]
    service.release("first")
    service.release("second")
    assert service.snapshot().servers[0].state == "ready"
    assert connections == ["echo"]
    barrier.clear()
    await service.reconnect("echo")
    cold = service.bind("cold", HarnessKind.PI)
    assert cold.servers == []
    assert not barrier.is_set()
    await service.aclose()


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_local_sdk_endpoint_preserves_results_and_scopes(tmp_path, upstream, mode):
    _, calls, _ = upstream
    service = McpService(tmp_path, 9000)
    service.start()
    await service.add(McpServerConfig(name="echo", command="fixture"))
    await ready(service)
    binding = service.bind("first", HarnessKind.CODEX)
    endpoint = binding.servers[0]

    async def application(scope, receive, send):
        scope = dict(scope)
        scope["root_path"] = "/v1/mcp/connect"
        await service.dispatch(scope, receive, send)

    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=application),
        headers={"Authorization": f"Bearer {binding.token}"},
    ) as http:
        async with Client(
            streamable_http_client(endpoint.url, http_client=http), mode=mode
        ) as client:
            assert [tool.name for tool in (await client.list_tools()).tools] == ["echo"]
            assert [resource.name for resource in (await client.list_resources()).resources] == [
                "resource"
            ]
            assert [
                template.name
                for template in (await client.list_resource_templates()).resource_templates
            ] == ["template"]
            assert [prompt.name for prompt in (await client.list_prompts()).prompts] == ["greeting"]
            resource = await client.read_resource("fixture://resource")
            assert resource.contents[0].text == "resource contents"
            prompt = await client.get_prompt("greeting")
            assert prompt.messages[0].content.text == "Hello"
            progress = []
            progressed = asyncio.Event()

            async def report(value, total, message):
                progress.append((value, total, message))
                progressed.set()

            result = await client.call_tool("echo", {"value": "hello"}, progress_callback=report)
            await asyncio.wait_for(progressed.wait(), 1)
            assert progress == [(1, 2, "Working")]
            assert result.structured_content == {"value": "hello"}
            assert calls == [{"value": "hello"}]
        wrong = await http.post(endpoint.url.replace(binding.id, "0" * 32), json={})
        assert wrong.status_code == 401
        service.release("first")
        revoked = await http.post(endpoint.url, json={})
        assert revoked.status_code == 401
    await service.aclose()


async def test_failure_is_independent_and_protected_sessions_omit_host_servers(tmp_path, upstream):
    service = McpService(tmp_path, 9000)
    service.start()
    await service.add(McpServerConfig(name="echo", command="fixture"))
    await ready(service)
    assert service.bind("protected", HarnessKind.CODEX, protected=True).servers == []
    assert service.bind("excluded", HarnessKind.AGY).servers == []
    row = service.snapshot().servers[0]
    await service.update(row.id, row.revision, {"enabled": False})
    assert service.bind("disabled", HarnessKind.CODEX).servers == []
    assert service.snapshot().servers[0].state == "disabled"
    await service.aclose()


def test_registry_encrypts_secrets_and_preserves_other_entries(tmp_path: Path):
    registry = Registry(tmp_path)
    local = registry.add(McpServerConfig(name="local", command="fixture", env={"KEY": "private"}))
    remote = registry.add(
        McpServerConfig(
            name="remote",
            transport="streamable_http",
            url="https://example.invalid/mcp",
            headers={"Authorization": "Bearer private"},
        )
    )
    assert "private" not in registry.path.read_text()
    assert all(b"private" not in path.read_bytes() for path in (tmp_path / "mcp-secrets").glob("*"))
    loaded = Registry(tmp_path)
    loaded.load()
    assert loaded.get(local.id).config.env == {"KEY": "private"}
    assert loaded.get(remote.id).config.headers == {"Authorization": "Bearer private"}
    with pytest.raises(ValueError):
        loaded.update(local.id, 0, {"enabled": False})
    loaded.remove(local.id)
    assert loaded.get(remote.id) == remote


def test_invalid_entry_does_not_block_valid_edits(tmp_path):
    registry = Registry(tmp_path)
    row = registry.add(McpServerConfig(name="valid", command="fixture"))
    with registry.path.open("a") as handle:
        handle.write(
            '\n[[servers]]\nid = "8d137bdc-8bdc-469f-b42a-36b7b04a88df"\nname = "invalid"\n'
        )
    restored = Registry(tmp_path)
    restored.load()
    assert restored.errors == {"entry-1": "Invalid server definition"}
    updated = restored.update(row.id, row.revision, {"enabled": False})
    assert not updated.config.enabled
    restored.discard_invalid("entry-1")
    reloaded = Registry(tmp_path)
    reloaded.load()
    assert not reloaded.errors
    assert reloaded.get(row.id) == updated


async def test_failed_server_does_not_withhold_other_bindings(tmp_path, upstream):
    service = McpService(tmp_path, 9000)
    service.start()
    try:
        await service.add(McpServerConfig(name="echo", command="fixture"))
        await service.add(McpServerConfig(name="broken", command="missing"))
        async with asyncio.timeout(TIMEOUT):
            while {row.state for row in service.supervisors.values()} != {"ready", "error"}:
                await asyncio.sleep(0)
        binding = service.bind("session", HarnessKind.CODEX)
        assert [endpoint.name for endpoint in binding.servers] == ["echo"]
        failed = next(row for row in service.snapshot().servers if row.name == "broken")
        assert failed.error == "Command or directory not found"
    finally:
        await service.aclose()


async def test_background_health_failure_withdraws_ready_binding(tmp_path, upstream, monkeypatch):
    service = McpService(tmp_path, 9000)
    service.start()
    try:
        row = await service.add(McpServerConfig(name="echo", command="fixture"))
        await ready(service)
        supervisor = service.supervisors[row.id]

        async def timed_out():
            raise TimeoutError()

        monkeypatch.setattr(supervisor.client.session, "send_ping", timed_out)
        supervisor.refresh.set()
        async with asyncio.timeout(TIMEOUT):
            while supervisor.state == "ready":
                await asyncio.sleep(0)
        assert supervisor.error == "Server did not respond"
        assert service.bind("session", HarnessKind.PI).servers == []
    finally:
        await service.aclose()


async def test_retry_is_loading_until_connection_fails(tmp_path, monkeypatch):
    entered = asyncio.Event()
    finish: asyncio.Future[Client] = asyncio.get_running_loop().create_future()
    attempts = 0

    @asynccontextmanager
    async def retry(config, base_dir, auth=None, message_handler=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FileNotFoundError()
        entered.set()
        yield await finish

    monkeypatch.setattr("mandri.mcp.supervisor.connect", retry)
    service = McpService(tmp_path, 9000)
    service.start()
    try:
        row = await service.add(McpServerConfig(name="echo", command="fixture"))
        supervisor = service.supervisors[row.id]
        async with asyncio.timeout(TIMEOUT):
            while supervisor.state != "error":
                await asyncio.sleep(0)
        assert supervisor.error == "Command or directory not found"
        await asyncio.wait_for(entered.wait(), TIMEOUT)
        assert supervisor.state == "reconnecting"
        assert supervisor.error is None
        assert service.bind("session", HarnessKind.CODEX).servers == []
        finish.set_exception(TimeoutError())
        async with asyncio.timeout(TIMEOUT):
            while supervisor.state != "error":
                await asyncio.sleep(0)
        assert supervisor.error == "Server did not respond"
    finally:
        await service.aclose()
