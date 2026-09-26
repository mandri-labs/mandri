import base64
import gzip
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from mandri.core.ids import HarnessKind
from mandri.runtime.control.errors import ControlError
from mandri.runtime.docker_command_catalog import DockerCatalogTransport, DockerCommandDiscovery
from mandri.runtime.docker_config import DockerConfig


def backend(tmp_path):
    return SimpleNamespace(
        config=DockerConfig("worker:qualified", tmp_path / "state"),
        readiness=AsyncMock(
            return_value=SimpleNamespace(
                image_id="sha256:qualified", harnesses=("codex", "claude", "opencode", "agy")
            )
        ),
        client=SimpleNamespace(
            run=AsyncMock(), remove=AsyncMock(), binary="docker", environment=Mock(return_value={})
        ),
        owner="daemon-owner",
        _await_workspace=AsyncMock(),
    )


async def test_worker_discovery_has_no_host_home_credentials_or_network(tmp_path):
    runtime = backend(tmp_path)
    attached = Mock()
    attached.stop = AsyncMock()
    managed = Mock()
    cleanup = None

    def docker_process(attach, client, name, cleanup_callback):
        nonlocal cleanup
        assert attach is attached
        cleanup = cleanup_callback
        return managed

    async def discover(kind, command, cwd, env, spawn, *, http_transport):
        assert kind is HarnessKind.CLAUDE
        assert command == ["claude", "-p"]
        assert cwd == "/workspace" and env == {}
        assert await spawn(command, cwd=cwd, env=env) is managed
        assert http_transport.container.startswith("mandri-catalog-")
        await cleanup()
        return [{"id": "native", "name": "native"}]

    with (
        patch("mandri.runtime.docker_command_catalog.discover_commands", discover),
        patch("mandri.runtime.docker_command_catalog.spawn", AsyncMock(return_value=attached)),
        patch("mandri.runtime.docker_command_catalog.DockerProcess", docker_process),
    ):
        result = await DockerCommandDiscovery(runtime).discover(
            HarnessKind.CLAUDE, ["/host/bin/claude", "-p"], tmp_path
        )
    assert result[0]["id"] == "native"
    args = runtime.client.run.call_args.args
    assert args[args.index("--network") + 1] == "none"
    assert "--read-only" in args
    assert any(arg.startswith("io.mandri.session=mandri-catalog-") for arg in args)
    assert "--publish" not in args
    mounts = [args[index + 1] for index, arg in enumerate(args) if arg == "--mount"]
    assert mounts == [f"type=bind,src={tmp_path},dst=/workspace,readonly,bind-propagation=rprivate"]
    env = runtime.client.run.call_args.kwargs["env"]
    assert env["HOME"] == "/home/worker"
    assert env["CODEX_HOME"] == "/home/worker/.codex"
    assert not {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"} & env.keys()
    runtime.client.remove.assert_awaited_once()


async def test_attach_failure_removes_owned_container(tmp_path):
    runtime = backend(tmp_path)
    runtime._await_workspace.side_effect = ControlError("Workspace identity changed")
    attached = Mock(stop=AsyncMock())

    async def discover(kind, command, cwd, env, spawn, **kwargs):
        return await spawn(command, cwd=cwd, env=env)

    with (
        patch("mandri.runtime.docker_command_catalog.discover_commands", discover),
        patch("mandri.runtime.docker_command_catalog.spawn", AsyncMock(return_value=attached)),
        pytest.raises(ControlError, match="Workspace identity"),
    ):
        await DockerCommandDiscovery(runtime).discover(
            HarnessKind.CODEX, ["codex", "app-server"], tmp_path
        )
    attached.stop.assert_awaited_once()
    runtime.client.remove.assert_awaited_once()


async def test_http_catalog_stays_inside_container_and_preserves_response(tmp_path):
    runtime = backend(tmp_path)
    runtime.client.run.return_value = json.dumps(
        {"status": 200, "body": base64.b64encode(b'[{"name":"custom"}]').decode()}
    )
    transport = DockerCatalogTransport(runtime)
    transport.container = "owned-container"
    response = await transport.handle_async_request(
        httpx.Request("GET", "http://127.0.0.1:3000/command?directory=/workspace")
    )
    assert response.json() == [{"name": "custom"}]
    args = runtime.client.run.call_args.args
    assert args[:4] == ("exec", "--env", "MANDRI_CATALOG_HTTP", "owned-container")
    payload = json.loads(runtime.client.run.call_args.kwargs["env"]["MANDRI_CATALOG_HTTP"])
    assert payload["url"] == "http://127.0.0.1:3000/command?directory=/workspace"
    with pytest.raises(ControlError):
        await transport.handle_async_request(httpx.Request("POST", "http://127.0.0.1:3000/session"))


async def test_container_http_preserves_content_encoding(tmp_path):
    runtime = backend(tmp_path)
    body = gzip.compress(b'[{"name":"custom"}]')
    runtime.client.run.return_value = json.dumps(
        {
            "status": 200,
            "body": base64.b64encode(body).decode(),
            "headers": {"content-encoding": "gzip", "content-type": "application/json"},
        }
    )
    transport = DockerCatalogTransport(runtime)
    transport.container = "owned-container"
    response = await transport.handle_async_request(
        httpx.Request("GET", "http://127.0.0.1:3000/command")
    )
    assert response.json() == [{"name": "custom"}]
