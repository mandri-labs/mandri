import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mandri.core.types.execution import ExecutionBackend, PrivacyMode, ProtectionError
from mandri.core.types.model_selection import ModelSource
from mandri.runtime.control.agy_policy import AgyPolicy
from mandri.runtime.docker_backend import DockerBackend, DockerReadiness
from mandri.runtime.docker_config import DockerConfig
from mandri.runtime.docker_ingress import WorkerIngress
from mandri.runtime.docker_network import network_exception
from mandri.runtime.docker_process import DockerProcess
from mandri.runtime.docker_state import StateLease
from mandri.runtime.docker_workspace import (
    native_state,
    translate_path,
    workspace_root,
    write_context,
)
from mandri.runtime.errors.docker import DockerExecutionError
from mandri.runtime.launch_preparation import LaunchPreparation
from mandri.runtime.service import RuntimeService


@pytest.mark.parametrize("mode", [PrivacyMode.SURROGATE])
async def test_native_protected_launch_is_rejected_before_process_or_scope(mode, monkeypatch):
    scopes = SimpleNamespace(create=AsyncMock())
    runtime = RuntimeService({"codex": ["codex"]}, privacy_scopes=scopes)
    spawn = AsyncMock()
    monkeypatch.setattr(runtime, "_spawn_harness", spawn)
    with pytest.raises(ProtectionError, match="Native models"):
        await runtime.start_session(
            "codex", "native", "/workspace", model_source=ModelSource.NATIVE, privacy_mode=mode
        )
    spawn.assert_not_awaited()
    scopes.create.assert_not_awaited()


async def test_missing_docker_never_falls_back_to_host(monkeypatch):
    runtime = RuntimeService({"codex": ["codex"]})
    spawn = AsyncMock()
    monkeypatch.setattr(runtime, "_spawn_harness", spawn)
    with pytest.raises(DockerExecutionError) as failure:
        await runtime.start_session(
            "codex", "provider/model", "/workspace", execution_backend=ExecutionBackend.DOCKER
        )
    assert failure.value.reason == "docker_unavailable"
    spawn.assert_not_awaited()


def test_worker_preparation_never_inherits_host_credentials(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-provider-canary")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/private/ssh-agent")
    prepared = LaunchPreparation(8000, lambda _: "scope-token", parent_env={}).prepare(
        ["codex"], None, "model", None, False, "route", None, None, {"TASK_VALUE": "selected"}
    )
    assert prepared.env == {"TASK_VALUE": "selected"}


def test_workspace_prefix_translation_preserves_new_nested_descendants(tmp_path):
    root = tmp_path / "Project"
    root.mkdir()
    state = tmp_path / "state"
    assert (
        translate_path(str(root / "mandri/mandri_api/src/new.py"), root, state)
        == "/workspace/mandri/mandri_api/src/new.py"
    )
    assert translate_path(str(tmp_path / "Project-old/secret"), root, state) == str(
        tmp_path / "Project-old/secret"
    )
    assert translate_path("mandri/mandri_api/src", root, state) == "mandri/mandri_api/src"


def test_git_worktree_external_state_is_not_implicitly_mounted(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".git").write_text("gitdir: ../private/common.git")
    with pytest.raises(DockerExecutionError, match="administrative state"):
        workspace_root(root)


def test_missing_workspace_has_typed_error(tmp_path):
    with pytest.raises(DockerExecutionError) as failure:
        workspace_root(tmp_path / "missing")
    assert failure.value.reason == "workspace_unavailable"


def test_cyclic_workspace_link_has_typed_error(tmp_path):
    root = tmp_path / "loop"
    root.symlink_to(root)
    with pytest.raises(DockerExecutionError) as failure:
        workspace_root(root)
    assert failure.value.reason == "workspace_unavailable"


def test_native_state_cannot_escape_or_be_silently_recreated(tmp_path):
    for name in ("../other", "x/y", "", "x\\y"):
        with pytest.raises(DockerExecutionError):
            native_state(tmp_path, name, resume=False)
    with pytest.raises(DockerExecutionError, match="unavailable"):
        native_state(tmp_path, "missing", resume=True)
    target = native_state(tmp_path, "session", resume=False)
    (target / "history").write_text("saved")
    assert native_state(tmp_path, "session", resume=True) == target
    assert (target / "history").read_text() == "saved"


@pytest.mark.skipif(
    sys.platform != "linux", reason="Docker execution requires a qualified Linux host"
)
def test_native_state_lease_excludes_concurrent_writer(tmp_path):
    lease = StateLease(tmp_path / "session.lock")
    try:
        with pytest.raises(DockerExecutionError, match="Another execution"):
            StateLease(tmp_path / "session.lock")
    finally:
        lease.close()
    StateLease(tmp_path / "session.lock").close()


def test_resume_requires_original_workspace_and_pinned_worker_image(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    backend = DockerBackend(DockerConfig("moving-tag", tmp_path / "state"))
    context = backend.context("session", root, resume=False, image_id="sha256:original")
    write_context(Path(context["native_state_root"]), context)
    assert backend.context("session", root, resume=True, image_id="sha256:original") == context
    with pytest.raises(DockerExecutionError, match="original worker image"):
        backend.context("session", root, resume=True, image_id="sha256:changed")
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(DockerExecutionError, match="workspace identity"):
        backend.context("session", other, resume=True, image_id="sha256:original")


def test_workspace_cannot_include_or_reside_inside_native_state_storage(tmp_path):
    state = tmp_path / "state"
    workspace = state / "workspace"
    workspace.mkdir(parents=True)
    backend = DockerBackend(DockerConfig("image", state))
    for root in (tmp_path, state, workspace):
        with pytest.raises(DockerExecutionError, match="must not overlap"):
            backend.context("session", root, resume=False)


@pytest.mark.parametrize(
    "value",
    ["0.0.0.0/0:443", "example.com:443", "127.0.0.1:0", "10.0.0.1:65536", "10.0.0.1:*", "::/0:443"],
)
def test_network_exceptions_require_narrow_literal_destinations(value):
    with pytest.raises(DockerExecutionError):
        network_exception(value)


def test_network_exception_ipv4_and_ipv6_formats():
    assert network_exception("10.23.4.1:443") == ("10.23.4.1/32", 443)
    assert network_exception("[fd01::5]:8443") == ("fd01::5/128", 8443)


def test_agy_container_paths_check_real_host_symlink_target(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    policy = AgyPolicy("acceptEdits", root, {}, runtime_root=Path("/workspace"))
    assert policy.decision("write_to_file", {"TargetFile": "/workspace/src/new.py"}) == "allow"
    assert policy.decision("write_to_file", {"TargetFile": "/workspace/escape/private"}) == "ask"


@pytest.mark.parametrize(
    "path",
    [
        "/v1/runtime/sessions",
        "/v1/gateway/routes",
        "/v1/gateway/llm/another/v1/responses",
        "/v1/gateway/llm/route-other/v1/responses",
        "/v1/gateway/llm/route/../../routes",
    ],
)
async def test_worker_ingress_rejects_control_plane_and_other_routes(path):
    ingress = WorkerIngress(8000, "route", "token")
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=ingress.app), base_url="http://worker"
        ) as client:
            response = await client.post(path, headers={"Authorization": "Bearer token"}, json={})
        assert response.status_code == 403
    finally:
        await ingress.aclose()


async def test_worker_ingress_forwards_stream_unchanged_only_with_scoped_capability():
    ingress = WorkerIngress(8000, "route", "token")
    seen = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"text":"one"}\n\n'
            yield b"data: [DONE]\n\n"

    def upstream(request):
        seen.append(request)
        return httpx.Response(
            200,
            stream=Stream(),
            headers={"content-type": "text/event-stream"},
        )

    await ingress._client.aclose()
    ingress._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=ingress.app), base_url="http://worker"
        ) as client:
            rejected = await client.post("/v1/gateway/llm/route/v1/responses", json={})
            response = await client.post(
                "/v1/gateway/llm/route/v1/responses",
                headers={"Authorization": "Bearer token"},
                json={"input": "original"},
            )
        assert rejected.status_code == 403
        assert response.content == b'data: {"text":"one"}\n\ndata: [DONE]\n\n'
        assert len(seen) == 1
        assert json.loads(seen[0].content) == {"input": "original"}
    finally:
        await ingress.aclose()


async def test_container_exit_status_does_not_use_attach_client_code():
    attach = SimpleNamespace(
        process=SimpleNamespace(), wait=AsyncMock(return_value=0), stop=AsyncMock()
    )
    client = SimpleNamespace(
        json=AsyncMock(return_value={"Running": False, "ExitCode": 137, "OOMKilled": True})
    )
    cleanup = AsyncMock()
    process = DockerProcess(attach, client, "container", cleanup)
    assert await process.wait() == 137
    assert process.returncode == 137
    assert process.oom_killed
    assert await process.stop() == 137
    assert await process.stop() == 137
    cleanup.assert_awaited_once()


@pytest.mark.skipif(
    sys.platform != "linux", reason="Docker execution requires a qualified Linux host"
)
async def test_docker_resource_limits_fail_before_engine_launch(tmp_path):
    backend = DockerBackend(DockerConfig("image", tmp_path, memory_mb=0))
    backend.client = SimpleNamespace(json=AsyncMock())
    with pytest.raises(DockerExecutionError) as failure:
        await backend.readiness()
    assert failure.value.reason == "docker_resource_limit_invalid"
    backend.client.json.assert_not_awaited()


@pytest.mark.skipif(
    sys.platform != "linux", reason="Docker execution requires a qualified Linux host"
)
@pytest.mark.parametrize("native", [False, True])
async def test_docker_create_contract_and_cleanup(tmp_path, monkeypatch, native):
    root = tmp_path / "workspace"
    root.mkdir()
    backend = DockerBackend(DockerConfig("image", tmp_path / "state"))
    commands = []

    async def run(*args, **kwargs):
        commands.append((args, kwargs))
        if args[0] == "ps":
            return ""
        if args[0] == "exec":
            return "172.17.0.1"
        return "MANDRI_NETWORK_READY" if args[0] == "logs" else "created"

    client = SimpleNamespace(
        run=run,
        json=AsyncMock(return_value=[{"Id": "container-id"}]),
        remove=AsyncMock(),
        environment=lambda: {},
        binary="docker",
    )
    backend.client = client
    monkeypatch.setattr(
        backend,
        "readiness",
        AsyncMock(
            return_value=DockerReadiness("sha256:pinned", "engine", "amd64", ("codex",), True)
        ),
    )
    metadata = root.stat()
    stdout = asyncio.StreamReader()
    ending = "\n"
    stdout.feed_data(f"MANDRI_WORKSPACE_READY {metadata.st_dev}:{metadata.st_ino}{ending}".encode())
    attached = SimpleNamespace(process=SimpleNamespace(stdout=stdout))
    monkeypatch.setattr("mandri.runtime.docker_backend.spawn", AsyncMock(return_value=attached))
    captured = {}
    managed = SimpleNamespace()

    def process(attach, docker, container_id, cleanup):
        captured.update(attach=attach, container_id=container_id, cleanup=cleanup)
        return managed

    monkeypatch.setattr("mandri.runtime.docker_backend.DockerProcess", process)
    ingress = SimpleNamespace(port=41000, socket_path=None, aclose=AsyncMock())
    assert (
        await backend.spawn(
            "session",
            "codex",
            ["codex", "app-server"],
            root,
            {"MANDRI_API_KEY": "scoped"},
            ingress,
            native=native,
        )
        is managed
    )
    creates = [(args, options) for args, options in commands if args[0] == "create"]
    guard_args, _ = creates[0]
    args, options = creates[1]
    assert "NET_ADMIN" in guard_args
    assert "NET_ADMIN" not in args
    assert "--privileged" not in args and "--read-only" in args
    assert args[args.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in args
    assert args[args.index("--network") + 1].startswith("container:mandri-")
    assert not any("docker.sock" in value for value in args)
    assert "scoped" not in args
    assert options["env"]["MANDRI_API_KEY"] == "scoped"
    assert "sha256:pinned" in args
    assert any(f"src={root},dst=/workspace" in value for value in args)
    assert "--tty" not in args
    assert ("--init" in args) == native
    assert ("--native-session" in args) == native
    if native:
        assert managed.native_argv[:4] == ["docker", "exec", "--interactive", "container-id"]
        assert managed.native_argv[-2:] == ["codex", "app-server"]
    await captured["cleanup"]()
    client.remove.assert_awaited_once()
    ingress.aclose.assert_awaited_once()
