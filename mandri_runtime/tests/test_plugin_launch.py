import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId
from mandri.core.types.execution import ExecutionBackend, PrivacyMode, SessionPolicy
from mandri.runtime.control.codex import CodexControlAdapter
from mandri.runtime.launch_preparation import LaunchPreparation
from mandri.runtime.service import RuntimeService


@pytest.mark.parametrize("kind", [HarnessKind.CODEX, HarnessKind.CLAUDE, HarnessKind.OPENCODE])
@pytest.mark.parametrize("backend", list(ExecutionBackend))
async def test_pseudonymized_launch_keeps_native_extensions(kind, backend, tmp_path, monkeypatch):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = {
        ".codex/config.toml": '[plugins."custom@local"]\nenabled = true\n',
        ".claude/settings.json": json.dumps({"enabledPlugins": {"custom@local": True}}),
        ".config/opencode/opencode.json": json.dumps({"plugin": ["custom-plugin"]}),
    }
    for name, content in settings.items():
        path = home / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (workspace / ".mcp.json").write_text('{"mcpServers":{"custom":{"command":"custom"}}}')
    args = ["--plugin-dir", str(workspace / "plugin")] if kind is HarnessKind.CLAUDE else []
    prepared = LaunchPreparation(8175, lambda _: "token", {"HOME": str(home)}).prepare(
        [kind.value, *args],
        kind,
        "chosen",
        None,
        False,
        "route",
        None,
        None,
        privacy_mode=PrivacyMode.SURROGATE,
    )
    runtime = RuntimeService({}, gateway_port=8175, token_issuer=lambda _: "token")
    process = SimpleNamespace(execution_context={}, container_id="container")
    spawn = AsyncMock(return_value=process)
    if backend is ExecutionBackend.HOST:
        monkeypatch.setattr(runtime, "_spawn_harness", spawn)
    else:
        ingress = SimpleNamespace(start=AsyncMock(return_value=8176), aclose=AsyncMock())
        monkeypatch.setattr("mandri.runtime.service.WorkerIngress", Mock(return_value=ingress))
        runtime._docker = SimpleNamespace(
            owner="owner",
            config=SimpleNamespace(ingress_host="gateway.invalid", ingress_bind="127.0.0.1"),
            readiness=AsyncMock(return_value=SimpleNamespace(image_id="image")),
            context=Mock(return_value={"native_state_root": str(home)}),
            spawn=spawn,
        )
    assert (
        await runtime._spawn_execution(
            "session",
            kind.value,
            prepared,
            workspace,
            SessionPolicy(backend, PrivacyMode.SURROGATE),
            "route",
            "chosen",
            None,
        )
        is process
    )
    spawn.assert_awaited_once()
    argv = spawn.call_args.args[0 if backend is ExecutionBackend.HOST else 2]
    assert argv[0] == kind.value
    assert argv[1 : 1 + len(args)] == args
    assert "--pure" not in argv
    assert "--strict-mcp-config" not in argv
    assert not any("plugins=false" in arg for arg in argv)
    for name, content in settings.items():
        assert (home / name).read_text() == content


@pytest.mark.parametrize("resume", [False, True])
async def test_codex_starts_and_sends_with_native_plugins_without_qualification(resume):
    params = {"modelProvider": "mandri", "config": {"plugins.custom.enabled": True}}
    control = CodexControlAdapter(
        Mock(),
        None,
        params,
        resume_thread_id=HarnessSessionId("thread") if resume else None,
    )
    control._ensure_reader = Mock()
    control._send = AsyncMock()
    control._call = AsyncMock(
        side_effect=[
            {"result": {"userAgent": "codex/999.0.0", "codexHome": "/native/home"}},
            {"result": {"thread": {"id": "thread"}}},
            {"result": {"turn": {"id": "turn"}}},
        ]
    )
    assert await control.capture_identity() == "thread"
    await control.send_prompt("hello")
    calls = control._call.await_args_list
    assert [call.args[0] for call in calls] == [
        "initialize",
        "thread/resume" if resume else "thread/start",
        "turn/start",
    ]
    assert calls[1].args[1] == {**params, **({"threadId": "thread"} if resume else {})}
    await control.aclose()
