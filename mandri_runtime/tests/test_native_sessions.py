import asyncio
import dataclasses
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import (
    EpochMs,
    HarnessKind,
    HarnessSessionId,
    ProjectPath,
    SessionId,
    SessionState,
)
from mandri.core.launch import merged_env
from mandri.core.ports.control import PromptOutcome, PromptState
from mandri.core.types.availability import SessionOwner
from mandri.core.types.model_selection import ModelSource
from mandri.core.types.sessions import Session
from mandri.providers.errors import ProviderInvalidError
from mandri.runtime.errors import SessionNotRunningError
from mandri.runtime.native_launch import native_launch
from mandri.runtime.service import RuntimeService
from mandri.sessions.ownership.service import NativeOwnership


def record(kind=HarnessKind.CODEX, model="gpt-native", effort="ultra"):
    return Session(
        id=SessionId("session-1"),
        harness=kind,
        native_id=HarnessSessionId("thread-1"),
        native_title=None,
        title_overlay=None,
        project_path=ProjectPath("/workspace"),
        created_at=EpochMs(1),
        updated_at=EpochMs(1),
        state=SessionState.STOPPED,
        model=model,
        gateway_route_id=None,
        deleted=False,
        last_synced_at=EpochMs(1),
        model_source=ModelSource.NATIVE,
        reasoning_effort=effort,
    )


def runtime_for(session):
    sessions = SimpleNamespace(
        create_session=AsyncMock(return_value=session),
        get_session=AsyncMock(return_value=session),
        set_session_state=AsyncMock(),
        native_ownership=AsyncMock(return_value=NativeOwnership(SessionOwner.UNOWNED)),
        set_session_effort=AsyncMock(),
        reveal_native_id=AsyncMock(),
    )
    runtime = RuntimeService(
        harness_commands={session.harness.value: [session.harness.value]},
        sessions=sessions,
        routes=Mock(),
        gateway_port=1234,
        token_issuer=Mock(side_effect=AssertionError("native token issuance")),
    )
    runtime._spawn_harness = AsyncMock(return_value=SimpleNamespace(returncode=None))
    runtime._resolve_metadata = AsyncMock(side_effect=AssertionError("gateway metadata lookup"))
    return runtime, sessions


@pytest.mark.parametrize(
    "kind,model,effort",
    [
        (HarnessKind.CODEX, "gpt-native", "ultra"),
        (HarnessKind.CLAUDE, "fable", "max"),
    ],
)
async def test_native_start_persists_session_without_gateway(kind, model, effort):
    runtime, sessions = runtime_for(record(kind, model, effort))
    launched = await runtime.start_session(
        kind.value,
        model,
        "/workspace",
        effort=effort,
        model_source=ModelSource.NATIVE,
    )
    assert launched.route_id is None
    assert launched.id == "session-1"
    assert sessions.create_session.call_args.args[2] is None
    assert sessions.create_session.call_args.kwargs["model_source"] is ModelSource.NATIVE
    assert sessions.create_session.call_args.kwargs["reasoning_effort"] == effort
    assert runtime._routes.mock_calls == []
    runtime._token_issuer.assert_not_called()
    runtime._resolve_metadata.assert_not_awaited()


@pytest.mark.parametrize("kind", [HarnessKind.CODEX, HarnessKind.CLAUDE])
@pytest.mark.parametrize("model", ["gpt-native", None])
async def test_native_resume_and_message_do_not_create_routes(kind, model):
    session = record(kind, model=model)
    runtime, _ = runtime_for(session)
    control = SimpleNamespace(
        capture_identity=AsyncMock(return_value=session.native_id),
        send_prompt=AsyncMock(return_value=PromptOutcome(PromptState.QUEUED)),
    )
    runtime._session_state(str(session.id)).control = control
    resumed = await runtime.resume_session(str(session.id))
    assert resumed.route_id is None
    assert resumed.id == session.id
    assert resumed.native_id == session.native_id
    await runtime.send_session_prompt(str(session.id), "hello")
    assert runtime._routes.mock_calls == []
    control.send_prompt.assert_awaited_once_with("hello")
    args = runtime._spawn_harness.call_args.args[0]
    if kind is HarnessKind.CLAUDE:
        assert args[args.index("--resume") + 1] == session.native_id
    else:
        assert 'model_provider="openai"' in args
        assert 'model_reasoning_effort="ultra"' in args


async def test_native_effort_never_touches_gateway():
    runtime, sessions = runtime_for(record())
    await runtime.set_session_effort("session-1", "ultra")
    sessions.set_session_effort.assert_awaited_once_with("session-1", "ultra")
    assert runtime._routes.mock_calls == []


async def test_pending_native_change_reuses_session_only_once():
    runtime, _sessions = runtime_for(record())
    runtime._session_state("session-1").launched_model = (ModelSource.NATIVE, "gpt-native", "high")
    runtime.stop_session = AsyncMock()

    async def resume(session_id):
        runtime._session_state(session_id).launched_model = (
            ModelSource.NATIVE,
            "gpt-native",
            "ultra",
        )

    runtime._resume_session = AsyncMock(side_effect=resume)
    await runtime._apply_pending_model("session-1")
    await runtime._apply_pending_model("session-1")
    runtime.stop_session.assert_awaited_once_with("session-1", restore_native=False)
    runtime._resume_session.assert_awaited_once_with("session-1")
    assert runtime._routes.mock_calls == []


async def test_gateway_selection_restarts_to_refresh_launch_metadata():
    session = dataclasses.replace(record(), model_source=ModelSource.GATEWAY, model="provider/new")
    runtime, _ = runtime_for(session)
    runtime._session_state("session-1").launched_model = (ModelSource.GATEWAY, "provider/old", None)
    runtime.stop_session = AsyncMock()
    runtime._resume_session = AsyncMock()
    await runtime._apply_pending_model("session-1")
    runtime.stop_session.assert_awaited_once_with("session-1", restore_native=False)
    runtime._resume_session.assert_awaited_once_with("session-1")


def test_native_launch_preserves_harness_settings_and_removes_gateway_auth():
    parent = {
        "ANTHROPIC_BASE_URL": "http://gateway",
        "ANTHROPIC_API_KEY": "api-key",
        "ANTHROPIC_AUTH_TOKEN": "gateway-token",
        "ANTHROPIC_MODEL": "gateway/model",
        "CLAUDE_CODE_OAUTH_TOKEN": "subscription-token",
        "CLAUDE_CONFIG_DIR": "/config",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-native",
    }
    launch = native_launch(HarnessKind.CLAUDE, "fable", "max")
    env = merged_env(parent, launch)
    assert launch.args == ("--model", "fable", "--effort", "max")
    assert "ANTHROPIC_BASE_URL" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert "ANTHROPIC_MODEL" not in env
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "subscription-token"
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "claude-opus-native"
    assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC" not in env


def test_default_native_launch_resets_claude_model_without_pinning_codex():
    assert native_launch(HarnessKind.CLAUDE).args == ("--model", "default")
    args = native_launch(HarnessKind.CODEX).args
    assert not any(
        arg.startswith("model=") or arg.startswith("model_reasoning_effort=") for arg in args
    )


def test_default_claude_resume_overrides_saved_model_and_clears_gateway_environment():
    launch = native_launch(HarnessKind.CLAUDE, resume_id=HarnessSessionId("native-session"))
    assert launch.args == ("--model", "default", "--resume", "native-session")
    env = merged_env(
        {
            "ANTHROPIC_MODEL": "gateway/old-model",
            "ANTHROPIC_BASE_URL": "http://gateway.invalid",
            "ANTHROPIC_AUTH_TOKEN": "gateway-token",
            "ANTHROPIC_API_KEY": "gateway-key",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "CLAUDE_CODE_USE_VERTEX": "1",
            "CLAUDE_CODE_USE_FOUNDRY": "1",
            "CLAUDE_CODE_USE_ANTHROPIC_AWS": "1",
            "CLAUDE_CODE_OAUTH_TOKEN": "native-subscription-token",
        },
        launch,
    )
    assert env == {"CLAUDE_CODE_OAUTH_TOKEN": "native-subscription-token"}


async def test_opencode_native_rejected_before_spawn():
    runtime = RuntimeService({"opencode": ["opencode"]})
    runtime._spawn_harness = AsyncMock()
    with pytest.raises(ProviderInvalidError):
        await runtime.start_session(
            "opencode", "default", "/workspace", model_source=ModelSource.NATIVE
        )
    runtime._spawn_harness.assert_not_awaited()


async def test_stop_during_resume_kills_new_process_before_initialization():
    runtime, _ = runtime_for(record())
    spawning = asyncio.Event()
    spawned = asyncio.Event()
    process = SimpleNamespace(returncode=None, kill=AsyncMock(return_value=-9))

    async def spawn(*args, **kwargs):
        spawning.set()
        await spawned.wait()
        return process

    runtime._spawn_execution = AsyncMock(side_effect=spawn)
    runtime._finalize_resume = AsyncMock()
    resume = asyncio.create_task(runtime.resume_session("session-1"))
    await asyncio.wait_for(spawning.wait(), 1)
    with pytest.raises(SessionNotRunningError):
        await runtime.stop_session("session-1", force=True, restore_native=False)
    spawned.set()
    with pytest.raises(SessionNotRunningError):
        await resume
    process.kill.assert_awaited_once()
    runtime._finalize_resume.assert_not_awaited()
    assert runtime.registry.process("session-1") is None
