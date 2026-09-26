import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId
from mandri.core.launch import merged_env
from mandri.core.types.execution import PrivacyMode
from mandri.core.types.model_selection import ModelSource
from mandri.runtime.launch_preparation import LaunchPreparation
from mandri.runtime.model_selection import ModelSelectionService
from mandri.runtime.native_catalog import discover_models, parse_models
from mandri.runtime.native_command_catalog import discover_commands
from mandri.runtime.native_launch import native_launch
from mandri.runtime.session_state import RuntimeStates


class CatalogProcess:
    def __init__(self, command, data):
        self.command = command
        self.data = data
        self.messages = []
        self.stop = AsyncMock()

    async def write_stdin(self, payload):
        self.messages.append(json.loads(payload))

    async def read_stdout_line(self):
        return (
            json.dumps(
                {
                    "id": self.messages[-1]["id"],
                    "type": "response",
                    "command": self.command,
                    "success": True,
                    "data": self.data,
                }
            )
            + "\n"
        )

    async def read_stderr_line(self):
        return ""


async def test_pi_native_catalog_preserves_custom_provider_and_model_names():
    process = CatalogProcess(
        "get_available_models",
        {"models": [{"provider": "custom", "id": "org/model", "name": "User model"}]},
    )
    rows = await discover_models(
        HarnessKind.PI, ["pi", "--mode", "rpc"], "/workspace", {}, AsyncMock(return_value=process)
    )
    assert [row.id for row in rows] == ["default", "custom/org/model"]
    assert rows[1].display_name == "User model"
    assert process.messages == [{"type": "get_available_models", "id": "1"}]
    process.stop.assert_awaited_once()


async def test_pi_command_probe_keeps_user_resources_and_never_sends_a_prompt():
    process = CatalogProcess(
        "get_commands",
        {"commands": [{"name": "skill:custom", "source": "skill", "description": "Custom"}]},
    )
    spawn = AsyncMock(return_value=process)
    rows = await discover_commands(
        HarnessKind.PI,
        ["pi", "--mode", "rpc", "--extension", "/extensions/custom.ts"],
        "/workspace",
        {"PI_CODING_AGENT_DIR": "/profile"},
        spawn,
    )
    assert any(row["name"] == "skill:custom" for row in rows)
    assert spawn.call_args.args[0] == [
        "pi",
        "--mode",
        "rpc",
        "--extension",
        "/extensions/custom.ts",
        "--no-session",
    ]
    assert spawn.call_args.kwargs["env"]["PI_CODING_AGENT_DIR"] == "/profile"
    assert all(message["type"] == "get_commands" for message in process.messages)
    process.stop.assert_awaited_once()


@pytest.mark.parametrize("privacy", list(PrivacyMode))
def test_pi_gateway_launch_preserves_extensions_and_resumes_exact_session(tmp_path, privacy):
    profile = tmp_path / "profile"
    profile.mkdir()
    settings = profile / "settings.json"
    settings.write_text('{"extensions":["/custom/plugin.ts"]}')
    command = ["pi", "--mode", "rpc", "--extension", "/custom/plugin.ts"]
    prepared = LaunchPreparation(
        8175,
        lambda _: "scoped-token",
        {"PI_CODING_AGENT_DIR": str(profile), "CUSTOM_EXTENSION_FLAG": "enabled"},
    ).prepare(
        command,
        HarnessKind.PI,
        "provider/model",
        "high",
        False,
        "route",
        None,
        None,
        resume_native_id=HarnessSessionId("conversation"),
        privacy_mode=privacy,
    )
    assert prepared.argv[: len(command)] == command
    assert prepared.argv[-2:] == ["--session", "conversation"]
    assert prepared.argv.count("--thinking") == 1
    assert prepared.argv[prepared.argv.index("--thinking") + 1] == "high"
    assert prepared.env["PI_CODING_AGENT_DIR"] == str(profile)
    assert prepared.env["CUSTOM_EXTENSION_FLAG"] == "enabled"
    assert prepared.env["MANDRI_PI_BASE_URL"] == "http://127.0.0.1:8175/v1/gateway/llm/route/v1"
    assert settings.read_text() == '{"extensions":["/custom/plugin.ts"]}'
    assert not any(arg.startswith("--no-") for arg in prepared.argv)


def test_pi_native_launch_keeps_auth_and_resources_without_gateway_override():
    launch = native_launch(HarnessKind.PI, "custom/model", "high", HarnessSessionId("native"))
    assert launch.args == ("--model", "custom/model", "--thinking", "high", "--session", "native")
    assert merged_env(
        {
            "PI_CODING_AGENT_DIR": "/custom",
            "CUSTOM_API_KEY": "user-key",
            "MANDRI_API_KEY": "old",
            "MANDRI_PI_MODEL": "old",
            "MANDRI_PI_BASE_URL": "old",
        },
        launch,
    ) == {"PI_CODING_AGENT_DIR": "/custom", "CUSTOM_API_KEY": "user-key"}


async def test_pi_model_probe_is_ephemeral_and_uses_native_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi-profile"))
    process = CatalogProcess("get_available_models", {"models": []})
    spawn = AsyncMock(return_value=process)
    service = ModelSelectionService({"pi": ["pi", "--mode", "rpc"]}, RuntimeStates(), None)
    await service.native_models("pi", str(tmp_path), spawn)
    assert "--no-session" in spawn.call_args.args[0]
    assert spawn.call_args.kwargs["env"]["PI_CODING_AGENT_DIR"] == str(tmp_path / "pi-profile")


async def test_pi_gateway_model_or_thinking_change_restarts_for_updated_capabilities():
    states = RuntimeStates()
    states.session("session").launched_model = (ModelSource.GATEWAY, "provider/old", None)
    record = SimpleNamespace(
        harness=HarnessKind.PI,
        native_id="native",
        model_source=ModelSource.GATEWAY,
        model="provider/new",
        reasoning_effort="high",
    )
    service = ModelSelectionService(
        {}, states, SimpleNamespace(get_session=AsyncMock(return_value=record))
    )
    assert await service.needs_restart("session")


def test_pi_host_resume_resolves_session_id_without_assuming_filename(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))
    path = tmp_path / "sessions" / "project" / "arbitrary.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "type": "session",
                "version": 3,
                "id": "native",
                "cwd": "/workspace",
                "timestamp": "2026-09-25T12:00:00Z",
            }
        )
        + "\n"
    )
    prepared = LaunchPreparation(None, None).prepare(
        ["pi", "--mode", "rpc"],
        HarnessKind.PI,
        "default",
        None,
        True,
        None,
        None,
        None,
        resume_native_id=HarnessSessionId("native"),
    )
    assert prepared.argv[-2:] == ["--session", str(path)]
    assert Path(prepared.argv[-1]).is_file()


def test_pi_native_thinking_catalog_respects_custom_capability_mapping():
    rows = parse_models(
        HarnessKind.PI,
        [
            {
                "provider": "custom",
                "id": "reasoner",
                "reasoning": True,
                "thinkingLevelMap": {"minimal": None, "low": None, "xhigh": "maximum"},
            },
            {"provider": "custom", "id": "plain", "reasoning": False},
        ],
    )
    assert rows[1].reasoning_efforts == ("off", "medium", "high", "xhigh")
    assert rows[2].reasoning_efforts == ()
