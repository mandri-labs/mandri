import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.runtime.control.agy_commands import AgyCommandRunner, command_argv, command_result
from mandri.runtime.control.errors import ControlError, ControlTransportError


def process_result(name="help", data=None, *, status="SUCCESS", code=0):
    envelope = {"status": status, "command": {"name": name, "data": data or {}}}
    return Mock(
        read_stdout_line=AsyncMock(side_effect=[json.dumps(envelope), ""]),
        read_stderr_line=AsyncMock(return_value=""),
        wait=AsyncMock(return_value=code),
        stop=AsyncMock(),
    )


async def test_dynamic_commands_preserve_aliases_and_use_separate_oneshot() -> None:
    catalog = process_result(
        data={
            "commands": [
                {"name": "new-native", "description": "Native inspection", "aliases": ["alias"]}
            ]
        }
    )
    result = process_result("new-native", {"remaining_credits": 12})
    spawn = AsyncMock(side_effect=[catalog, result])
    runner = AgyCommandRunner(
        [
            "agy",
            "-p",
            "--input-format",
            "stream-json",
            "--output-format=stream-json",
            "--conversation",
            "native-session",
            "--gemini_dir",
            "session-profile",
            "--model",
            "selected-model",
        ],
        spawn,
    )
    output = await runner.execute_command("new-native", "")
    assert output == {
        "kind": "fields",
        "title": "New-native",
        "fields": [{"label": "Remaining credits", "value": 12}],
    }
    for call in spawn.call_args_list:
        argv = call.args[0]
        assert "stream-json" not in argv and "--conversation" not in argv
        assert argv[argv.index("--gemini_dir") + 1] == "session-profile"
        assert argv[argv.index("--model") + 1] == "selected-model"
    assert spawn.call_args_list[1].args[0][-4:] == ["-p", "/new-native", "--output-format", "json"]
    catalog.stop.assert_awaited_once_with(grace=1)
    result.stop.assert_awaited_once_with(grace=1)


async def test_unknown_command_never_becomes_prompt() -> None:
    spawn = AsyncMock(return_value=process_result(data={"commands": []}))
    runner = AgyCommandRunner(["agy"], spawn)
    with pytest.raises(ControlError, match="no longer available"):
        await runner.execute_command("unknown", "")
    spawn.assert_awaited_once_with(["agy", "-p", "/help", "--output-format", "json"])


@pytest.mark.parametrize("status,code", [("ERROR", 0), ("SUCCESS", 1), ("WAITING", 0)])
async def test_error_envelope_is_failure_even_when_exit_zero(status, code) -> None:
    process = process_result(status=status, code=code)
    runner = AgyCommandRunner(["agy"], AsyncMock(return_value=process))
    with pytest.raises(ControlError, match="failed") as failure:
        await runner.list_commands()
    assert isinstance(failure.value, ControlTransportError) is (status != "ERROR")
    process.stop.assert_awaited_once()


async def test_argument_mutations_are_not_forwarded() -> None:
    spawn = AsyncMock(return_value=process_result(data={"commands": [{"name": "model"}]}))
    with pytest.raises(ControlError, match="argument interface"):
        await AgyCommandRunner(["agy"], spawn).execute_command("model", "new-model")
    assert spawn.await_count == 1


async def test_cancelled_command_stops_auxiliary_process() -> None:
    ready = asyncio.Event()

    async def block():
        ready.set()
        await asyncio.Event().wait()

    process = process_result()
    process.read_stdout_line = block
    task = asyncio.create_task(
        AgyCommandRunner(["agy"], AsyncMock(return_value=process)).list_commands()
    )
    await asyncio.wait_for(ready.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    process.stop.assert_awaited_once_with(grace=1)


async def test_timeout_reports_uncertain_outcome_and_cleans_up() -> None:
    process = process_result()

    async def block():
        await asyncio.sleep(2)
        return ""

    process.read_stdout_line = block
    runner = AgyCommandRunner(["agy"], AsyncMock(return_value=process), timeout=0.01)
    with pytest.raises(ControlTransportError, match="not retried"):
        await runner.list_commands()
    process.stop.assert_awaited_once_with(grace=1)


def test_config_is_presented_as_fields_without_nested_json() -> None:
    result = command_result(
        "config",
        {"config": {"model": "example", "customModelsConfig": {"secret": "hidden"}}},
        "{raw config}",
    )
    assert result["fields"] == [
        {"label": "Model", "value": "example"},
        {"label": "Custom models", "value": "Details unavailable"},
    ]


def test_unrecognized_structures_never_fall_back_to_json() -> None:
    result = command_result("future", {"nested": {"data": []}}, '{"nested":{}}')
    assert result["kind"] == "notice"
    assert "nested" not in str(result)


def test_strip_all_conversation_and_stream_options() -> None:
    assert command_argv(
        [
            "agy",
            "--print",
            "--continue",
            "--conversation=native",
            "--print-timeout",
            "9",
            "--input-format=stream-json",
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
            "--add-dir",
            "workspace",
        ],
        "/help",
    ) == ["agy", "--add-dir", "workspace", "-p", "/help", "--output-format", "json"]


def test_changelog_is_readable_text_not_a_configuration_field() -> None:
    assert command_result("changelog", {"changelog": "# Release notes"}, None) == {
        "kind": "text",
        "title": "Changelog",
        "text": "# Release notes",
    }


async def test_discovery_transport_failure_does_not_quarantine_an_unsent_command() -> None:
    process = process_result()
    process.read_stdout_line = AsyncMock(side_effect=["invalid", ""])
    runner = AgyCommandRunner(["agy"], AsyncMock(return_value=process))
    with pytest.raises(ControlError, match="was not sent") as failure:
        await runner.execute_command("model", "")
    assert not isinstance(failure.value, ControlTransportError)


@pytest.mark.parametrize(
    "name,data,kind",
    [
        (
            "help",
            {"commands": [{"name": "usage", "aliases": ["quota"], "description": "Usage"}]},
            "list",
        ),
        ("agents", {"agents": []}, "list"),
        ("changelog", {"changelog": "Release notes"}, "text"),
        (
            "config",
            {
                "config": {
                    "model": "example",
                    "gcp": None,
                    "customModelsConfig": {
                        "customModels": {"example": {"modelName": "example-route"}}
                    },
                }
            },
            "fields",
        ),
        ("credits", {"remaining_credits": 0}, "fields"),
        ("effort", {"adjustable": False}, "fields"),
        ("hooks", {"hooks": []}, "list"),
        ("model", {"id": "", "label": "example", "is_default": False}, "fields"),
        (
            "permissions",
            {"permissions": [{"scope": "project"}, {"scope": "shared"}, {"scope": "global"}]},
            "list",
        ),
        (
            "skills",
            {
                "skills": [
                    {
                        "name": "example",
                        "description": "Example skill",
                        "path": "skills/example/SKILL.md",
                        "builtin": True,
                        "model_invocable": True,
                    }
                ]
            },
            "list",
        ),
        ("usage", {"groups": []}, "list"),
    ],
)
def test_qualified_sanitized_headless_result_shapes(name, data, kind) -> None:
    result = command_result(name, data, None)
    assert result["kind"] == kind
    if name == "config":
        assert {"label": "Custom model · example", "value": "example-route"} in result["fields"]
        assert {"label": "Gcp", "value": None} in result["fields"]
    if name == "help":
        assert "Aliases: /quota" in result["items"][0]["description"]
    if name == "skills":
        description = result["items"][0]["description"]
        assert "Built-in skill" in description
        assert "Model can invoke" in description
        assert "skills/example/SKILL.md" in description


def test_unqualified_nonempty_usage_is_honest_and_never_partially_dropped() -> None:
    result = command_result("usage", {"groups": [{"name": "Example"}, {"new_shape": 42}]}, None)
    assert result["kind"] == "notice"
    assert "not supported yet" in result["text"]
