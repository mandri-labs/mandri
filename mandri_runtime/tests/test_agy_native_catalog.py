import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId
from mandri.core.launch import merged_env
from mandri.runtime.control.errors import ControlTransportError
from mandri.runtime.model_selection import ModelSelectionService
from mandri.runtime.native_catalog import discover_models
from mandri.runtime.native_launch import native_launch
from mandri.runtime.session_state import RuntimeStates


def process_with(lines: list[str], code: int = 0):
    remaining = iter([*lines, ""])
    return Mock(
        read_stdout_line=AsyncMock(side_effect=lambda: next(remaining, "")),
        read_stderr_line=AsyncMock(return_value=""),
        write_stdin=AsyncMock(),
        wait=AsyncMock(return_value=code),
        stop=AsyncMock(),
    )


async def test_agy_catalog_preserves_native_slugs_without_inventing_effort_metadata() -> None:
    process = process_with(
        [
            "gemini-example-high\tGemini Example (High)\n",
            "gemini-example-medium\tGemini Example (Medium)\n",
            "claude-example\tClaude Example (Thinking)\n",
        ]
    )
    models = await discover_models(
        HarnessKind.AGY,
        ["agy", "models"],
        "synthetic-workspace",
        {},
        AsyncMock(return_value=process),
    )
    assert [model.id for model in models] == [
        "default",
        "gemini-example-high",
        "gemini-example-medium",
        "claude-example",
    ]
    assert models[1].display_name == "Gemini Example (High)"
    assert all(model.reasoning_efforts == () and model.default_effort is None for model in models)
    process.write_stdin.assert_not_called()
    process.stop.assert_awaited_once_with(grace=1)


@pytest.mark.parametrize("lines,code", [([], 1), (["Sign in to Google\n"], 0), (["m\tModel"], 1)])
async def test_agy_missing_auth_or_invalid_catalog_closes_probe(
    lines: list[str], code: int
) -> None:
    process = process_with(lines, code)
    with pytest.raises(ControlTransportError, match="Google sign-in"):
        await discover_models(
            HarnessKind.AGY,
            ["agy", "models"],
            "synthetic-workspace",
            {},
            AsyncMock(return_value=process),
        )
    process.stop.assert_awaited_once_with(grace=1)
    process.write_stdin.assert_not_called()


async def test_cancelled_catalog_probe_closes_process() -> None:
    reading = asyncio.Event()

    async def read() -> str:
        reading.set()
        await asyncio.Event().wait()
        return ""

    process = process_with([])
    process.read_stdout_line = read
    pending = asyncio.create_task(
        discover_models(
            HarnessKind.AGY,
            ["agy", "models"],
            "synthetic-workspace",
            {},
            AsyncMock(return_value=process),
        )
    )
    await asyncio.wait_for(reading.wait(), 1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    process.stop.assert_awaited_once_with(grace=1)


async def test_catalog_uses_google_profile_and_strips_gateway_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_GENAI_USE_VERTEXAI",
    ):
        monkeypatch.setenv(key, "synthetic-gateway-value")
    monkeypatch.setenv("SYNTHETIC_PRESERVED_SETTING", "keep")
    prepare = Mock(return_value=Path("synthetic-profile"))
    monkeypatch.setattr("mandri.runtime.model_selection.prepare_agy_profile", prepare)
    service = ModelSelectionService(
        {"agy": ["agy", "-p", "--input-format", "stream-json"]},
        RuntimeStates(),
        None,
        Path("synthetic-native-root"),
        Path("synthetic-profiles-root"),
    )
    process = process_with(["gemini-example\tGemini Example\n"])
    spawn = AsyncMock(return_value=process)
    models = await service.native_models("agy", "synthetic-workspace", spawn)
    assert models[1].id == "gemini-example"
    prepare.assert_called_once_with(
        Path("synthetic-profiles-root"),
        "catalog",
        native=True,
        canonical_root=Path("synthetic-native-root"),
    )
    assert spawn.call_args.args == (["agy", "--gemini_dir", "synthetic-profile", "models"],)
    env = spawn.call_args.kwargs["env"]
    assert env["SYNTHETIC_PRESERVED_SETTING"] == "keep"
    assert all(
        key not in env
        for key in (
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "GOOGLE_GEMINI_BASE_URL",
            "GOOGLE_GENAI_USE_VERTEXAI",
        )
    )
    assert await service.native_models("agy", "synthetic-workspace", spawn) == models
    spawn.assert_awaited_once()


def test_explicit_native_effort_and_resume_are_forwarded_without_gateway_auth() -> None:
    plan = native_launch(HarnessKind.AGY, "gemini-example", "high", HarnessSessionId("native-id"))
    assert plan.args == (
        "--model",
        "gemini-example",
        "--effort",
        "high",
        "--conversation",
        "native-id",
    )
    assert merged_env({"GEMINI_API_KEY": "synthetic"}, plan, None) == {
        "AGY_CLI_DISABLE_AUTO_UPDATE": "true",
    }
