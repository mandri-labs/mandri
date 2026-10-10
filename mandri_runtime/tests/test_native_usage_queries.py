import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from mandri.core.ids import HarnessKind
from mandri.runtime.control.errors import ControlTransportError
from mandri.runtime.native_usage import read_native_usage

pytestmark = pytest.mark.usefixtures("agy_native_credentials")


class QuotaProcess:
    def __init__(self, responses):
        self.responses = iter(json.dumps(response) for response in responses)
        self.messages = []
        self.stop = AsyncMock()
        self.wait = AsyncMock(return_value=0)

    async def write_stdin(self, payload):
        self.messages.append(json.loads(payload))

    async def read_stdout_line(self):
        return next(self.responses, "")

    async def read_stderr_line(self):
        return ""


async def test_codex_queries_quotas_without_a_thread_or_prompt():
    process = QuotaProcess(
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"rateLimits": {"primary": {"usedPercent": 42}}}},
        ]
    )
    result = await read_native_usage(
        HarnessKind.CODEX,
        "profile",
        ["codex", "app-server"],
        Path("/empty"),
        {},
        AsyncMock(return_value=process),
    )
    assert result[0].windows[0]["used_percent"] == 42
    assert [message["method"] for message in process.messages] == [
        "initialize",
        "initialized",
        "account/rateLimits/read",
    ]
    process.stop.assert_awaited_once()


async def test_claude_quotas_use_control_request_without_reading_transcripts_or_prompting():
    process = QuotaProcess(
        [
            {
                "type": "control_response",
                "response": {"request_id": "1", "subtype": "success", "response": {}},
            },
            {
                "type": "control_response",
                "response": {
                    "request_id": "2",
                    "subtype": "success",
                    "response": {
                        "subscription_type": "max",
                        "rate_limits": {
                            "five_hour": {"utilization": 0, "resets_at": "2026-10-01T10:00:00Z"},
                            "seven_day": {"utilization": 54},
                        },
                    },
                },
            },
        ]
    )
    result = await read_native_usage(
        HarnessKind.CLAUDE,
        "profile",
        ["claude"],
        Path("/empty"),
        {},
        AsyncMock(return_value=process),
    )
    assert result[0].plan == "max"
    assert [window["used_percent"] for window in result[0].windows] == [0, 54]
    assert [message["request"]["subtype"] for message in process.messages] == [
        "initialize",
        "get_usage",
    ]
    assert process.messages[1]["request"]["skip_behaviors"] is True
    process.stop.assert_awaited_once()


async def test_agy_reads_structured_command_groups_without_using_response_text():
    process = QuotaProcess(
        [
            {
                "command": {
                    "data": {
                        "plan_name": "Google AI Ultra",
                        "groups": [
                            {
                                "name": "Gemini",
                                "buckets": [
                                    {
                                        "window": "WEEKLY",
                                        "remaining_fraction": 0.75,
                                        "reset_time": "2026-10-01T10:00:00Z",
                                    }
                                ],
                            }
                        ],
                    }
                },
                "response": "private text",
            }
        ]
    )
    result = await read_native_usage(
        HarnessKind.AGY,
        "profile",
        ["agy", "-p", "/usage"],
        Path("/empty"),
        {},
        AsyncMock(return_value=process),
    )
    assert result[0].windows[0]["remaining_fraction"] == 0.75
    assert result[0].plan == "Google AI Ultra"
    assert "private text" not in repr(result)
    assert process.messages == []
    process.stop.assert_awaited_once()


@pytest.mark.parametrize("interactive", [None, "true", "false"])
async def test_agy_quota_reads_force_noninteractive_authentication(interactive):
    process = QuotaProcess([{"status": "SUCCESS", "command": {"data": {}}}])
    spawn = AsyncMock(return_value=process)
    env = {"PATH": "synthetic", "AGY_CLI_NONINTERACTIVE_HEADLESS": "false"}
    if interactive is not None:
        env["AGY_CLI_INTERACTIVE_HEADLESS"] = interactive
    original = dict(env)
    await read_native_usage(
        HarnessKind.AGY, "profile", ["agy", "-p", "/usage"], Path("/empty"), env, spawn
    )
    launched = spawn.call_args.kwargs["env"]
    assert launched["AGY_CLI_NONINTERACTIVE_HEADLESS"] == "true"
    assert launched["AGY_CLI_DISABLE_AUTO_UPDATE"] == "true"
    assert "AGY_CLI_INTERACTIVE_HEADLESS" not in launched
    assert launched["PATH"] == "synthetic"
    assert env == original


async def test_agy_login_prompt_fails_immediately_and_stops_the_process():
    process = QuotaProcess([])

    async def pending_stdout():
        await asyncio.Event().wait()

    process.read_stdout_line = pending_stdout
    process.read_stderr_line = AsyncMock(
        side_effect=["Authentication required. Please visit the URL to log in:", ""]
    )
    async with asyncio.timeout(1):
        with pytest.raises(PermissionError, match="Antigravity CLI authentication required"):
            await read_native_usage(
                HarnessKind.AGY,
                "profile",
                ["agy"],
                Path("/empty"),
                {},
                AsyncMock(return_value=process),
            )
    process.stop.assert_awaited_once()
    process.wait.assert_not_awaited()


async def test_agy_authentication_error_is_not_accepted_as_a_quota_result():
    process = QuotaProcess(
        [{"status": "ERROR", "error": "authentication failed or timed out", "response": ""}]
    )
    process.wait.return_value = 1
    with pytest.raises(PermissionError, match="Antigravity CLI authentication required"):
        await read_native_usage(
            HarnessKind.AGY,
            "profile",
            ["agy"],
            Path("/empty"),
            {},
            AsyncMock(return_value=process),
        )
    process.stop.assert_awaited_once()


async def test_pi_quotas_are_separate_per_provider_and_require_no_conversation():
    process = QuotaProcess(
        [
            {"type": "irrelevant"},
            {
                "type": "mandri_usage",
                "accounts": [
                    {"provider": "anthropic", "data": {"five_hour": {"utilization": 12}}},
                    {
                        "provider": "openai-codex",
                        "data": {
                            "rate_limit": {
                                "primary_window": {
                                    "used_percent": 34,
                                    "limit_window_seconds": 18000,
                                }
                            }
                        },
                    },
                ],
            },
        ]
    )
    accounts = await read_native_usage(
        HarnessKind.PI, "pi-profile", ["pi"], Path("/empty"), {}, AsyncMock(return_value=process)
    )
    assert [account.account_id for account in accounts] == [
        "pi-profile:anthropic",
        "pi-profile:openai-codex",
    ]
    assert [account.windows[0]["used_percent"] for account in accounts] == [12, 34]
    assert process.messages == []
    process.stop.assert_awaited_once()


@pytest.mark.parametrize("failure", [ControlTransportError("closed"), asyncio.CancelledError()])
async def test_quota_process_is_stopped_even_on_cancellation_or_protocol_failure(failure):
    process = QuotaProcess([])
    process.read_stdout_line = AsyncMock(side_effect=failure)
    with pytest.raises(type(failure)):
        await read_native_usage(
            HarnessKind.CODEX,
            "profile",
            ["codex"],
            Path("/empty"),
            {},
            AsyncMock(return_value=process),
        )
    process.stop.assert_awaited_once()
