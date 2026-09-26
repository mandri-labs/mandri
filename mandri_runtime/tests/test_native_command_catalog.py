import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from mandri.core.ids import HarnessKind
from mandri.runtime.control.errors import ControlError
from mandri.runtime.native_command_catalog import discover_commands


class CatalogProcess:
    def __init__(self, responses):
        self.responses = iter(
            response if isinstance(response, str) else json.dumps(response)
            for response in responses
        )
        self.messages = []
        self.stop = AsyncMock()
        self.wait = AsyncMock(return_value=0)

    async def write_stdin(self, payload):
        self.messages.append(json.loads(payload))

    async def read_stdout_line(self):
        return next(self.responses, "")

    async def read_stderr_line(self):
        return ""


async def test_codex_discovers_scoped_skills_without_creating_thread_or_turn():
    process = CatalogProcess(
        [
            {"id": 1, "result": {}},
            {
                "id": 2,
                "result": {
                    "data": [
                        {
                            "errors": [],
                            "skills": [
                                {
                                    "name": "workspace-skill",
                                    "path": "/work/skills/test/SKILL.md",
                                    "description": "Inspect workspace",
                                    "enabled": True,
                                }
                            ],
                        }
                    ]
                },
            },
        ]
    )
    spawn = AsyncMock(return_value=process)
    result = await discover_commands(HarnessKind.CODEX, ["codex", "app-server"], "/work", {}, spawn)
    assert result[0]["name"] == "workspace-skill"
    assert [message["method"] for message in process.messages] == [
        "initialize",
        "initialized",
        "skills/list",
    ]
    assert process.messages[-1]["params"] == {"cwds": ["/work"], "forceReload": True}
    process.stop.assert_awaited_once()


async def test_claude_only_initializes_and_preserves_dynamic_metadata():
    process = CatalogProcess(
        [
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": "1",
                    "response": {
                        "commands": [
                            {
                                "name": "workspace-command",
                                "description": "Native command",
                                "argumentHint": "topic",
                            }
                        ]
                    },
                },
            }
        ]
    )
    spawn = AsyncMock(return_value=process)
    result = await discover_commands(
        HarnessKind.CLAUDE,
        ["claude", "-p"],
        "/work",
        {},
        spawn,
    )
    assert result[0]["name"] == "workspace-command"
    assert result[0]["argument_hint"] == "topic"
    assert process.messages == [
        {
            "type": "control_request",
            "request_id": "1",
            "request": {"subtype": "initialize"},
        }
    ]
    process.stop.assert_awaited_once()

    assert "--no-session-persistence" in spawn.call_args.args[0]


async def test_agy_help_preserves_profile_and_never_starts_stream_conversation():
    process = CatalogProcess(
        [
            {
                "status": "SUCCESS",
                "command": {
                    "name": "help",
                    "data": {"commands": [{"name": "future-command", "description": "Native"}]},
                },
            }
        ]
    )
    spawn = AsyncMock(return_value=process)
    result = await discover_commands(
        HarnessKind.AGY,
        ["agy", "--gemini_dir", "/profiles/catalog", "--input-format", "stream-json"],
        "/work",
        {"AGY_CLI_DISABLE_AUTO_UPDATE": "true"},
        spawn,
    )
    assert result[0]["name"] == "future-command"
    assert spawn.call_args.args[0] == [
        "agy",
        "--gemini_dir",
        "/profiles/catalog",
        "-p",
        "/help",
        "--output-format",
        "json",
    ]
    assert spawn.call_args.kwargs["cwd"] == "/work"
    process.stop.assert_awaited_once()


async def test_opencode_catalog_only_http_get_loopback_and_cleanup():
    process = CatalogProcess(["opencode server listening on http://127.0.0.1:42345"])
    spawn = AsyncMock(return_value=process)
    requests = []
    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        assert kwargs["base_url"] == "http://127.0.0.1:42345"
        assert kwargs["trust_env"] is False
        assert kwargs["auth"][1] == spawn.call_args.kwargs["env"]["OPENCODE_SERVER_PASSWORD"]

        def handle(request):
            requests.append(request)
            return httpx.Response(
                200, json=[{"name": "custom", "description": "Configured command"}]
            )

        kwargs["transport"] = httpx.MockTransport(handle)
        return real_client(**kwargs)

    with patch("mandri.runtime.native_command_catalog.httpx.AsyncClient", client_factory):
        result = await discover_commands(
            HarnessKind.OPENCODE,
            ["opencode", "serve", "--port", "{listen_port}", "--hostname=0.0.0.0"],
            "/work/café",
            {},
            spawn,
        )
    assert result[0]["name"] == "custom"
    assert spawn.call_args.args[0] == [
        "opencode",
        "serve",
        "--hostname",
        "127.0.0.1",
        "--port",
        "0",
        "--mdns=false",
    ]
    assert [(request.method, request.url.path) for request in requests] == [("GET", "/command")]
    assert requests[0].url.params["directory"] == "/work/café"
    process.stop.assert_awaited_once()


@pytest.mark.parametrize("kind", [HarnessKind.CODEX, HarnessKind.CLAUDE, HarnessKind.OPENCODE])
async def test_unavailable_native_catalog_releases_process(kind):
    process = CatalogProcess([])
    with pytest.raises(ControlError):
        await discover_commands(kind, [kind.value], "/work", {}, AsyncMock(return_value=process))
    process.stop.assert_awaited_once()


async def test_cancellation_releases_read_only_discovery_process():
    process = CatalogProcess([])
    pending = asyncio.Event()

    async def wait_forever():
        pending.set()
        await asyncio.Future()

    process.read_stdout_line = wait_forever
    task = asyncio.create_task(
        discover_commands(
            HarnessKind.CODEX,
            ["codex"],
            "/work",
            {},
            AsyncMock(return_value=process),
        )
    )
    await pending.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    process.stop.assert_awaited_once()
