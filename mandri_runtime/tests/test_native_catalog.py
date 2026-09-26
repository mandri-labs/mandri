import json
from unittest.mock import AsyncMock

import pytest
from mandri.core.ids import HarnessKind
from mandri.runtime.control.errors import ControlTransportError
from mandri.runtime.native_catalog import discover_models, parse_models


class CatalogProcess:
    def __init__(self, responses):
        self.responses = iter(json.dumps(response) for response in responses)
        self.messages = []
        self.stop = AsyncMock()

    async def write_stdin(self, payload):
        self.messages.append(json.loads(payload))

    async def read_stdout_line(self):
        return next(self.responses, "")

    async def read_stderr_line(self):
        return ""


async def test_codex_catalog_is_paginated_without_starting_a_thread_or_turn():
    process = CatalogProcess(
        [
            {"id": 1, "result": {}},
            {
                "id": 2,
                "result": {
                    "data": [
                        {
                            "model": "gpt-native",
                            "displayName": "Native Codex",
                            "supportedReasoningEfforts": [
                                {"reasoningEffort": "high"},
                                {"reasoningEffort": "ultra"},
                            ],
                            "defaultReasoningEffort": "high",
                        }
                    ],
                    "nextCursor": "page-2",
                },
            },
            {"id": 3, "result": {"data": [{"model": "gpt-other"}], "nextCursor": None}},
        ]
    )
    spawn = AsyncMock(return_value=process)
    models = await discover_models(HarnessKind.CODEX, ["codex", "app-server"], "/work", {}, spawn)
    assert [model.id for model in models] == ["default", "gpt-native", "gpt-other"]
    assert models[1].reasoning_efforts == ("high", "ultra")
    assert [message["method"] for message in process.messages] == [
        "initialize",
        "initialized",
        "model/list",
        "model/list",
    ]
    assert process.messages[-1]["params"]["cursor"] == "page-2"
    process.stop.assert_awaited_once()


async def test_claude_catalog_preserves_aliases_and_effort_without_prompt():
    process = CatalogProcess(
        [
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": "1",
                    "response": {
                        "models": [
                            {
                                "value": "fable",
                                "displayName": "Fable",
                                "supportedEffortLevels": ["high", "max"],
                            }
                        ]
                    },
                },
            }
        ]
    )
    models = await discover_models(
        HarnessKind.CLAUDE,
        ["claude", "-p"],
        "/work",
        {},
        AsyncMock(return_value=process),
    )
    assert models[1].id == "fable"
    assert models[1].reasoning_efforts == ("high", "max")
    assert process.messages == [
        {
            "type": "control_request",
            "request_id": "1",
            "request": {"subtype": "initialize"},
        }
    ]
    process.stop.assert_awaited_once()


@pytest.mark.parametrize(
    "response",
    [
        {"id": 1, "error": {"message": "not logged in"}},
        {},
    ],
)
async def test_catalog_failure_always_releases_process(response):
    process = CatalogProcess([response])
    with pytest.raises(ControlTransportError):
        await discover_models(
            HarnessKind.CODEX,
            ["codex"],
            "/work",
            {},
            AsyncMock(return_value=process),
        )
    process.stop.assert_awaited_once()


def test_hidden_models_not_advertised_and_custom_efforts_preserved():
    models = parse_models(
        HarnessKind.CODEX,
        [
            {"model": "hidden", "hidden": True},
            {
                "model": "visible",
                "supportedReasoningEfforts": [{"reasoningEffort": "future-level"}],
            },
        ],
    )
    assert [model.id for model in models] == ["default", "visible"]
    assert models[1].reasoning_efforts == ("future-level",)
