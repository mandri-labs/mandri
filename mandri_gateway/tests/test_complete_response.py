import json

import pytest
from mandri.gateway.complete_response import complete_sse
from mandri.gateway.privacy_protocol import GatewayProtocol
from mandri.gateway.privacy_response_json import restore_json
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope


def events(wire):
    return [
        json.loads(line[6:])
        for line in wire.decode().splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def test_responses_preserves_interleaved_tool_identities_and_complete_arguments():
    engine = SurrogateEngine(SurrogateScope("complete-tools"))
    alias = engine.protect_text("customer-721@example.invalid")
    value = {
        "id": "resp_one",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "id": "fc_one",
                "type": "function_call",
                "call_id": "call_one",
                "name": "read_file",
                "arguments": json.dumps({"nested": {"owner": alias}}),
            },
            {
                "id": "custom_two",
                "type": "custom_tool_call",
                "call_id": "call_two",
                "name": "apply_patch",
                "input": alias,
            },
        ],
        "usage": {"input_tokens": 4, "output_tokens": 8},
    }
    restored = restore_json(value, engine)
    output = events(complete_sse(restored, GatewayProtocol.RESPONSES))
    assert output[-1]["response"] == restored
    assert [item["sequence_number"] for item in output] == list(range(len(output)))
    function = next(
        item for item in output if item["type"] == "response.function_call_arguments.delta"
    )
    assert function["item_id"] == "fc_one"
    assert json.loads(function["delta"])["nested"]["owner"] == "customer-721@example.invalid"
    custom = next(
        item for item in output if item["type"] == "response.custom_tool_call_input.delta"
    )
    assert custom["item_id"] == "custom_two"
    assert custom["delta"] == "customer-721@example.invalid"


def test_anthropic_tools_have_empty_start_and_complete_json_delta():
    value = {
        "type": "message",
        "id": "msg_one",
        "content": [
            {
                "type": "tool_use",
                "id": "tool_one",
                "name": "read_file",
                "input": {"nested": ["file.py"]},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    output = events(complete_sse(value, GatewayProtocol.ANTHROPIC))
    assert output[1]["content_block"]["input"] == {}
    assert json.loads(output[2]["delta"]["partial_json"]) == {"nested": ["file.py"]}
    assert output[-1]["type"] == "message_stop"


@pytest.mark.parametrize("protocol", [GatewayProtocol.CHAT, GatewayProtocol.GEMINI])
def test_complete_other_protocols_preserve_tool_payload(protocol):
    if protocol is GatewayProtocol.CHAT:
        value = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "tool_one",
                                "type": "function",
                                "function": {"name": "read", "arguments": '{"path":"file.py"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        output = events(complete_sse(value, protocol))[0]
        assert output["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
        assert output["choices"][0]["finish_reason"] == "tool_calls"
    else:
        value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"functionCall": {"name": "read", "args": {"path": "file.py"}}}]
                    }
                }
            ]
        }
        assert events(complete_sse(value, protocol)) == [value]


def test_nullable_sdk_tool_calls_are_valid_in_reconstructed_chat_stream():
    response = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "synthetic reply",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ]
    }
    chunks = events(complete_sse(response, GatewayProtocol.CHAT))
    assert chunks[0]["choices"][0]["delta"]["content"] == "synthetic reply"
    assert chunks[0]["choices"][0]["delta"]["tool_calls"] is None
