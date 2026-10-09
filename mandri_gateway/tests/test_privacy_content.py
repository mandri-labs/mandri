import json
import uuid

import pytest
from mandri.gateway.privacy_protocol import transform_content, validate_request
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope


@pytest.fixture
def engine():
    result = SurrogateEngine(SurrogateScope("content-test"))
    result.register_root("/srv/private-customer/Project")
    result.register("private-customer")
    result.reserve_root("/workspace")
    return result


def protect(body, engine):
    transform_content(body, engine)
    result = transform_content(body, engine)
    assert transform_content(result, engine, restore=True) == body
    return result


@pytest.mark.parametrize("control", ["type", "role", "model", "index", "id", "status"])
@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "gemini"])
def test_application_tool_data_has_no_global_control_key_exemptions(engine, control, protocol):
    data = {control: "private-customer", "nested": {"private-customer": "private-customer"}}
    if protocol == "chat":
        body = {"messages": [{"role": "tool", "tool_call_id": "call_123", "content": data}]}
    elif protocol == "responses":
        body = {"input": [{"type": "function_call_output", "call_id": "call_123", "output": data}]}
    elif protocol == "anthropic":
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call_123", "content": data}
                    ],
                }
            ]
        }
    else:
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"functionResponse": {"name": "lookup", "response": data}}],
                }
            ]
        }
    protected = protect(body, engine)
    assert "private-customer" not in json.dumps(protected)


def test_namespace_schemas_preserve_control_types_and_protect_dynamic_names(engine):
    engine.register("object")
    engine.register("user")
    body = {
        "input": [{"role": "user", "content": "private-customer"}],
        "include": ["reasoning.encrypted_content"],
        "tools": [
            {
                "type": "namespace",
                "name": "team_tools",
                "description": "Tools for private-customer",
                "tools": [
                    {
                        "type": "function",
                        "name": "inspect",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "private-customer": {
                                    "type": "string",
                                    "enum": ["private-customer"],
                                },
                                "image_url": {"type": "string"},
                                "type": {"type": "string"},
                            },
                            "required": ["private-customer"],
                            "additionalProperties": False,
                            "$defs": {"private-customer": {"type": "object"}},
                            "allOf": [{"$ref": "#/$defs/private-customer"}],
                        },
                    }
                ],
            }
        ],
    }
    validate_request(body)
    result = protect(body, engine)
    schema = result["tools"][0]["tools"][0]["parameters"]
    assert schema["type"] == "object"
    assert result["input"][0]["role"] == "user"
    assert "image_url" in schema["properties"]
    assert "type" in schema["properties"]
    assert schema["required"][0] in schema["properties"]
    assert schema["allOf"][0]["$ref"].removeprefix("#/$defs/") in schema["$defs"]
    assert "private-customer" not in json.dumps(result)


def test_metadata_uuid_version_seven_and_window_suffix_restore_exactly(engine):
    identifier = "01994421-6030-7154-a025-58bc742c8152"
    body = {
        "client_metadata": {
            "thread_id": identifier,
            "x-codex-window-id": identifier + ":0",
            "x-codex-turn-metadata": json.dumps(
                {"thread_id": identifier, "type": "private-customer"}
            ),
        }
    }
    result = protect(body, engine)
    metadata = result["client_metadata"]
    alias = metadata["thread_id"]
    assert identifier not in json.dumps(result)
    assert uuid.UUID(alias).version == 7
    assert metadata["x-codex-window-id"] == alias + ":0"
    assert json.loads(metadata["x-codex-turn-metadata"])["thread_id"] == alias


@pytest.mark.parametrize(
    "data",
    [
        {"messages": [], "tools": None, "tool_choice": None, "stream_options": None},
        {
            "contents": [],
            "tools": None,
            "toolConfig": None,
            "safetySettings": None,
            "generationConfig": None,
        },
    ],
)
def test_sdk_null_optional_fields_are_structurally_valid(engine, data):
    assert protect(data, engine) == data


@pytest.mark.parametrize(
    "body",
    [
        {"temperature": "private-customer"},
        {"stream": {"data": "private-customer"}},
        {"messages": [{"role": "private-customer", "content": "hi"}]},
        {"tools": [{"type": "function", "parameters": {"type": "private-customer"}}]},
        {"input": [{"type": "image", "source": {"data": "private-customer"}}]},
        {"input": [{"type": "reasoning", "encrypted_content": "opaque"}]},
        {"tools": [{"type": "computer_use_preview"}]},
        {
            "tools": [
                {
                    "type": "function",
                    "parameters": {"type": "string", "pattern": "private-customer"},
                }
            ]
        },
    ],
)
def test_unrecognized_protocol_controls_pass_through(engine, body):
    protected = transform_content(body, engine)
    assert transform_content(protected, engine, restore=True) == body


def test_reserved_docker_paths_preserve_descendants_despite_identity_collision(engine):
    body = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "/workspace/private-customer/api/src/main.py belongs to private-customer"
                ),
            }
        ]
    }
    protected = protect(body, engine)
    text = protected["messages"][0]["content"]
    assert text.startswith("/workspace/private-customer/api/src/main.py belongs to ")
    assert not text.endswith("private-customer")


def test_incomplete_url_notation_is_prose_not_a_private_authority(engine):
    body = {"instructions": "URLs start with https://... and support https://example.com."}
    assert protect(body, engine) == body


def test_clear_responses_reasoning_replay_preserves_envelopes(engine):
    body = {
        "input": [
            {
                "type": "reasoning",
                "id": "rs_123",
                "content": None,
                "encrypted_content": None,
                "summary": [{"type": "summary_text", "text": "Inspect private-customer"}],
            }
        ]
    }
    validate_request(body)
    protected = protect(body, engine)
    assert protected["input"][0]["type"] == "reasoning"
    assert protected["input"][0]["id"] == "rs_123"
    assert "private-customer" not in json.dumps(protected)


def test_schema_reference_controls_are_contextual_even_when_their_names_are_private(engine):
    engine.register("$defs")
    body = {
        "tools": [
            {
                "type": "function",
                "parameters": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "$defs": {
                        "private-customer": {
                            "type": "object",
                            "properties": {"private-customer": {"type": "string"}},
                        }
                    },
                    "type": "object",
                    "properties": {
                        "item": {"$ref": "#/$defs/private-customer/properties/private-customer"}
                    },
                },
            }
        ]
    }
    result = protect(body, engine)
    schema = result["tools"][0]["parameters"]
    pointer = schema["properties"]["item"]["$ref"].split("/")
    assert pointer[1] == "$defs"
    assert pointer[3] == "properties"
    assert pointer[2] in schema["$defs"]
    assert pointer[4] in schema["$defs"][pointer[2]]["properties"]


@pytest.mark.parametrize(
    "body",
    [
        {"messages": [{"role": {"private": "customer"}}]},
        {"input": [{"type": ["text"]}]},
        {"tools": [{"type": {"sensitive": "value"}}]},
        {"include": [{"private": "customer"}]},
    ],
)
def test_unknown_control_shapes_pass_through(engine, body):
    assert transform_content(body, engine) == body


def test_request_wide_discovery_reaches_metadata_after_earlier_message_urls():
    engine = SurrogateEngine(SurrogateScope("request-discovery"))
    body = {
        "messages": [
            {
                "role": "user",
                "content": "https://service.private.example/api/private-customer/issues",
            }
        ],
        "metadata": {"organization": "private-customer"},
    }
    result = protect(body, engine)
    assert "private-customer" not in json.dumps(result)
    assert result["metadata"]["organization"] in result["messages"][0]["content"]


@pytest.mark.parametrize("keep", ["all", {"type": "thinking_turns", "value": 2}])
def test_claude_context_edit_controls_preserve_full_protected_history(engine, keep):
    body = {
        "model": "synthetic/model",
        "messages": [{"role": "user", "content": "private-customer"}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "metadata": {"user_id": json.dumps({"device_id": "private-customer"})},
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": keep}]},
    }
    validate_request(body)
    result = protect(body, engine)
    assert result["context_management"] == body["context_management"]
    assert "private-customer" not in json.dumps(result)


def test_context_tool_exclusions_share_protected_names_with_tool_definitions(engine):
    body = {
        "messages": [{"role": "user", "content": "private-customer"}],
        "tools": [{"name": "private-customer", "input_schema": {"type": "object"}}],
        "context_management": {
            "edits": [
                {"type": "clear_thinking_20251015"},
                {
                    "type": "clear_tool_uses_20250919",
                    "trigger": {"type": "input_tokens", "value": 1000},
                    "keep": {"type": "tool_uses", "value": 3},
                    "clear_tool_inputs": True,
                    "clear_at_least": {"type": "input_tokens", "value": 500},
                    "exclude_tools": ["private-customer"],
                },
            ]
        },
    }
    result = protect(body, engine)
    assert result["context_management"]["edits"][1]["exclude_tools"] == [result["tools"][0]["name"]]


@pytest.mark.parametrize(
    "context",
    [
        {"id": "remote-context"},
        {"edits": [{"type": "compact_20260112"}]},
        {"edits": [{"type": "clear_thinking_20251015", "keep": "private-customer"}]},
        {
            "edits": [
                {
                    "type": "clear_thinking_20251015",
                    "keep": {"type": "thinking_turns", "value": True},
                }
            ]
        },
        {
            "edits": [
                {"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 0}}
            ]
        },
        {"edits": [{"type": "clear_thinking_20251015", "opaque": "private-customer"}]},
        {
            "edits": [
                {
                    "type": "clear_tool_uses_20250919",
                    "exclude_tools": {"secret": "private-customer"},
                }
            ]
        },
        {"edits": [{"type": "clear_tool_uses_20250919", "clear_tool_inputs": "private-customer"}]},
        {"edits": [{"type": "clear_tool_uses_20250919"}, {"type": "clear_thinking_20251015"}]},
    ],
)
def test_unknown_context_references_and_controls_pass_through(engine, context):
    body = {"context_management": context}
    assert transform_content(body, engine) == body


def test_signed_thinking_preserves_signature_and_transforms_known_text(engine):
    body = {
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "private-customer",
                        "signature": "opaque-signed-state",
                    }
                ],
            }
        ],
    }
    protected = transform_content(body, engine)
    assert transform_content(protected, engine, restore=True) == body


@pytest.mark.parametrize(
    "kind", ["web_search", "web_search_preview", "web_search_preview_2025_03_11"]
)
@pytest.mark.parametrize("external", [True, False])
def test_search_declaration_preserves_access_and_protects_nested_data(engine, kind, external):
    engine.register("approximate")
    engine.register("medium")
    body = {
        "tools": [
            {
                "type": kind,
                "external_web_access": external,
                "search_context_size": "medium",
                "user_location": {"type": "approximate", "city": "private-customer"},
                "filters": {"allowed_domains": ["private-customer.example.com"]},
            }
        ]
    }
    protected = protect(body, engine)
    tool = protected["tools"][0]
    assert tool["type"] == kind
    assert tool["external_web_access"] is external
    assert tool["search_context_size"] == "medium"
    assert tool["user_location"]["type"] == "approximate"
    assert "private-customer" not in json.dumps(protected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("external_web_access", "false"),
        ("search_context_size", "huge"),
        ("user_location", {"type": "exact"}),
    ],
)
def test_unknown_search_control_values_pass_through(engine, field, value):
    body = {"tools": [{"type": "web_search", field: value}]}
    assert transform_content(body, engine) == body


def test_search_history_preserves_action_kind_and_protects_queries(engine):
    engine.register("search")
    body = {
        "input": [
            {
                "type": "web_search_call",
                "id": "ws_123",
                "action": {"type": "search", "query": "private-customer"},
            }
        ]
    }
    protected = protect(body, engine)
    assert protected["input"][0]["action"]["type"] == "search"
    assert "private-customer" not in json.dumps(protected)


@pytest.mark.parametrize("protocol", ["chat", "responses", "anthropic", "gemini"])
def test_images_pass_through_while_known_caption_text_is_replaced(engine, protocol):
    encoded = "cHJpdmF0ZS1jdXN0b21lcg=="
    caption = "Inspect private-customer"
    if protocol == "gemini":
        image = {"inlineData": {"mimeType": "image/png", "data": encoded}}
        body = {"contents": [{"role": "user", "parts": [{"text": caption}, image]}]}
        collection, parts = "contents", "parts"
    elif protocol == "anthropic":
        image = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": encoded},
        }
        body = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": caption}, image]}]
        }
        collection, parts = "messages", "content"
    elif protocol == "responses":
        image = {"type": "input_image", "image_url": "data:image/png;base64," + encoded}
        body = {
            "input": [{"role": "user", "content": [{"type": "input_text", "text": caption}, image]}]
        }
        collection, parts = "input", "content"
    else:
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}
        body = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": caption}, image]}]
        }
        collection, parts = "messages", "content"
    transformed = protect(body, engine)
    blocks = transformed[collection][0][parts]
    assert blocks[1] == image
    assert "private-customer" not in blocks[0]["text"]
    assert blocks[0]["text"] != caption


@pytest.mark.parametrize("kind", ["function_call_output", "custom_tool_call_output"])
@pytest.mark.parametrize(
    "image",
    [
        {"image_url": "data:image/png;base64,cHJpdmF0ZS1jdXN0b21lcg=="},
        {"image_url": "https://private-customer.example.invalid/image.png"},
        {"file_id": "file-private-customer"},
    ],
)
@pytest.mark.parametrize("existing_mappings", [False, True])
def test_responses_tool_images_preserve_payload_and_controls(
    engine, kind, image, existing_mappings
):
    if existing_mappings:
        engine.protect_text("data:image/png")
        for value in ("input_image", "input_text", "high", "cHJpdmF0ZS1jdXN0b21lcg"):
            engine.register(value)
    image = {"type": "input_image", "detail": "high", **image}
    body = {
        "input": [
            {
                "type": kind,
                "call_id": "call_image",
                "output": [{"type": "input_text", "text": "Inspect private-customer"}, image],
            }
        ]
    }
    protected = protect(body, engine)
    output = protected["input"][0]["output"]
    if str(image.get("image_url", "")).startswith("https://"):
        assert output[1]["image_url"].startswith("https://")
        assert "private-customer" not in output[1]["image_url"]
        assert output[1]["detail"] == image["detail"]
    else:
        assert output[1] == image
    assert output[0]["type"] == "input_text"
    assert "private-customer" not in output[0]["text"]


@pytest.mark.parametrize("kind", ["function_call_output", "custom_tool_call_output"])
def test_responses_tool_application_arrays_still_transform(engine, kind):
    body = {
        "input": [
            {
                "type": kind,
                "call_id": "call_data",
                "output": [
                    {"type": "private-customer", "name": "private-customer"},
                    {"type": ["private-customer"], "image_url": "private-customer"},
                    "private-customer",
                ],
            }
        ]
    }
    assert "private-customer" not in json.dumps(protect(body, engine))
