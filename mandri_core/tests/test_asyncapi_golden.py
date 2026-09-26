"""Golden AsyncAPI document stability test for the wire contract."""

import json
import pathlib

import pytest
from mandri.core.ids import SessionStopCause
from mandri.core.protocol.asyncapi import build_asyncapi
from mandri.core.protocol.errors import ProtocolErrorCode
from mandri.core.protocol.frames import CLIENT_ADAPTER, SERVER_ADAPTER
from mandri.core.protocol.registry import ACTIONS, TOPICS, ActionSpec, parse_params, topic_slug
from mandri.core.version import __version__
from pydantic import BaseModel, Field

GOLDEN_PATH = pathlib.Path(__file__).parent / "asyncapi_golden.json"


class ContractProbeParams(BaseModel):
    limit: int = Field(ge=1, le=7)


def test_asyncapi_document_is_stable() -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert build_asyncapi(__version__) == golden


def test_asyncapi_document_reflects_app_version() -> None:
    document = build_asyncapi("9.9.9")
    assert document["info"]["version"] == "9.9.9"
    assert document["asyncapi"] == "3.1.0"
    assert document["channels"]["ws"]["address"] == "/v1/ws"


def test_asyncapi_document_covers_all_topics_and_actions() -> None:
    document = build_asyncapi("0.1.0")
    messages = document["components"]["messages"]
    for name in TOPICS:
        assert topic_slug(name) in messages
    schemas = document["components"]["schemas"]
    for spec in ACTIONS.values():
        assert spec.params.__name__ in schemas
        payload = messages[spec.name]["payload"]
        assert payload["properties"]["action"]["const"] == spec.name
        assert payload["properties"]["params"] == {
            "$ref": f"#/components/schemas/{spec.params.__name__}"
        }
        assert document["operations"][spec.name]["messages"] == [
            {"$ref": f"#/channels/ws/messages/{spec.name}"}
        ]


def test_asyncapi_document_covers_runtime_frame_adapters() -> None:
    document = build_asyncapi("0.1.0")
    schemas = document["components"]["schemas"]
    for adapter in (CLIENT_ADAPTER, SERVER_ADAPTER):
        expected = adapter.json_schema(ref_template="#/components/schemas/{model}")["$defs"]
        for name, schema in expected.items():
            assert schemas[name] == schema


def test_asyncapi_document_follows_new_executable_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    action = ActionSpec("contract.probe", ContractProbeParams, "Check live action discovery.")
    monkeypatch.setitem(ACTIONS, action.name, action)

    assert parse_params(action.name, {"limit": 3}) == ContractProbeParams(limit=3)
    document = build_asyncapi("0.1.0")

    schema = document["components"]["schemas"]["ContractProbeParams"]
    assert schema["properties"]["limit"]["minimum"] == 1
    assert schema["properties"]["limit"]["maximum"] == 7
    message = document["components"]["messages"][action.name]
    assert message["description"] == action.description
    assert message["payload"]["properties"]["params"] == {
        "$ref": "#/components/schemas/ContractProbeParams"
    }
    assert "params" in message["payload"]["required"]
    assert action.name in document["operations"]


def test_asyncapi_action_parameters_preserve_runtime_optional_and_required_fields() -> None:
    document = build_asyncapi("0.1.0")
    schemas = document["components"]["schemas"]
    history = schemas["SessionHistoryParams"]
    assert history["required"] == ["session_id"]
    assert history["properties"]["limit"]["maximum"] == 500
    assert history["properties"]["limit"]["minimum"] == 1
    assert "cursor" in history["properties"]
    messages = document["components"]["messages"]
    assert "params" in messages["session.history"]["payload"]["required"]
    assert "params" not in messages["session.list"]["payload"]["required"]


def test_asyncapi_operations_reference_their_channel_messages() -> None:
    document = build_asyncapi("0.1.0")
    channel = document["channels"]["ws"]
    for operation in document["operations"].values():
        assert operation["channel"] == {"$ref": "#/channels/ws"}
        for entry in operation["messages"]:
            reference = entry["$ref"]
            assert reference.startswith("#/channels/ws/messages/")
            name = reference.rsplit("/", 1)[1]
            assert channel["messages"][name] == {"$ref": f"#/components/messages/{name}"}


def test_asyncapi_topic_messages_describe_wire_envelopes() -> None:
    messages = build_asyncapi("0.1.0")["components"]["messages"]
    sessions = messages["sessionsAll"]["payload"]
    assert sessions["properties"]["raw"] == {"$ref": "#/components/schemas/SessionLifecyclePayload"}
    assert {"topic", "seq", "source", "raw", "ts"} <= set(sessions["required"])
    session = messages["session"]["payload"]
    assert {"topic", "seq", "source", "raw", "ts"} <= set(session["required"])
    assert "approval_id" in session["properties"]
    assert "status" in session["properties"]


def test_asyncapi_document_documents_stop_cause_vocabulary() -> None:
    document = build_asyncapi("0.1.0")
    schema = document["components"]["schemas"]["SessionLifecyclePayload"]
    assert "control_lost" in schema["properties"]["type"]["enum"]
    cause_schema = schema["properties"]["cause"]
    assert cause_schema["anyOf"][0]["$ref"] == "#/components/schemas/SessionStopCause"
    assert document["components"]["schemas"]["SessionStopCause"]["enum"] == list(SessionStopCause)
    assert document["components"]["schemas"]["ProtocolErrorCode"]["enum"] == list(ProtocolErrorCode)
