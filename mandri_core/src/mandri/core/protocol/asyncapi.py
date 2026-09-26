"""AsyncAPI document generation from executable websocket contracts."""

from collections.abc import Iterator
from copy import deepcopy
from typing import Any

from mandri.core.protocol.frames import CLIENT_ADAPTER, SERVER_ADAPTER
from mandri.core.protocol.registry import ACTIONS, TOPICS, ActionSpec, TopicSpec, topic_slug
from pydantic import BaseModel, TypeAdapter

API_VERSION = "3.1.0"
SCHEMA_REF = "#/components/schemas/{model}"


def build_asyncapi(app_version: str) -> dict[str, Any]:
    schemas: dict[str, Any] = {}
    messages: dict[str, Any] = {}
    for adapter in (CLIENT_ADAPTER, SERVER_ADAPTER):
        _add_frames(schemas, messages, adapter)
    for name, spec in TOPICS.items():
        _add_schema(schemas, spec.payload)
        messages[topic_slug(name)] = {
            "title": name,
            "description": spec.description,
            "payload": _topic_payload(spec, schemas),
        }
    for name, action in ACTIONS.items():
        _add_schema(schemas, action.params)
        if action.result is not None:
            _add_schema(schemas, action.result)
            response = deepcopy(schemas["ResponseFrame"])
            response["properties"]["result"] = {
                "anyOf": [
                    {"$ref": SCHEMA_REF.format(model=action.result.__name__)},
                    {"type": "null"},
                ]
            }
            messages[f"{name}.response"] = {
                "payload": response,
                "correlationId": {"location": "$message.payload#/op_id"},
            }
        messages[name] = {
            "title": name,
            "description": action.description,
            "payload": _action_payload(action, schemas["RequestFrame"]),
            "correlationId": {"location": "$message.payload#/op_id"},
        }
    return {
        "asyncapi": API_VERSION,
        "info": {
            "title": "Mandri Daemon",
            "version": app_version,
            "description": "Local control-plane daemon unifying coding harnesses.",
        },
        "defaultContentType": "application/json",
        "channels": {
            "ws": {
                "address": "/v1/ws",
                "title": "Live feed",
                "description": "Topic-based websocket feed carrying raw harness events.",
                "bindings": {"ws": {"method": "GET"}},
                "messages": {name: {"$ref": f"#/components/messages/{name}"} for name in messages},
            }
        },
        "operations": _operations(),
        "components": {"messages": messages, "schemas": schemas},
    }


def _add_schema(schemas: dict[str, Any], model: type[BaseModel]) -> None:
    schema = model.model_json_schema(ref_template=SCHEMA_REF)
    schemas.update(schema.pop("$defs", {}))
    schemas[model.__name__] = schema


def _variant_refs(schema: dict[str, Any]) -> Iterator[str]:
    if "$ref" in schema:
        yield schema["$ref"]
    for variant in schema.get("anyOf", schema.get("oneOf", [])):
        yield from _variant_refs(variant)


def _add_frames(
    schemas: dict[str, Any], messages: dict[str, Any], adapter: TypeAdapter[Any]
) -> None:
    document = adapter.json_schema(ref_template=SCHEMA_REF)
    schemas.update(document.pop("$defs", {}))
    for reference in _variant_refs(document):
        model_name = reference.rsplit("/", 1)[1]
        properties = schemas[model_name]["properties"]
        name = next(
            (properties[field]["const"] for field in ("type", "op") if field in properties),
            model_name.removesuffix("Frame").lower(),
        )
        messages[name] = {"title": model_name, "payload": {"$ref": reference}}


def _action_payload(action: ActionSpec, request_schema: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(request_schema)
    payload["title"] = action.name
    payload["properties"]["action"]["const"] = action.name
    payload["properties"]["params"] = {"$ref": SCHEMA_REF.format(model=action.params.__name__)}
    if any(field.is_required() for field in action.params.model_fields.values()):
        payload["required"].append("params")
    return payload


def _topic_payload(spec: TopicSpec, schemas: dict[str, Any]) -> dict[str, Any]:
    fields = spec.payload.model_fields
    reference = {"$ref": SCHEMA_REF.format(model=spec.payload.__name__)}
    envelope: dict[str, Any] = deepcopy(schemas["EventFrame"])
    if {"source", "raw", "ts"} <= fields.keys():
        payload = schemas[spec.payload.__name__]
        envelope["properties"].update(payload["properties"])
        envelope["required"] = sorted(set(envelope["required"] + payload.get("required", [])))
    else:
        envelope["properties"]["raw"] = reference
    return envelope


def _operation(name: str, direction: str, title: str) -> dict[str, Any]:
    return {
        "action": direction,
        "channel": {"$ref": "#/channels/ws"},
        "title": title,
        "messages": [{"$ref": f"#/channels/ws/messages/{name}"}],
    }


def _operations() -> dict[str, Any]:
    operations: dict[str, Any] = {}
    for name, topic in TOPICS.items():
        slug = topic_slug(name)
        operations[f"{slug}.subscribe"] = _operation("subscribe", "receive", f"Subscribe to {name}")
        operations[f"{slug}.events"] = _operation(slug, topic.direction, f"{name} events")
    operations["request"] = _operation("request", "receive", "Correlated request")
    operations["response"] = _operation("response", "send", "Correlated response")
    for name, action in ACTIONS.items():
        response_name = f"{name}.response" if action.result is not None else "response"
        operations[name] = {
            **_operation(name, "receive", name),
            "description": action.description,
            "reply": {
                "channel": {"$ref": "#/channels/ws"},
                "messages": [{"$ref": f"#/channels/ws/messages/{response_name}"}],
            },
        }
    return operations
