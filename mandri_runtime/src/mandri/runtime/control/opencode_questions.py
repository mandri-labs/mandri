import json
import math
from typing import Any

from mandri.core.types.approvals import ApprovalRequest
from mandri.runtime.control.errors import ControlTransportError
from mandri.runtime.question_answers import valid_question_answers


def form_answers(form: dict[str, Any], answers: list[list[str]]) -> dict[str, Any]:
    fields = form.get("fields")
    if not isinstance(fields, list) or len(fields) != len(answers):
        raise ControlTransportError("Answer every native form field before submitting")
    result: dict[str, Any] = {}
    for field, values in zip(fields, answers, strict=True):
        key, kind = field["key"], field["type"]
        conditions = field.get("when", [])
        if any(
            condition["key"] not in result
            or (
                condition["value"] in result.get(condition["key"], [])
                if isinstance(result.get(condition["key"]), list)
                else result.get(condition["key"]) == condition["value"]
            )
            != (condition["op"] == "eq")
            for condition in conditions
        ):
            continue
        options = {item["label"]: item["value"] for item in field.get("options", [])}
        converted = [options.get(value, value) for value in values]
        if kind == "multiselect":
            result[key] = converted
            continue
        if len(converted) != 1:
            raise ControlTransportError("This native form field accepts one answer")
        value = converted[0]
        if kind == "boolean":
            if value not in {"true", "false"}:
                raise ControlTransportError("Native form requires a boolean answer")
            result[key] = value == "true"
        elif kind in {"number", "integer"}:
            try:
                result[key] = int(value) if kind == "integer" else float(value)
            except ValueError:
                raise ControlTransportError("Native form requires a numeric answer") from None
            if not math.isfinite(result[key]):
                raise ControlTransportError("Native form requires a finite numeric answer")
        elif kind == "string":
            result[key] = value
        else:
            raise ControlTransportError("This native form field cannot be answered here")
    return result


def question_answers(request: ApprovalRequest) -> list[list[str]]:
    if not valid_question_answers(request.answers):
        raise ControlTransportError("Native question requires non-empty answers")
    raw = json.loads(request.native_request)
    questions = raw.get("properties", {}).get("questions", [])
    if not isinstance(questions, list) or len(questions) != len(request.answers or []):
        raise ControlTransportError("Answer every native question before submitting")
    result: list[list[str]] = []
    for index, question in enumerate(questions):
        matches = [
            answer
            for answer in request.answers or []
            if answer["question"] in (str(index), question.get("question"))
        ]
        if len(matches) != 1:
            raise ControlTransportError("Native question answer is ambiguous")
        values = matches[0]["answers"]
        if not question.get("multiple", False) and len(values) != 1:
            raise ControlTransportError("This native question accepts one answer")
        options = [item.get("label") for item in question.get("options", [])]
        if question.get("custom") is False and any(value not in options for value in values):
            raise ControlTransportError("Choose one of the native question options")
        result.append(values)
    return result


def recovered_inputs(records: Any, event_type: str, session_id: str) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ControlTransportError("OpenCode returned invalid pending inputs")
    events = []
    for record in records:
        if not isinstance(record, dict):
            continue
        properties = record.get("properties", record)
        if isinstance(properties, dict) and properties.get("sessionID") == session_id:
            events.append({"type": event_type, "properties": properties})
    return events
