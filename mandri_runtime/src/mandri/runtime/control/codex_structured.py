import json
from typing import Any

from jsonschema import Draft202012Validator
from mandri.core.types.approvals import ApprovalRequest
from mandri.runtime.control.errors import ControlError
from referencing import Registry


def response_object(request: ApprovalRequest) -> dict[str, Any]:
    try:
        result = json.loads(request.updated_input or "null")
    except (TypeError, ValueError) as error:
        raise ControlError("The native response is not valid JSON") from error
    if not isinstance(result, dict):
        raise ControlError("The native response must be an object")
    return result


def requested_params(request: ApprovalRequest) -> dict[str, Any]:
    native = json.loads(request.native_request)
    params = native.get("params")
    if not isinstance(params, dict):
        raise ControlError("The native request has invalid parameters")
    return params


def permission_grant(request: ApprovalRequest) -> dict[str, Any]:
    result = response_object(request)
    requested = requested_params(request).get("permissions")
    if (
        not isinstance(requested, dict)
        or result.get("permissions") != requested
        or result.get("scope") not in {"turn", "session"}
        or set(result) != {"permissions", "scope"}
    ):
        raise ControlError("Approve only the requested native permissions")
    return result


def elicitation_response(request: ApprovalRequest) -> dict[str, Any]:
    result = response_object(request)
    params = requested_params(request)
    if result.get("action") != "accept" or set(result) - {"action", "content"}:
        raise ControlError("The native elicitation requires an acceptance response")
    if params.get("mode") == "url":
        if result.get("content") is not None:
            raise ControlError("URL elicitation does not accept form content")
        return result
    schema = params.get("requestedSchema")
    content = result.get("content")
    if not isinstance(schema, dict) or not isinstance(content, dict):
        raise ControlError("The native elicitation requires form content")
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, registry=Registry()).validate(content)
    except Exception as error:
        raise ControlError("The response does not match the requested form") from error
    return result
