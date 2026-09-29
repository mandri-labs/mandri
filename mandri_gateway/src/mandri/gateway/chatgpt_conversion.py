from types import SimpleNamespace
from typing import Any, cast

from litellm import ModelResponse
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)
from litellm.types.llms.openai import ResponsesAPIResponse

_BRIDGE = cast(Any, LiteLLMResponsesTransformationHandler)()


def chat_request(body: dict[str, Any]) -> dict[str, Any]:
    logging: Any = SimpleNamespace()
    result = _BRIDGE.transform_request(
        model=body["model"],
        messages=body.get("messages", []),
        optional_params={
            key: value for key, value in body.items() if key not in {"model", "messages"}
        },
        litellm_params={},
        headers={},
        litellm_logging_obj=logging,
    )
    return {
        key: value for key, value in result.items() if key not in {"client", "litellm_logging_obj"}
    }


def chat_response(body: dict[str, Any]) -> dict[str, Any]:
    logging: Any = SimpleNamespace()
    response = _BRIDGE.transform_response(
        model=body["model"],
        raw_response=ResponsesAPIResponse.model_validate(body),
        model_response=ModelResponse(id=body["id"], created=body.get("created_at", 0)),
        logging_obj=logging,
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    return dict(response.model_dump(mode="json", exclude_none=True))
