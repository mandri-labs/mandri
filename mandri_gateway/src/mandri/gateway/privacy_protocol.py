from collections.abc import Callable
from enum import StrEnum
from typing import Any

from mandri.core.types.execution import ProtectionError
from mandri.gateway.privacy_content import ContentVisitor
from mandri.gateway.surrogate import SurrogateEngine


class GatewayProtocol(StrEnum):
    CHAT = "chat"
    RESPONSES = "responses"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    COUNT_TOKENS = "count_tokens"


_TRANSPORT_OPTIONS = frozenset(
    {
        "api_base",
        "api_key",
        "base_url",
        "custom_llm_provider",
        "extra_headers",
        "headers",
        "client",
        "http_client",
        "http_session",
        "aclient_session",
        "client_session",
        "callbacks",
        "success_callback",
        "failure_callback",
        "logger_fn",
        "litellm_logging_obj",
        "proxy",
        "cache",
        "caching",
        "cache_key",
        "mock_response",
        "mock_tool_calls",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "vertex_credentials",
        "api_version",
        "privacy_scope_id",
        "privacy_mode",
        "execution_backend",
        "metadata_passthrough",
        "drop_params",
        "additional_drop_params",
        "allowed_openai_params",
        "num_retries",
    }
)


def validate_request(body: Any) -> None:
    if not isinstance(body, dict):
        raise ProtectionError("gateway_request_invalid", "The model request must be an object")
    if body.keys() & _TRANSPORT_OPTIONS:
        raise ProtectionError("gateway_request_invalid", "Transport overrides are not accepted")


def transform_content(value: Any, engine: SurrogateEngine, *, restore: bool = False) -> Any:
    if not restore:
        visit_content(value, engine.discover)
    transform = engine.restore if restore else engine.protect
    return visit_content(value, transform)


def visit_content(value: Any, transform: Callable[[Any], Any]) -> Any:
    return ContentVisitor(transform).root(value)
