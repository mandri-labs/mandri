"""Direct litellm adapter handlers, one per wire format. No Router, per-call creds."""

import logging
from typing import Any

import httpx
import litellm
import litellm.anthropic_interface
from mandri.core.ids import ProviderKind
from mandri.core.provider_headers import inference_headers
from mandri.core.types.execution import PrivacyMode, ProtectionError
from mandri.gateway.errors.upstream import UpstreamError
from mandri.gateway.gemini_completion import generate_content
from mandri.gateway.gemini_request import normalize_contents
from mandri.gateway.privacy_count import protected_count_tokens
from mandri.gateway.privacy_egress import EgressGuard, provider_base
from mandri.gateway.prompt_trace import PromptTrace
from mandri.gateway.provider_call import provider_call
from mandri.gateway.provider_identity import identity_headers
from mandri.gateway.ratelimit_headers import from_exception
from mandri.gateway.reasoning_transport import apply_reasoning_transport
from mandri.gateway.responses_input import normalize_tool_results
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.types.model import Model
from mandri.gateway.unauthenticated_client import unauthenticated_client
from openai import omit

_CONNECT_TIMEOUT_SECONDS = 30.0
_IDLE_TIMEOUT_SECONDS = 300.0
_CALL_TIMEOUT_SECONDS = 600.0
logger = logging.getLogger(__name__)
_STREAM_TIMEOUT = httpx.Timeout(
    connect=_CONNECT_TIMEOUT_SECONDS,
    read=_IDLE_TIMEOUT_SECONDS,
    write=_CONNECT_TIMEOUT_SECONDS,
    pool=_CONNECT_TIMEOUT_SECONDS,
)
_OPENAI_REASONING_KEYS = frozenset({"reasoning", "reasoning_effort"})
_ANTHROPIC_REASONING_KEYS = frozenset({"thinking", "output_config"})


def upstream_error(error: Exception, model: Model) -> UpstreamError:
    status = getattr(error, "status_code", None)
    if not isinstance(status, int):
        status = 502
    raw_provider = getattr(error, "llm_provider", None)
    try:
        provider = ProviderKind(str(raw_provider))
    except ValueError:
        provider = ProviderKind.CUSTOM
    message = getattr(error, "message", None)
    if not isinstance(message, str) or not message:
        message = str(error)
    return UpstreamError(status, _redact(model, message), provider, headers=from_exception(error))


def _redact(model: Model, message: str) -> str:
    key = str(model.api_key)
    if key:
        return message.replace(key, "[redacted]")
    return message


def _credentials(route: ResolvedRoute, guard: EgressGuard | None = None) -> dict[str, Any]:
    if route.privacy_mode is PrivacyMode.SURROGATE and guard is None:
        raise ProtectionError(
            "privacy_unavailable", "Protected requests require privacy validation"
        )
    model = route.model
    credentials: dict[str, Any] = {"api_key": str(model.api_key)}
    if guard is not None:
        credentials["api_base"] = provider_base(route)
    elif model.api_base:
        credentials["api_base"] = str(model.api_base)
    if (
        not str(model.api_key)
        and model.provider in {ProviderKind.CUSTOM, ProviderKind.LM_STUDIO}
        and credentials.get("api_base")
    ):
        credentials["client"] = unauthenticated_client(credentials["api_base"])
    context = (
        f"session:{route.conversation_id}"
        if route.conversation_id is not None
        else f"route:{route.route_id}"
    )
    headers: dict[str, Any] = {
        **inference_headers(model.provider, context),
        **identity_headers(model),
    }
    if "client" in credentials:
        headers["Authorization"] = omit
    if headers:
        credentials["extra_headers"] = headers
    return credentials


class OpenAIHandler:
    def __init__(self) -> None:
        self._prompts = PromptTrace()

    async def chat_completions(
        self, route: ResolvedRoute, body: dict[str, Any], *, guard: EgressGuard | None = None
    ) -> Any:
        self._prompts.observe(route.conversation_id or str(route.route_id), body)
        payload = {key: value for key, value in body.items() if key != "model"}
        if route.reasoning_effort:
            payload = {
                key: value for key, value in payload.items() if key not in _OPENAI_REASONING_KEYS
            }
            payload["reasoning_effort"] = route.reasoning_effort
        apply_reasoning_transport(str(route.model.model_ref), payload, responses=False)
        stream = bool(payload.pop("stream", False))
        call_kwargs: dict[str, Any] = {
            "model": str(route.model.model_ref),
            "drop_params": True,
            "num_retries": 0,
            "timeout": _STREAM_TIMEOUT if stream else _CALL_TIMEOUT_SECONDS,
            "stream": stream,
            "allowed_openai_params": ["reasoning_effort"],
            **payload,
            **_credentials(route, guard),
            "additional_drop_params": ["web_search_options"],
        }
        if stream:
            call_kwargs["stream_options"] = {"include_usage": True}
        try:
            return await provider_call(route, litellm.acompletion, call_kwargs, guard)
        except (ProtectionError, UpstreamError):
            raise
        except Exception as error:
            raise upstream_error(error, route.model) from error


class ResponsesHandler:
    def __init__(self) -> None:
        self._prompts = PromptTrace()

    _PASSTHROUGH_PARAMS = frozenset(
        {
            "instructions",
            "max_output_tokens",
            "metadata",
            "parallel_tool_calls",
            "previous_response_id",
            "reasoning",
            "store",
            "temperature",
            "text",
            "tool_choice",
            "tools",
            "top_p",
            "truncation",
            "user",
            "include",
            "service_tier",
            "prompt_cache_key",
            "prompt_cache_retention",
        }
    )

    async def responses(
        self, route: ResolvedRoute, body: dict[str, Any], *, guard: EgressGuard | None = None
    ) -> Any:
        self._prompts.observe(route.conversation_id or str(route.route_id), body)
        payload = {key: value for key, value in body.items() if key in self._PASSTHROUGH_PARAMS}
        stream = bool(body.get("stream", False))
        input_value = body.get("input")
        use_chat_completions = route.model.provider not in {
            ProviderKind.OPENAI,
            ProviderKind.OPENROUTER,
            ProviderKind.CHATGPT,
        }
        if use_chat_completions:
            input_value = normalize_tool_results(input_value)
        governed_effort = route.reasoning_effort
        if governed_effort:
            payload.pop("reasoning", None)
            payload["reasoning"] = {"effort": governed_effort}
        else:
            reasoning = payload.get("reasoning")
            effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
            if isinstance(effort, str) and effort:
                payload["reasoning"] = {"effort": effort}
            else:
                payload.pop("reasoning", None)
        apply_reasoning_transport(
            str(route.model.model_ref),
            payload,
            responses=True,
            native_responses=not use_chat_completions,
        )
        call_kwargs: dict[str, Any] = {
            "model": str(route.model.model_ref),
            "input": input_value,
            "stream": stream,
            "use_chat_completions_api": use_chat_completions,
            "allowed_openai_params": ["reasoning_effort"],
            "num_retries": 0,
            "timeout": _STREAM_TIMEOUT if stream else _CALL_TIMEOUT_SECONDS,
            **_credentials(route, guard),
            **payload,
        }
        client_metadata = body.get("client_metadata")
        extra_body: dict[str, Any] = dict(call_kwargs.get("extra_body") or {})
        if isinstance(client_metadata, dict):
            extra_body["client_metadata"] = client_metadata
        if use_chat_completions and "prompt_cache_retention" in payload:
            extra_body["prompt_cache_retention"] = payload["prompt_cache_retention"]
        if use_chat_completions and stream:
            extra_body["stream_options"] = {"include_usage": True}
        if extra_body:
            call_kwargs["extra_body"] = extra_body
        try:
            return await provider_call(route, litellm.aresponses, call_kwargs, guard)
        except (ProtectionError, UpstreamError):
            raise
        except Exception as error:
            raise upstream_error(error, route.model) from error


class AnthropicHandler:
    def __init__(self) -> None:
        self._prompts = PromptTrace()

    async def messages(
        self, route: ResolvedRoute, body: dict[str, Any], *, guard: EgressGuard | None = None
    ) -> Any:
        self._prompts.observe(route.conversation_id or str(route.route_id), body)
        payload = {key: value for key, value in body.items() if key != "model"}
        if route.reasoning_effort:
            payload = {
                key: value for key, value in payload.items() if key not in _ANTHROPIC_REASONING_KEYS
            }
            payload["output_config"] = {"effort": route.reasoning_effort}
        stream = bool(payload.pop("stream", False))
        safeguards = payload.pop("safeguards", None)
        if safeguards is not None:
            payload["extra_body"] = {**payload.get("extra_body", {}), "safeguards": safeguards}
        model_ref = str(route.model.model_ref)
        if route.model.provider is ProviderKind.CUSTOM:
            model_ref = "custom_openai/" + model_ref.removeprefix("openai/")
        try:
            kwargs = {
                "model": model_ref,
                "drop_params": True,
                "num_retries": 0,
                "timeout": _STREAM_TIMEOUT if stream else _CALL_TIMEOUT_SECONDS,
                "stream": stream,
                **payload,
                **_credentials(route, guard),
                "additional_drop_params": ["web_search_options"],
            }
            return await provider_call(
                route, litellm.anthropic_interface.messages.acreate, kwargs, guard
            )
        except (ProtectionError, UpstreamError):
            raise
        except Exception as error:
            raise upstream_error(error, route.model) from error

    async def count_tokens(
        self, route: ResolvedRoute, body: dict[str, Any], *, guard: EgressGuard | None = None
    ) -> Any:
        try:
            if guard is not None:
                return await protected_count_tokens(guard, body)
            credentials = _credentials(route)
            credentials.pop("extra_headers", None)
            return await litellm.acount_tokens(
                model=str(route.model.model_ref),
                messages=body.get("messages"),
                tools=body.get("tools"),
                system=body.get("system"),
                **credentials,
            )
        except (ProtectionError, UpstreamError):
            raise
        except Exception as error:
            raise upstream_error(error, route.model) from error


class GeminiHandler:
    def __init__(self) -> None:
        self._prompts = PromptTrace()

    async def generate_content(
        self,
        route: ResolvedRoute,
        body: dict[str, Any],
        stream: bool,
        *,
        guard: EgressGuard | None = None,
    ) -> Any:
        self._prompts.observe(route.conversation_id or str(route.route_id), body)
        try:
            kwargs = {
                "stream": stream,
                "model": str(route.model.model_ref),
                "contents": normalize_contents(body.get("contents")),
                "systemInstruction": body.get("systemInstruction"),
                "config": body.get("generationConfig"),
                "tools": body.get("tools"),
                **_credentials(route, guard),
                "drop_params": True,
                "additional_drop_params": ["web_search_options"],
                "num_retries": 0,
                "timeout": _STREAM_TIMEOUT if stream else _CALL_TIMEOUT_SECONDS,
            }
            return await provider_call(route, generate_content, kwargs, guard)
        except (ProtectionError, UpstreamError):
            raise
        except Exception as error:
            raise upstream_error(error, route.model) from error
