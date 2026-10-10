from types import SimpleNamespace

import pytest
from mandri.core.ids import ProviderKind
from mandri.gateway.output_budget import remaining_output, retry_kwargs
from mandri.gateway.provider_call import provider_call


class BudgetError(Exception):
    status_code = 400


@pytest.mark.parametrize(
    "message,remaining",
    [
        (
            "prompt (70000 tokens) + max tokens (16384) exceeds the context (81920); "
            "max_tokens (at most 11912 here)",
            11912,
        ),
        ("maximum context length is 32768 tokens, including 30000 tokens in the messages", 2768),
        ("input: 64000 tokens exceeds context window: 32768", None),
        ("invalid temperature", None),
        ("max_tokens must be at most 1024", 1024),
    ],
)
def test_budget_fallback_uses_counts_and_reserved_space_reported_by_the_provider(
    message, remaining
):
    assert remaining_output(BudgetError(message)) == remaining


@pytest.mark.parametrize(
    "kwargs,key",
    [
        ({"messages": [], "max_tokens": 16384}, "max_tokens"),
        ({"input": [], "max_output_tokens": 16384}, "max_output_tokens"),
        ({"messages": [], "max_completion_tokens": 16384}, "max_completion_tokens"),
    ],
)
def test_retry_only_changes_the_output_budget(kwargs, key):
    error = BudgetError("prompt (70000 tokens) exceeds context (81920)")
    assert retry_kwargs(kwargs, error) == {**kwargs, key: 11920}
    assert kwargs[key] == 16384
    assert retry_kwargs({**kwargs, key: 8192}, error) is None


async def test_rejected_gemini_request_retries_with_the_reported_budget_and_same_prompt():
    seen = []
    kwargs = {
        "contents": [{"role": "user", "parts": [{"text": "fixture"}]}],
        "config": {"maxOutputTokens": 16384, "thinkingConfig": {"includeThoughts": True}},
        "tools": [{"fixture": True}],
    }

    async def call(**request):
        seen.append(request)
        if len(seen) == 1:
            raise BudgetError(
                "prompt (70000 tokens) exceeds context (81920); max_tokens (at most 11912 here)"
            )
        return "complete"

    route = SimpleNamespace(model=SimpleNamespace(provider=ProviderKind.CUSTOM))
    assert await provider_call(route, call, kwargs, None) == "complete"
    assert len(seen) == 2
    assert seen[1] == {**kwargs, "config": {**kwargs["config"], "maxOutputTokens": 11912}}
    assert kwargs["config"]["maxOutputTokens"] == 16384


async def test_budget_recovery_stops_after_one_rejected_retry():
    seen = []

    async def call(**request):
        seen.append(request)
        raise BudgetError("prompt (70000 tokens) exceeds context (81920)")

    route = SimpleNamespace(model=SimpleNamespace(provider=ProviderKind.CUSTOM))
    with pytest.raises(BudgetError):
        await provider_call(route, call, {"messages": [], "max_tokens": 16384}, None)
    assert len(seen) == 2
