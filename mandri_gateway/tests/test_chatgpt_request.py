import json

import httpx
import pytest
from mandri.gateway.chatgpt_adapter import ChatGptAdapter


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize(
    "content", ["System context", [{"type": "input_text", "text": "System context"}]]
)
def test_system_messages_preserve_content_and_history_position(typed, content):
    system = {"role": "system", "content": content}
    if typed:
        system["type"] = "message"
    history = [
        system,
        {"role": "developer", "content": "Developer context"},
        {"role": "user", "content": "Synthetic question"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{}",
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "Synthetic result"},
        {"role": "system", "content": "Updated system context"},
        {"role": "assistant", "content": "Synthetic answer", "phase": "final_answer"},
    ]
    request = httpx.Request(
        "POST",
        "https://provider.invalid/v1/responses",
        json={"model": "gpt-6.1-sol", "instructions": "Existing instructions", "input": history},
    )

    converted = ChatGptAdapter("https://provider.invalid/v1").request(request)
    payload = json.loads(converted.content)

    assert payload["instructions"] == "Existing instructions"
    assert payload["input"] == [
        {**system, "role": "developer"},
        *history[1:5],
        {"role": "developer", "content": "Updated system context"},
        history[6],
    ]
    assert json.loads(request.content)["input"] == history
