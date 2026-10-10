import json

import httpx
import pytest
from mandri.gateway.chat_usage_transport import normalize_chat_usage, normalize_frame


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


@pytest.mark.parametrize("delimiter", [b"\n", b"\r\n", b"\r"])
@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
async def test_combined_finish_and_usage_survive_arbitrary_wire_boundaries(delimiter, chunk_size):
    payload = {
        "id": "fixture",
        "choices": [{"index": 0, "delta": {"content": "café"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 70000,
            "completion_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 69000},
        },
    }
    raw = (
        b"id: fixture"
        + delimiter
        + b"data: "
        + json.dumps(payload, ensure_ascii=False).encode()
        + delimiter * 2
    )
    raw += b"data: [DONE]" + delimiter * 2
    response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://upstream.invalid/v1/chat/completions"),
        headers={"content-type": "text/event-stream"},
        stream=Chunks(
            [raw[index : index + chunk_size] for index in range(0, len(raw), chunk_size)]
        ),
    )
    normalize_chat_usage(response)
    output = b"".join([chunk async for chunk in response.aiter_bytes()])
    frames = [
        json.loads(line[6:]) for line in output.decode().splitlines() if line.startswith("data: {")
    ]
    assert len(frames) == 2
    assert frames[0]["choices"] == payload["choices"]
    assert "usage" not in frames[0]
    assert frames[1]["choices"] == []
    assert frames[1]["usage"] == payload["usage"]
    assert output.count(b"id: fixture") == 2
    assert b"data: [DONE]" in output


@pytest.mark.parametrize(
    "frame",
    [
        b"data: [DONE]\n\n",
        b": keepalive\n\n",
        b"data: invalid\n\n",
        b'data: {"usage":{}}\n\n',
        b'data: {"choices":[]}\n\n',
        b"data: \xff\n\n",
    ],
)
def test_unrelated_or_unreadable_frames_are_preserved(frame):
    assert normalize_frame(frame) == frame


async def test_final_frame_without_a_delimiter_is_retained():
    raw = b'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":17}}'
    response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://upstream.invalid/chat/completions"),
        headers={"content-type": "text/event-stream"},
        stream=Chunks([raw]),
    )
    normalize_chat_usage(response)
    output = b"".join([chunk async for chunk in response.aiter_bytes()])
    assert b'"prompt_tokens":17' in output
