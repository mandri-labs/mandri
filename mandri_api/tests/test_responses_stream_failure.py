import json

from mandri.api.routers.gateway import _responses_stream_all
from mandri.core.ids import ModelRef, ProviderKind, Url
from mandri.gateway.errors.upstream import UpstreamError
from mandri.gateway.types.model import Model


async def test_upstream_overload_terminates_responses_stream_with_native_failure_event():
    async def upstream():
        yield {"type": "response.created", "response": {"id": "resp_fixture", "output": []}}
        raise UpstreamError(502, "Service temporarily overloaded", ProviderKind.CUSTOM)

    model = Model(
        ProviderKind.CUSTOM, ModelRef("fixture/model"), Url("https://provider.invalid"), None
    )
    frames = [frame async for frame in _responses_stream_all(upstream(), model)]
    assert frames[-1].startswith(b"event: response.failed\n")
    payload = json.loads(frames[-1].split(b"data: ", 1)[1])
    assert payload["response"]["id"] == "resp_fixture"
    assert payload["response"]["status"] == "failed"
    assert payload["response"]["error"] == {
        "code": "server_error",
        "message": "Service temporarily overloaded",
    }
    assert (
        payload["sequence_number"] > json.loads(frames[0].split(b"data: ", 1)[1])["sequence_number"]
    )
