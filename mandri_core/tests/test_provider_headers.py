from uuid import UUID

import pytest
from mandri.core.ids import ProviderKind
from mandri.core.provider_headers import conversation_headers, inference_headers


@pytest.mark.parametrize("kind", [ProviderKind.OPENCODE, ProviderKind.OPENCODE_GO])
def test_inference_has_stable_session_and_unique_request(kind):
    first = inference_headers(kind, "session:one")
    second = inference_headers(kind, "session:one")
    other = inference_headers(kind, "session:two")
    first_request = first.pop("x-opencode-request")
    second_request = second.pop("x-opencode-request")
    assert UUID(first_request).version == 4
    assert UUID(second_request).version == 4
    assert first_request != second_request
    assert first == second
    assert first["x-opencode-session"] != other["x-opencode-session"]
    assert first["x-opencode-session-id"] == first["x-opencode-session"]
    assert first["User-Agent"] == "Mandri Gateway"
    assert first["x-opencode-client"] == "mandri"
    metadata = conversation_headers(kind, "session:one")
    assert metadata["x-opencode-session"] == first["x-opencode-session"]
    assert metadata["User-Agent"] == "mandri/0.1"
    assert "x-opencode-request" not in metadata


@pytest.mark.parametrize(
    "kind",
    [
        kind
        for kind in ProviderKind
        if kind not in {ProviderKind.OPENCODE, ProviderKind.OPENCODE_GO}
    ],
)
def test_other_providers_have_no_opencode_headers(kind):
    assert inference_headers(kind, "session:one") == {}
