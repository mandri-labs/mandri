import uuid

from mandri.core.ids import ProviderKind

_SESSION_NAMESPACE = uuid.UUID("30c828d1-6fda-48cc-bb41-b606a06b0042")


def conversation_headers(kind: ProviderKind, context: str) -> dict[str, str]:
    if kind not in (ProviderKind.OPENCODE, ProviderKind.OPENCODE_GO):
        return {}
    return {
        "User-Agent": "mandri/0.1",
        "x-opencode-session": str(uuid.uuid5(_SESSION_NAMESPACE, context)),
    }


def inference_headers(kind: ProviderKind, context: str) -> dict[str, str]:
    headers = conversation_headers(kind, context)
    if not headers:
        return {}
    return {
        **headers,
        "User-Agent": "Mandri Gateway",
        "x-opencode-client": "mandri",
        "x-opencode-session-id": headers["x-opencode-session"],
        "x-opencode-request": str(uuid.uuid4()),
    }
