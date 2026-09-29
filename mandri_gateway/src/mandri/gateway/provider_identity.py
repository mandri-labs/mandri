from mandri.core.ids import ProviderKind
from mandri.gateway.types.model import Model
from mandri.providers.chatgpt.identity import request_headers


def identity_headers(model: Model) -> dict[str, str]:
    if model.provider is not ProviderKind.CHATGPT:
        return {}
    return {
        name: value
        for name, value in request_headers(str(model.api_key)).items()
        if name.lower() != "authorization"
    }
