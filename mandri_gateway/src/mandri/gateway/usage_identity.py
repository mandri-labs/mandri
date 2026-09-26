from mandri.core.ids import ProviderKind


def billing_mode(provider: ProviderKind) -> str:
    if provider in {ProviderKind.OLLAMA, ProviderKind.LM_STUDIO}:
        return "local"
    if provider is ProviderKind.OPENCODE_GO:
        return "subscription"
    return "api"
