from mandri.core.ids import ProviderKind

MODEL_REF_PREFIXES: dict[ProviderKind, str] = {
    ProviderKind.OPENROUTER: "openrouter/",
    ProviderKind.OPENCODE: "custom_openai/",
    ProviderKind.OPENCODE_GO: "custom_openai/",
    ProviderKind.OLLAMA: "ollama_chat/",
    ProviderKind.LM_STUDIO: "lm_studio/",
    ProviderKind.OPENAI: "openai/",
    ProviderKind.ANTHROPIC: "anthropic/",
    ProviderKind.GEMINI: "gemini/",
    ProviderKind.CUSTOM: "openai/",
}
