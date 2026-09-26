type UsageRegistry = {
  getAvailable(): Array<{ provider: string }>;
  isUsingOAuth(model: { provider: string }): boolean;
  getApiKeyForProvider(provider: string): Promise<string | undefined>;
};

type UsageExtension = {
  on(event: "session_start", handler: (event: unknown, ctx: { modelRegistry: UsageRegistry; ui: { notify(message: string, type: string): void } }) => Promise<void>): void;
};

export default function usageExtension(pi: UsageExtension) {
  pi.on("session_start", async (_event, ctx) => {
    const providers = [...new Set(ctx.modelRegistry.getAvailable()
      .filter((model) => ctx.modelRegistry.isUsingOAuth(model))
      .map((model) => model.provider))];
    const accounts = await Promise.all(providers.map(async (provider) => {
      const url = provider === "openai-codex"
        ? "https://chatgpt.com/backend-api/wham/usage"
        : provider === "anthropic" ? "https://api.anthropic.com/api/oauth/usage" : undefined;
      if (!url) return { provider, status: "unsupported" };
      try {
        const token = await ctx.modelRegistry.getApiKeyForProvider(provider);
        if (!token) return { provider, status: "authentication_required" };
        const headers: Record<string, string> = { Authorization: `Bearer ${token}` };
        if (provider === "anthropic") headers["anthropic-beta"] = "oauth-2025-04-20";
        const response = await fetch(url, { headers, signal: AbortSignal.timeout(15000), redirect: "error" });
        if (!response.ok) return { provider, status: "unavailable" };
        return { provider, data: await response.json() };
      } catch {
        return { provider, status: "unavailable" };
      }
    }));
    ctx.ui.notify(JSON.stringify({ type: "mandri_usage", accounts }), "info");
  });
}
