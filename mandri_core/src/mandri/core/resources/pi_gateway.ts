import { getModels, getProviders } from "@earendil-works/pi-ai/compat";

export default function (pi: any) {
  const baseUrl = process.env.MANDRI_PI_BASE_URL;
  const apiKey = process.env.MANDRI_API_KEY;
  const metadata = process.env.MANDRI_PI_MODEL;
  if (!baseUrl || !apiKey || !metadata) {
    throw new Error("Mandri gateway configuration is incomplete");
  }
  const supplied = JSON.parse(metadata);
  const models = getProviders().flatMap((provider) => getModels(provider));
  const matched = models.filter((model) =>
    supplied.name === model.id || supplied.name.endsWith(`/${model.id}`),
  );
  const native = matched.find((model) => model.api === "openai-completions")
    ?? models.find((model) => model.api === "openai-completions");
  const resolved = { ...native, ...supplied, api: "openai-completions", baseUrl };
  delete resolved.headers;
  delete resolved.compat;
  if (resolved.contextWindow && !supplied.maxTokens && native?.maxTokens) {
    resolved.maxTokens = Math.min(native.maxTokens, resolved.contextWindow);
  }
  pi.registerProvider("mandri", {
    baseUrl,
    apiKey,
    api: "openai-completions",
    models: [resolved],
  });
}
