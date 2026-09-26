export default function (pi: any) {
  const baseUrl = process.env.MANDRI_PI_BASE_URL;
  const apiKey = process.env.MANDRI_API_KEY;
  const metadata = process.env.MANDRI_PI_MODEL;
  if (!baseUrl || !apiKey || !metadata) {
    throw new Error("Mandri gateway configuration is incomplete");
  }
  pi.registerProvider("mandri", {
    baseUrl,
    apiKey,
    api: "openai-completions",
    models: [JSON.parse(metadata)],
  });
}
