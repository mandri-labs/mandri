import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  const servers = JSON.parse(process.env.MANDRI_MCP_SERVERS ?? "[]") as {
    name: string;
    url: string;
  }[];
  const token = process.env.MANDRI_MCP_TOKEN;
  if (!token) return;
  for (const server of servers) {
    pi.registerMcpServer(server.name, {
      url: server.url,
      headers: { Authorization: `Bearer ${token}` },
    });
  }
}
