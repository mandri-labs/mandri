export default function registerPermissions(pi: any) {
  const permissionModes = new Set(["default", "acceptEdits", "plan", "bypassPermissions"]);
  let permissionMode = process.env.MANDRI_PI_PERMISSION_MODE ?? "default";
  if (!permissionModes.has(permissionMode)) throw new Error("Invalid Mandri permission mode");
  pi.registerCommand("mandri-permissions", {
    description: "Change permissions for this managed session",
    handler: async (args: string, ctx: any) => {
      const [mode, token] = args.trim().split(/\s+/);
      if (!permissionModes.has(mode) || !token) throw new Error("Invalid permission mode request");
      permissionMode = mode;
      ctx.ui.setStatus("_mandri_permissions", `${token}:${mode}`);
    },
  });
  pi.on("tool_call", async (event: any, ctx: any) => {
    if (["read", "grep", "find", "ls"].includes(event.toolName)) return;
    if (permissionMode === "plan") return { block: true, reason: "Plan mode permits read tools only" };
    if (permissionMode === "bypassPermissions") return;
    if (permissionMode === "acceptEdits" && ["edit", "write"].includes(event.toolName)) return;
    if (!ctx.hasUI || !await ctx.ui.confirm(
      `Allow ${event.toolName}?`, JSON.stringify(event.input),
    )) return { block: true, reason: "Tool execution was not approved" };
  });
}
